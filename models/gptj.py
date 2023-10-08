from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
import numpy as np
import sys, os
import argparse
import time
from utils import show_model, make_mha, make_fc, make_mvn, make_embedding, save_tokenizer, OV_XML_FILE_NAME, configs as make_configs
from tqdm import tqdm

def layer(configs, consts, layer_idx, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'transformer.h'
    # layerNorm operation
    input_layernorm = make_mvn('transformer.h.ln_1', hidden_states, consts['layers'][layer_idx], configs, name_suffix)

    q = make_fc('transformer.h.attn.q_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    k = make_fc('transformer.h.attn.k_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)
    v = make_fc('transformer.h.attn.v_proj', input_layernorm, consts['layers'][layer_idx], name_suffix)

    # custom op
    attn_output = make_mha([q, k, v], kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
                           layer_idx, configs['rotary_dims'], configs['hidden_size'], configs['head_num'],
                           name=f'{name_prefix}.mha{name_suffix}', rope_type='original')

    attn_output = make_fc('transformer.h.attn.out_proj', attn_output, consts['layers'][layer_idx], name_suffix)

    # mlp
    def mlp(states):
        dense_h_to_4h = make_fc('transformer.h.mlp.fc_in', states, consts['layers'][layer_idx], name_suffix)
        gelu = opset.gelu(dense_h_to_4h, approximation_mode=configs['gelu_mode'], name=f'{name_prefix}.mlp.gelu{name_suffix}')
        dense_4h_to_h = make_fc('transformer.h.mlp.fc_out', gelu, consts['layers'][layer_idx], name_suffix)
        return dense_4h_to_h

    mlp_output = mlp(input_layernorm)
    # residual connection.
    output = opset.add(
        opset.add(attn_output, mlp_output, auto_broadcast="numpy", name=f'{name_prefix}.add0{name_suffix}'),
        hidden_states, auto_broadcast='numpy', name=f'{name_prefix}.add1{name_suffix}')
    return output

def create_model(configs, consts):
    print(f'start generate ov model...')
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i32, name='input_ids')
    # [2 * n_layers, batch, n_head, max_kv_len, head_size]
    kv_cache = opset.parameter([2 * configs['layer_num'], -1, configs['head_num'], -1, configs['head_size']], Type.f32, name='kv_cache')
    # [batch, max_kv_len]
    beam_table = opset.parameter([-1, -1], Type.i32, name='beam_table')
    # [batch, query_len+past_len]
    attn_mask = opset.parameter([-1, -1], Type.f32, name='attn_mask')
    # [max_kv_len, rotary_dims//2]
    cos_tab = opset.parameter([-1, configs['rotary_dims'] // 2], Type.f32, name='cos_tab')
    sin_tab = opset.parameter([-1, configs['rotary_dims'] // 2], Type.f32, name='sin_tab')

    inputs_embeds = make_embedding('transformer.wte.weight', input_ids, consts)
    hidden_states = inputs_embeds

    for i in tqdm(range(configs['layer_num'])):
        hidden_states = layer(configs, consts, i, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab)
    # final_layernorm
    final_layernorm = make_mvn('transformer.ln_f', hidden_states, consts, configs)
    # embed_out
    embed_out = make_fc('lm_head', final_layernorm, consts)
    embed_out_result = opset.result(embed_out, name='logits')
    cost = time.time() - beg
    print(f'generate ov model done, cost {cost:.2f} seconds.')
    return Model([embed_out_result],
                 [input_ids, kv_cache, beam_table, attn_mask, cos_tab, sin_tab])

def get_params_from_model(path):
    print(f'extracting from model "{path}"...')
    beg = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to('cpu').eval()
    assert(model.config.rotary == True)
    assert(model.config.activation_function in ['gelu_new', 'gelu'])

    configs = {
        'layer_num': model.config.n_layer,
        'head_num': model.config.n_head,
        'head_size': model.config.n_embd // model.config.n_head,
        'hidden_size': model.config.n_embd,
        'layer_norm_eps': model.config.layer_norm_epsilon,
        'max_position_embeddings': model.config.n_positions,
        'rotary_dims': model.config.rotary_dim,
        'gelu_mode': 'erf' if model.config.activation_function == 'gelu_new' else 'tanh',
    }
    consts = {
        'transformer.wte.weight': model.transformer.wte.weight.detach().numpy(),
        'transformer.ln_f.bias': model.transformer.ln_f.bias.detach().numpy(),
        'transformer.ln_f.weight': model.transformer.ln_f.weight.detach().numpy(),
        'lm_head.weight': model.lm_head.weight.detach().numpy(),
        'lm_head.bias': model.lm_head.bias.detach().numpy(),
        'layers': [
            {
                'transformer.h.ln_1.bias': l.ln_1.bias.detach().numpy(),
                'transformer.h.ln_1.weight': l.ln_1.weight.detach().numpy(),
                'transformer.h.attn.q_proj.bias': None if l.attn.q_proj.bias is None else l.attn.q_proj.bias.detach().numpy(),
                'transformer.h.attn.q_proj.weight': l.attn.q_proj.weight.detach().numpy(),
                'transformer.h.attn.k_proj.bias': None if l.attn.k_proj.bias is None else l.attn.k_proj.bias.detach().numpy(),
                'transformer.h.attn.k_proj.weight': l.attn.k_proj.weight.detach().numpy(),
                'transformer.h.attn.v_proj.bias': None if l.attn.v_proj.bias is None else l.attn.v_proj.bias.detach().numpy(),
                'transformer.h.attn.v_proj.weight': l.attn.v_proj.weight.detach().numpy(),
                'transformer.h.attn.out_proj.bias': None if l.attn.out_proj.bias is None else l.attn.out_proj.bias.detach().numpy(),
                'transformer.h.attn.out_proj.weight': l.attn.out_proj.weight.detach().numpy(),
                'transformer.h.mlp.fc_in.bias': l.mlp.fc_in.bias.detach().numpy(),
                'transformer.h.mlp.fc_in.weight': l.mlp.fc_in.weight.detach().numpy(),
                'transformer.h.mlp.fc_out.bias': l.mlp.fc_out.bias.detach().numpy(),
                'transformer.h.mlp.fc_out.weight': l.mlp.fc_out.weight.detach().numpy()
            } for l in model.transformer.h
        ],
    }
    cost = time.time() - beg
    print(f'extracting done, cost {cost:.2f} seconds.\nmodel configs:')
    for k, v in configs.items():
        print(f'	{k}: {v}')
    return configs, consts

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--org_model_path', type=str, nargs='?', default='/home/llm_irs/pytorch_frontend_models/gpt-j-6b/pytorch_original/')
    parser.add_argument('--ov_model_path', type=str, nargs='?', default='./gen/gptj_6b/')
    parser.add_argument('--compressed_weight', type=bool, nargs='?', default=False)
    parser.add_argument('--quant_type', type=str, nargs='?', default='')
    args = parser.parse_args()
    # for compatible, will remove
    if args.compressed_weight:
        print(f'warning: please use "--quant=nncf_w8" instead.')
        if args.quant_type:
            raise ValueError('compressed_weight and quant_type can not be set at the same time.')
        args.quant_type = 'nncf_w8'
    make_configs['quant_type'] = args.quant_type

    if args.quant_type:
        args.ov_model_path = os.path.join(args.ov_model_path, args.quant_type)
    os.makedirs(args.ov_model_path, exist_ok=True)

    configs, consts = get_params_from_model(args.org_model_path)
    model = create_model(configs, consts)
    show_model(model)
    print(f'serialize ov model to "{args.ov_model_path}"...')
    beg = time.time()
    serialize(model, args.ov_model_path)
    cost = time.time() - beg
    print(f'serialize done, cost {cost:.2f} seconds.')
    print(f'save tokenzier to "{args.ov_model_path}" ...')
    save_tokenzier(args.org_model_path, args.ov_model_path)
