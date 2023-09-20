from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
import numpy as np
import sys, os
import argparse
import time
from utils import show_model, make_mha, make_fc, make_mvn, make_embedding, save_tokenzier, OV_XML_FILE_NAME, configs as make_configs

def layer(configs, consts, layer_idx, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'gpt_neox.layers.attention'
    # layerNorm operation
    input_layernorm = make_mvn('gpt_neox.layers.input_layernorm', hidden_states, consts['layers'][layer_idx], configs, name_suffix)

    qkv = make_fc('gpt_neox.layers.attention.query_key_value', input_layernorm, consts['layers'][layer_idx], name_suffix)

    # custom op
    attn_output = make_mha([qkv], kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
                           layer_idx, configs['rotary_dims'], configs['hidden_size'], configs['head_num'],
                           name=f'{name_prefix}.mha{name_suffix}')

    attn_output = make_fc('gpt_neox.layers.attention.dense', attn_output, consts['layers'][layer_idx], name_suffix)

    post_attention_layernorm = make_mvn('gpt_neox.layers.post_attention_layernorm', hidden_states, consts['layers'][layer_idx], configs, name_suffix)

    # mlp
    def mlp(states):
        dense_h_to_4h = make_fc('gpt_neox.layers.mlp.dense_h_to_4h', states, consts['layers'][layer_idx], name_suffix)
        gelu = opset.gelu(dense_h_to_4h, approximation_mode=configs['gelu_mode'], name=f'{name_prefix}.mlp.gelu{name_suffix}')
        dense_4h_to_h = make_fc('gpt_neox.layers.mlp.dense_4h_to_h', gelu, consts['layers'][layer_idx], name_suffix)
        return dense_4h_to_h

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = opset.add(
        opset.add(mlp_output, attn_output, auto_broadcast="numpy", name=f'{name_prefix}.add0{name_suffix}'),
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

    inputs_embeds = make_embedding('gpt_neox.embed_in.weight', input_ids, consts)
    hidden_states = inputs_embeds

    for i in range(configs['layer_num']):
        hidden_states = layer(configs, consts, i, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab)
    # final_layernorm
    final_layernorm = make_mvn('gpt_neox.final_layer_norm', hidden_states, consts, configs)
    # embed_out
    embed_out = make_fc('embed_out', final_layernorm, consts)
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

    assert(model.config.use_parallel_residual == True)
    assert(model.config.hidden_act in ['gelu_new', 'gelu'])
    configs = {
        'layer_num': model.config.num_hidden_layers,
        'head_num': model.config.num_attention_heads,
        'head_size': model.config.hidden_size // model.config.num_attention_heads,
        'hidden_size': model.config.hidden_size,
        'layer_norm_eps': model.config.layer_norm_eps,
        'max_position_embeddings': model.config.max_position_embeddings,
        'rotary_dims': int(model.config.hidden_size // model.config.num_attention_heads * model.config.rotary_pct),
        'gelu_mode': 'erf' if model.config.hidden_act == 'gelu_new' else 'tanh',
    }

    consts = {
        'gpt_neox.embed_in.weight': model.gpt_neox.embed_in.weight.detach().numpy(),
        'gpt_neox.final_layer_norm.bias': model.gpt_neox.final_layer_norm.bias.detach().numpy(),
        'gpt_neox.final_layer_norm.weight': model.gpt_neox.final_layer_norm.weight.detach().numpy(),
        'embed_out.weight': model.embed_out.weight.detach().numpy(),
        'embed_out.bias': model.embed_out.bias.detach().numpy() if model.embed_out.bias is not None else None,
        'layers': [
            {
                'gpt_neox.layers.input_layernorm.bias': l.input_layernorm.bias.detach().numpy(),
                'gpt_neox.layers.input_layernorm.weight': l.input_layernorm.weight.detach().numpy(),
                'gpt_neox.layers.post_attention_layernorm.bias': l.post_attention_layernorm.bias.detach().numpy(),
                'gpt_neox.layers.post_attention_layernorm.weight': l.post_attention_layernorm.weight.detach().numpy(),
                'gpt_neox.layers.attention.query_key_value.bias': l.attention.query_key_value.bias.detach().numpy(),
                'gpt_neox.layers.attention.query_key_value.weight': l.attention.query_key_value.weight.detach().numpy(),
                'gpt_neox.layers.attention.dense.bias': l.attention.dense.bias.detach().numpy(),
                'gpt_neox.layers.attention.dense.weight': l.attention.dense.weight.detach().numpy(),
                'gpt_neox.layers.mlp.dense_h_to_4h.bias': l.mlp.dense_h_to_4h.bias.detach().numpy(),
                'gpt_neox.layers.mlp.dense_h_to_4h.weight': l.mlp.dense_h_to_4h.weight.detach().numpy(),
                'gpt_neox.layers.mlp.dense_4h_to_h.bias': l.mlp.dense_4h_to_h.bias.detach().numpy(),
                'gpt_neox.layers.mlp.dense_4h_to_h.weight': l.mlp.dense_4h_to_h.weight.detach().numpy()
            } for l in model.gpt_neox.layers
        ],
    }
    cost = time.time() - beg
    print(f'extracting done, cost {cost:.2f} seconds.\nmodel configs:')
    for k, v in configs.items():
        print(f'	{k}: {v}')
    return configs, consts

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--org_model_path', type=str, nargs='?', default='/home/llm_irs/pytorch_frontend_models/dolly-v2-12b/pytorch_original/')
    parser.add_argument('--ov_model_path', type=str, nargs='?', default='./gen/dolly_v2_12b/')
    parser.add_argument('--compressed_weight', type=bool, nargs='?', default=False)
    args = parser.parse_args()
    make_configs['compressed_weight'] = args.compressed_weight

    configs, consts = get_params_from_model(args.org_model_path)
    model = create_model(configs, consts)
    show_model(model)
    print(f'serialize ov model to "{args.ov_model_path}"...')
    beg = time.time()
    serialize(model, os.path.join(args.ov_model_path, OV_XML_FILE_NAME))
    cost = time.time() - beg
    print(f'serialize done, cost {cost:.2f} seconds.')
    print(f'save tokenzier to "{args.ov_model_path}" ...')
    save_tokenzier(args.org_model_path, args.ov_model_path)
