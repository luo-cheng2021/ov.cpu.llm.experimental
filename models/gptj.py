from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
import numpy as np
import sys, os
import argparse

ext_path = "./custom_ops/build/libov-cpu-llm-experimental.so"
custom_opset = opset_utils._get_node_factory()
custom_opset.add_extension(ext_path)

def show_model(m):
    print('inputs of the model:')
    for port, _input in enumerate(m.inputs):
        print('	[{}] {}'.format(port, _input))
    print('outputs of the model:')
    for port, _output in enumerate(m.outputs):
        print('	[{}] {}'.format(port, _output))

def create_attn(q, k, v, kv_cache, beam_table, attn_mask, cos_tab, sin_tab, layer_idx, rotary_dim, n_hidden, n_head, name):
    mha_attr = {'arg_q': 0,
                'arg_k': 1,
                'arg_v': 2,
                'arg_kv_cache': 3,
                'arg_beam_table': 4,
                'arg_attn_mask': 5,
                'arg_cos': 6,
                'arg_sin': 7,
                'layer_id': layer_idx,
                'rotary_dim': rotary_dim,
                'n_hidden': n_hidden,
                'n_head': n_head}

    output = custom_opset.create('MultiHeadAttention', 
        [q, k, v, kv_cache, beam_table, attn_mask, cos_tab, sin_tab], mha_attr)
    output.set_friendly_name(name)
    return output

def layer(configs, consts, layer_idx, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab):
    def make_fc(key, input):
        weights = opset.constant(consts[f'{key}.weight'][layer_idx], Type.f32, name=f'{key}.weight.layer{layer_idx}')
        matmul = opset.matmul(input, weights, transpose_a=False, transpose_b=True, name=f'{key}.matmul.layer{layer_idx}')
        if consts[f'{key}.bias'][layer_idx] is not None:
            bias = opset.constant(consts[f'{key}.bias'][layer_idx], Type.f32, name=f'{key}.bias.layer{layer_idx}')
            matmul = opset.add(matmul, bias, auto_broadcast='numpy', name=f'{key}.add.layer{layer_idx}')
        return matmul
    key = 'transformer.h.ln_1'
    # layerNorm operation
    input_layernorm_bias = opset.constant(consts[f'{key}.bias'][layer_idx], Type.f32, name=f'{key}.bias.layer{layer_idx}')
    input_layernorm_weight = opset.constant(consts[f'{key}.weight'][layer_idx], Type.f32, name=f'{key}.weight.layer{layer_idx}')
    input_layernorm_mvn = opset.mvn(hidden_states, axes=[-1], normalize_variance=True, eps=configs['layer_norm_eps'], eps_mode="inside_sqrt", name=f'{key}.mvn.layer{layer_idx}')
    input_layernorm_mul = opset.multiply(input_layernorm_mvn, input_layernorm_weight, auto_broadcast='numpy', name=f'{key}.mul.layer{layer_idx}')
    input_layernorm = opset.add(input_layernorm_mul, input_layernorm_bias, auto_broadcast='numpy', name=f'{key}.add.layer{layer_idx}')

    q = make_fc('transformer.h.attn.q_proj', input_layernorm)
    k = make_fc('transformer.h.attn.k_proj', input_layernorm)
    v = make_fc('transformer.h.attn.v_proj', input_layernorm)

    # custom op
    attn_output = create_attn(q, k, v, kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
                              layer_idx, configs['rotary_dim'], configs['hidden_size'], configs['head_num'],
                              name=f'transformer.h.attn.layer{layer_idx}')

    attn_output = make_fc('transformer.h.attn.out_proj', attn_output)

    # mlp
    def mlp(states):
        dense_h_to_4h = make_fc('transformer.h.mlp.fc_in', states)
        gelu = opset.gelu(dense_h_to_4h, approximation_mode=configs['gelu_mode'], name=f'transformer.h.mlp.gelu.layer{layer_idx}')
        dense_4h_to_h = make_fc('transformer.h.mlp.fc_out', gelu)
        return dense_4h_to_h

    mlp_output = mlp(input_layernorm)
    # residual connection.
    output = opset.add(
        opset.add(attn_output, mlp_output, auto_broadcast="numpy", name=f'transformer.h.add0.layer{layer_idx}'),
        hidden_states, auto_broadcast='numpy', name=f'transformer.h.add1.layer{layer_idx}')
    return output

def create_model(configs, consts):
    print(f'start generate ov model...')
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i32, name='input_ids')
    # [2 * n_layers, batch, n_head, max_kv_len, head_size]
    kv_cache = opset.parameter([2 * configs['layer_num'], -1, configs['head_num'], -1, configs['head_size']], Type.f32, name='kv_cache')
    # [batch, max_kv_len]
    beam_table = opset.parameter([-1, -1], Type.i32, name='beam_table')
    # [batch, query_len+past_len]
    attn_mask = opset.parameter([-1, -1], Type.f32, name='attn_mask')
    # [max_kv_len, rotary_dim//2]
    cos_tab = opset.parameter([-1, configs['rotary_dim'] // 2], Type.f32, name='cos_tab')
    sin_tab = opset.parameter([-1, configs['rotary_dim'] // 2], Type.f32, name='sin_tab')

    key = 'transformer.wte.weight'
    embed_in_const = opset.constant(consts[key], Type.f32, name=key)
    inputs_embeds = opset.gather(embed_in_const, indices=input_ids, axis=0)
    hidden_states = inputs_embeds

    for i in range(configs['layer_num']):
        hidden_states = layer(configs, consts, i, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab)
    # final_layernorm
    key = 'transformer.ln_f'
    final_layernorm_bias = opset.constant(consts[f'{key}.bias'], Type.f32)
    final_layernorm_weight = opset.constant(consts[f'{key}.weight'], Type.f32)
    final_layer_norm_mvn = opset.mvn(hidden_states, axes=[-1], normalize_variance=True, eps=configs['layer_norm_eps'], eps_mode="inside_sqrt", name=f'{key}.mvn')
    final_layer_norm_mul = opset.multiply(final_layer_norm_mvn, final_layernorm_weight, auto_broadcast='numpy', name=f'{key}.mul')
    final_layernorm = opset.add(final_layer_norm_mul, final_layernorm_bias, auto_broadcast='numpy', name=f'{key}.add')
    # embed_out
    key = 'logits'
    embed_out_weight = opset.constant(consts['lm_head.weight'], Type.f32)
    embed_out_ = opset.matmul(final_layernorm, embed_out_weight, transpose_a=False,transpose_b=True, name=f'{key}.matmul')
    embed_out_bias = opset.constant(consts['lm_head.bias'], Type.f32)
    embed_out = opset.add(embed_out_, embed_out_bias, auto_broadcast='numpy', name=f'{key}.add')
    embed_out_result = opset.result(embed_out, name=f'{key}')
    print(f'generate ov model done.')
    return Model([embed_out_result],
                 [input_ids, kv_cache, beam_table, attn_mask, cos_tab, sin_tab])

def get_params_from_model(path):
    print(f'extracting from model "{path}"...')
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).to('cpu').eval()
    assert(model.config.rotary == True)

    configs = {
        'layer_num': model.config.n_layer,
        'head_num': model.config.n_head,
        'head_size': model.config.n_embd // model.config.n_head,
        'hidden_size': model.config.n_embd,
        'layer_norm_eps': model.config.layer_norm_epsilon,
        'max_position_embeddings': model.config.n_positions,
        'rotary_dim': model.config.rotary_dim,
        'gelu_mode': 'erf' if model.config.activation_function == 'gelu_new' else 'tanh',
    }
    consts = {
        'transformer.wte.weight': model.transformer.wte.weight.detach().numpy(),
        'lm_head.weight': model.lm_head.weight.detach().numpy(),
        'lm_head.bias': model.lm_head.bias.detach().numpy(),
        'transformer.ln_f.bias': model.transformer.ln_f.bias.detach().numpy(),
        'transformer.ln_f.weight': model.transformer.ln_f.weight.detach().numpy(),
        'transformer.h.ln_1.bias': [
            l.ln_1.bias.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.ln_1.weight': [
            l.ln_1.weight.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.q_proj.bias': [
            None if l.attn.q_proj.bias is None else l.attn.q_proj.bias.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.q_proj.weight': [
            l.attn.q_proj.weight.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.k_proj.bias': [
            None if l.attn.k_proj.bias is None else l.attn.k_proj.bias.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.k_proj.weight': [
            l.attn.k_proj.weight.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.v_proj.bias': [
            None if l.attn.v_proj.bias is None else l.attn.v_proj.bias.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.v_proj.weight': [
            l.attn.v_proj.weight.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.out_proj.bias': [
            None if l.attn.out_proj.bias is None else l.attn.out_proj.bias.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.attn.out_proj.weight': [
            l.attn.out_proj.weight.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.mlp.fc_in.bias': [
            l.mlp.fc_in.bias.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.mlp.fc_in.weight': [
            l.mlp.fc_in.weight.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.mlp.fc_out.bias': [
            l.mlp.fc_out.bias.detach().numpy() for l in model.transformer.h
        ],
        'transformer.h.mlp.fc_out.weight': [
            l.mlp.fc_out.weight.detach().numpy() for l in model.transformer.h
        ],
    }
    print(f'extracting done. model configs: {configs}')
    return configs, consts

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('org_model_path', type=str, nargs='?', default='/home/llm_irs/pytorch_frontend_models/gpt-j-6b/pytorch_original/')
    parser.add_argument('ov_model_path', type=str, nargs='?', default='./gen/gptj_6b.xml')
    args = parser.parse_args()

    configs, consts = get_params_from_model(args.org_model_path)
    model = create_model(configs, consts)
    show_model(model)
    print(f'serialize ov model to "{args.ov_model_path}"...')
    serialize(model, args.ov_model_path)
    print('serialize done.')
