from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
import numpy as np
import sys, os
import argparse
import time
from utils import show_model, make_mha, make_fc, pt_as_np, make_rms_norm, make_embedding, save_tokenzier, OV_XML_FILE_NAME, configs as make_configs
from tqdm import tqdm

def layer(configs, consts, layer_idx, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab):
    name_suffix = f'.layer{layer_idx}'
    name_prefix = 'transformer.layers.self_attention'
    # layerNorm operation
    input_layernorm = make_rms_norm('transformer.layers.input_layernorm', hidden_states, consts['layers'][layer_idx], configs['layernorm_epsilon'], name_suffix)

    qkv = make_fc('transformer.layers.self_attention.query_key_value', input_layernorm, consts['layers'][layer_idx], name_suffix)

    # custom op
    attn_output = make_mha([qkv], kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
                           layer_idx, configs['rotary_dims'], configs['hidden_size'], configs['head_num'],
                           name=f'{name_prefix}.mha{name_suffix}', num_kv_heads=configs['num_kv_heads'], rope_type='original', multi_query_is_planar=True)

    attn_output = make_fc('transformer.layers.self_attention.dense', attn_output, consts['layers'][layer_idx], name_suffix)

    attn_output = opset.add(hidden_states, attn_output, auto_broadcast='numpy', name=f'{name_prefix}.add0{name_suffix}')
    post_attention_layernorm = make_rms_norm('transformer.layers.post_attention_layernorm', attn_output, consts['layers'][layer_idx], configs['layernorm_epsilon'], name_suffix)

    # mlp
    def mlp(states):
        dense_h_to_4h = make_fc('transformer.layers.mlp.dense_h_to_4h', states, consts['layers'][layer_idx], name_suffix)
        nodes = opset.split(dense_h_to_4h, axis=-1, num_splits=2, name=f'{name_prefix}.mlp.split{name_suffix}')
        silu = opset.swish(nodes.output(0), name=f'{name_prefix}.mlp.silu{name_suffix}')
        mul = opset.multiply(silu, nodes.output(1), auto_broadcast='numpy', name=f'{name_prefix}.mlp.mul{name_suffix}')
        dense_4h_to_h = make_fc('transformer.layers.mlp.dense_4h_to_h', mul, consts['layers'][layer_idx], name_suffix)
        return dense_4h_to_h

    mlp_output = mlp(post_attention_layernorm)
    # residual connection.
    output = opset.add(attn_output, mlp_output, auto_broadcast='numpy', name=f'{name_prefix}.add1{name_suffix}')
    return output

def create_model(configs, consts):
    print(f'start generate ov model...')
    beg = time.time()
    # [batch, query_len]
    input_ids = opset.parameter([-1, -1], Type.i32, name='input_ids')
    # [2 * n_layers, batch, num_kv_heads, max_kv_len, head_size]
    kv_cache = opset.parameter([2 * configs['layer_num'], -1, configs['num_kv_heads'], -1, configs['head_size']], Type.f32, name='kv_cache')
    # [batch, max_kv_len]
    beam_table = opset.parameter([-1, -1], Type.i32, name='beam_table')
    # [batch, query_len+past_len]
    attn_mask = opset.parameter([-1, -1], Type.f32, name='attn_mask')
    # [max_kv_len, rotary_dims//2]
    cos_tab = opset.parameter([-1, configs['rotary_dims'] // 2], Type.f32, name='cos_tab')
    sin_tab = opset.parameter([-1, configs['rotary_dims'] // 2], Type.f32, name='sin_tab')

    inputs_embeds = make_embedding('transformer.embedding.word_embeddings.weight', input_ids, consts)
    hidden_states = inputs_embeds

    for i in tqdm(range(configs['layer_num'])):
        hidden_states = layer(configs, consts, i, hidden_states, kv_cache, beam_table, attn_mask, cos_tab, sin_tab)
    # final_layernorm
    final_layernorm = make_rms_norm('transformer.encoder.final_layernorm', hidden_states, consts, configs['layernorm_epsilon'])
    # embed_out
    embed_out = make_fc('transformer.output_layer', final_layernorm, consts)
    embed_out_result = opset.result(embed_out, name='logits')
    cost = time.time() - beg
    print(f'generate ov model done, cost {cost:.2f} seconds.')
    return Model([embed_out_result],
                 [input_ids, kv_cache, beam_table, attn_mask, cos_tab, sin_tab])

def get_params_from_model(path):
    print(f'extracting from model "{path}"...')
    beg = time.time()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(path, trust_remote_code=True).to('cpu').eval()

    assert(model.config.add_bias_linear == False)
    assert(model.config.add_qkv_bias == True)
    assert(model.config.apply_query_key_layer_scaling == True)
    assert(model.config.apply_residual_connection_post_layernorm == False)
    assert(model.config.bias_dropout_fusion == True)
    assert(model.config.multi_query_attention == True)
    assert(model.config.original_rope == True)
    assert(model.config.post_layer_norm == True)
    assert(model.config.rmsnorm == True)

    configs = {
        'layer_num': model.config.num_layers,
        'head_num': model.config.num_attention_heads,
        'head_size': model.config.hidden_size // model.config.num_attention_heads,
        'hidden_size': model.config.hidden_size,
        'max_position_embeddings': model.config.seq_length,
        'rotary_dims': model.config.kv_channels // 2, #int(model.config.hidden_size // model.config.num_attention_heads),
        #'gelu_mode': model.config.hidden_act,
        #'intermediate_size': model.config.intermediate_size,
        #'num_key_value_heads': model.config.num_key_value_heads,
        'ffn_hidden_size': model.config.ffn_hidden_size,
        'kv_channels': model.config.kv_channels,
        'layernorm_epsilon': model.config.layernorm_epsilon,
        'num_kv_heads': model.config.multi_query_group_num,
    }

    consts = {
        'transformer.embedding.word_embeddings.weight': pt_as_np(model.transformer.embedding.word_embeddings.weight),
        'transformer.encoder.final_layernorm.weight': pt_as_np(model.transformer.encoder.final_layernorm.weight),
        'transformer.output_layer.weight': pt_as_np(model.transformer.output_layer.weight),
        'transformer.output_layer.bias': pt_as_np(model.transformer.output_layer.bias),
        'layers': [
            {
                'transformer.layers.input_layernorm.weight': pt_as_np(l.input_layernorm.weight),
                'transformer.layers.post_attention_layernorm.weight': pt_as_np(l.post_attention_layernorm.weight),
                'transformer.layers.self_attention.query_key_value.bias': pt_as_np(l.self_attention.query_key_value.bias),
                'transformer.layers.self_attention.query_key_value.weight': pt_as_np(l.self_attention.query_key_value.weight),
                'transformer.layers.self_attention.dense.bias': pt_as_np(l.self_attention.dense.bias),
                'transformer.layers.self_attention.dense.weight': pt_as_np(l.self_attention.dense.weight),
                'transformer.layers.mlp.dense_h_to_4h.bias': pt_as_np(l.mlp.dense_h_to_4h.bias),
                'transformer.layers.mlp.dense_h_to_4h.weight': pt_as_np(l.mlp.dense_h_to_4h.weight),
                'transformer.layers.mlp.dense_4h_to_h.bias': pt_as_np(l.mlp.dense_4h_to_h.bias),
                'transformer.layers.mlp.dense_4h_to_h.weight': pt_as_np(l.mlp.dense_4h_to_h.weight)
            } for l in model.transformer.encoder.layers
        ],
    }
    cost = time.time() - beg
    print(f'extracting done, cost {cost:.2f} seconds.\nmodel configs:')
    for k, v in configs.items():
        print(f'	{k}: {v}')
    return configs, consts

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--org_model_path', type=str, nargs='?', default='/home/llm_irs/pytorch_frontend_models/chatglm2-6b/')
    parser.add_argument('--ov_model_path', type=str, nargs='?', default='./gen/chatglm2-6b/')
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
    serialize(model, os.path.join(args.ov_model_path, OV_XML_FILE_NAME))
    cost = time.time() - beg
    print(f'serialize done, cost {cost:.2f} seconds.')
    print(f'save tokenzier to "{args.ov_model_path}" ...')
    save_tokenzier(args.org_model_path, args.ov_model_path)
