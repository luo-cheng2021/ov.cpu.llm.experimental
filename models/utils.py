from transformers import AutoTokenizer
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
from openvino.runtime.op import Constant
import numpy as np
import os
import sys
import torch

OV_XML_FILE_NAME="openvino.xml"

ext_path = None
if sys.platform == 'win32':
    ext_path = ".\\custom_ops\\build\\Release\\ov-cpu-llm-experimental.dll"
elif sys.platform == 'linux':
    ext_path = "./custom_ops/build/libov-cpu-llm-experimental.so"
else:
    print(f"Sample code not supported on platform: {sys.platform}")
    exit(1)

custom_opset = opset_utils._get_node_factory()
custom_opset.add_extension(ext_path)

configs = {
    'quant_type': 'nncf_w8',        # valid: '', 'nncf_w8', 'llama_w8_0',
    'llama_quant_type': 'group',    # only llama_xx support: 'tensor', 'channel', 'group'
    'llama_group_k': 32,
    'llama_group_n': 32,
}

# copy from nncf/torch/quantization/weights_compression.py: _insert_pre_compression_operations
def _compress_weight_nncf(weight_np, bits:int = 8):
    def get_scale_zp_from_input_low_input_high(level_low, level_high, input_low, input_high):
        y_scale = (input_high - input_low) / (level_high - level_low)
        y_zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

        type_ = torch.int8 if level_low < 0 else torch.uint8
        level_low *= torch.ones_like(y_zero_point).to(type_)
        level_high *= torch.ones_like(y_zero_point).to(type_)
        level_low = level_low.to(y_zero_point.device)
        level_high = level_high.to(y_zero_point.device)
        y_zero_point = torch.min(torch.max(level_low, torch.round(y_zero_point).to(type_)), level_high)

        y_scale = torch.squeeze(y_scale)
        y_zero_point = torch.squeeze(y_zero_point)
        return y_scale, y_zero_point
    
    level_high = 2**bits - 1

    assert level_high < 256
    weight = torch.from_numpy(weight_np)

    target_dim = 0 # layer.target_weight_dim_for_compression
    stat_dim = (target_dim + 1) % 2
    input_low = torch.min(weight, dim=stat_dim).values.detach()
    input_high = torch.max(weight, dim=stat_dim).values.detach()
    scale, zero_point = get_scale_zp_from_input_low_input_high(0, level_high, input_low, input_high)

    scale = scale.unsqueeze(stat_dim)
    zero_point = zero_point.unsqueeze(stat_dim)

    compressed_weight = weight.data / scale + zero_point
    compressed_weight = torch.clamp(torch.round(compressed_weight), 0, level_high)

    return compressed_weight.type(dtype=torch.uint8).numpy(), zero_point.numpy(), scale.numpy()

def _make_compressed_weight_nncf(weight, key, bits:int = 8):
    compressed_weight, zero_point, scale = _compress_weight_nncf(weight, bits)
    weight_node = Constant(compressed_weight, True)
    zp_node = Constant(zero_point, True)
    scale_node = Constant(scale, True)
    weight_node.set_friendly_name(f'{key}.weight')
    zp_node.set_friendly_name(f'{key}.weight.zp')
    scale_node.set_friendly_name(f'{key}.weight.scale')
    weight_node = opset.convert(weight_node, 'f32', name=f'{key}.weight.convert')
    zp_node = opset.convert(zp_node, 'f32', name=f'{key}.weight.zp.convert')
    sub = opset.subtract(weight_node, zp_node, name=f'{key}.weight.sub')
    scale = opset.multiply(sub, scale_node, name=f'{key}.weight.mul')
    return scale

def pt_as_np(t):
    if t is not None: return t.detach().numpy().astype(np.float32)
    return None

def show_model(m):
    print('inputs of the model:')
    for port, _input in enumerate(m.inputs):
        print('	[{}] {}'.format(port, _input))
    print('outputs of the model:')
    for port, _output in enumerate(m.outputs):
        print('	[{}] {}'.format(port, _output))

def make_mha(qkvs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
             layer_idx, rotary_dim, n_hidden, n_head, name, num_kv_heads=0, rope_type='modified', multi_query_is_planar=False):
    qkvs_len = len(qkvs)
    mha_attr = {'arg_kv_cache': qkvs_len,
                'arg_beam_table': qkvs_len + 1,
                'arg_attn_mask': qkvs_len + 2,
                'arg_cos': qkvs_len + 3,
                'arg_sin': qkvs_len + 4,
                'layer_id': layer_idx,
                'rotary_dims': rotary_dim,
                'n_hidden': n_hidden,
                'n_head': n_head,
                'num_kv_heads': num_kv_heads,
                'multi_query_is_planar': multi_query_is_planar,
                'rope_type': ['original', 'modified'].index(rope_type)}

    if qkvs_len == 1:
        mha_attr['arg_q'] = 0
        mha_attr['arg_k'] = 0
        mha_attr['arg_v'] = 0
    else:
        mha_attr['arg_q'] = 0
        mha_attr['arg_k'] = 1
        mha_attr['arg_v'] = 2

    output = custom_opset.create('MultiHeadAttention', 
        [*qkvs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab], mha_attr)
    output.set_friendly_name(name)
    return output

# custom FC
def make_experimental_fc(input, weight, name):
    quant_type = configs['quant_type']

    if 'w8' in quant_type:
        weight_bits = 8
    elif 'w4' in quant_type:
        weight_bits = 4
    else:
        raise ValueError(f'invalid quant_type: {quant_type}, only w8, w4 support')

    fc_attr = {}
    fc_attr['quant_type'] = ['', 'nncf_w8', 'llama_w8_0'].index(quant_type)
    fc_attr['llama_quant_type'] = ['', 'tensor', 'channel', 'group'].index(configs['llama_quant_type'])
    fc_attr['llama_group_k'] = configs['llama_group_k']
    fc_attr['llama_group_n'] = configs['llama_group_n']
    fc_attr['N'] = weight.shape[0]
    fc_attr['K'] = weight.shape[1]
    fc_attr['bits'] = weight_bits
    fc_attr['evaluate_qweight'] = 1

    # build a FC node in `evaluate_qweight` mode to quantize & relayout weight
    # runtime FC node is built based on them
    weight_node = Constant(weight, True)
    qweight_node = custom_opset.create('FC', [weight_node], fc_attr)

    # print(weight.shape, weight.dtype)
    output_vec = []
    for i in range(qweight_node.get_output_size()):
        ov_type = qweight_node.get_output_element_type(i)
        ov_shape = qweight_node.get_output_shape(i)
        # print(f" output {i} : {ov_type}, {ov_shape}")
        output_vec.append(Tensor(ov_type, ov_shape))

    if not qweight_node.evaluate(output_vec, [Tensor(weight)]):
        raise Exception("weight quantization failed!")

    # create actual FC node based on quantized weights
    fc_inputs = [input]
    for t in output_vec:
        fc_inputs.append(Constant(t))
    
    fc_attr['evaluate_qweight'] = 0
    fc_node = custom_opset.create('FC', fc_inputs, fc_attr)
    # print(output_vec)
    return fc_node

def make_fc(key, input, consts, name_suffix=''):
    if 'llama' in configs['quant_type']:
        matmul = make_experimental_fc(input, consts[f'{key}.weight'], key)
    else:
        if configs['quant_type'] == 'nncf_w8':
            weights = _make_compressed_weight_nncf(consts[f'{key}.weight'], key)
        else:
            weights = Constant(consts[f'{key}.weight'], True)
            weights.set_friendly_name(name=f'{key}.weight{name_suffix}')
        matmul = opset.matmul(input, weights, transpose_a=False, transpose_b=True, name=f'{key}.matmul{name_suffix}')
    if consts[f'{key}.bias'] is not None:
        bias = Constant(consts[f'{key}.bias'], True)
        bias.set_friendly_name(name=f'{key}.bias{name_suffix}')
        matmul = opset.add(matmul, bias, auto_broadcast='numpy', name=f'{key}.add{name_suffix}')
    return matmul

def make_mvn(key, input, consts, configs, name_suffix=''):
    mvn = opset.mvn(input, axes=[-1], normalize_variance=True, eps=configs['layer_norm_eps'], eps_mode="inside_sqrt", name=f'{key}.mvn{name_suffix}')
    if consts[f'{key}.weight'] is not None:
        weights = opset.constant(consts[f'{key}.weight'], Type.f32, name=f'{key}.weight{name_suffix}')
        mvn = opset.multiply(mvn, weights, auto_broadcast='numpy', name=f'{key}.mul{name_suffix}')
    if consts[f'{key}.bias'] is not None:
        bias = opset.constant(consts[f'{key}.bias'], Type.f32, name=f'{key}.bias{name_suffix}')
        mvn = opset.add(mvn, bias, auto_broadcast='numpy', name=f'{key}.add{name_suffix}')
    return mvn

def make_rms_norm(key, input, consts, epsilon, name_suffix=''):
    weights = opset.constant(consts[f'{key}.weight'], Type.f32, name=f'{key}.weight{name_suffix}')
    pow = opset.multiply(input, input, name=f'{key}.pow{name_suffix}')
    #pow = opset.power(input, np.array([2], np.float32), name=f'{key}.pow{name_suffix}')
    variance = opset.reduce_mean(pow, reduction_axes=[-1], keep_dims=True, name=f'{key}.var{name_suffix}')
    add = opset.add(variance, opset.constant(epsilon, Type.f32), name=f'{key}.add{name_suffix}')
    sqrt = opset.sqrt(add, name=f'{key}.sqrt{name_suffix}')
    div = opset.divide(input, sqrt, name=f'{key}.div{name_suffix}')
    mul = opset.multiply(div, weights, auto_broadcast='numpy', name=f'{key}.mul{name_suffix}')
    return mul

def make_embedding(key, input, consts):
    if configs['quant_type'] != '':
        embed_in_const = _make_compressed_weight_nncf(consts[key], key)
    else:
        embed_in_const = Constant(consts[key], True)
        embed_in_const.set_friendly_name(name=key)
    inputs_embeds = opset.gather(embed_in_const, indices=input, axis=0)
    return inputs_embeds

def save_tokenzier(orig_model_path, ov_model_path):
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path)
    tokenizer.save_pretrained(ov_model_path)
