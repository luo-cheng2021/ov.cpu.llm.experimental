from transformers import AutoTokenizer
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils, Node, Output, Extension
from openvino.runtime import opset10 as opset
from openvino.runtime.op import Constant
from openvino.runtime.op import _MultiHeadAttention as mha
import numpy as np
import os
import sys
import torch
from typing import Any, Dict, List, Optional, Union
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import binary_op, nameable_op, unary_op
from functools import partial
from openvino.runtime.utils.types import NodeInput, as_node, as_nodes

_get_node_factory_opset4 = partial(_get_node_factory, "opset4")

OV_XML_FILE_NAME="openvino_model.xml"

configs = {
    'quant_type': 'nncf_w8',        # valid: '', 'nncf_w8', 'llama_w8_0',
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

def _arguments_as_outputs(arguments: List[Union[Node, Output]]) -> List[Output]:
    outputs = []
    for argument in arguments:
        if issubclass(type(argument), Output):
            outputs.append(argument)
        else:
            outputs.extend(argument.outputs())
    return outputs

def make_mha(qkvs, kv_cache, beam_table, attn_mask, cos_tab, sin_tab,
             layer_idx, rotary_dim, n_hidden, n_head, name, num_kv_heads=0, rope_type='modified', multi_query_is_planar=False):
    mha_attr = {'layer_id': layer_idx,
                'rotary_dims': rotary_dim,
                'n_hidden': n_hidden,
                'n_head': n_head,
                'num_kv_heads': num_kv_heads,
                'multi_query_is_planar': multi_query_is_planar,
                'rope_type': ['original', 'modified'].index(rope_type)}

    output = mha(_arguments_as_outputs([kv_cache, beam_table, attn_mask, cos_tab, sin_tab, *qkvs]), mha_attr)
    output.set_friendly_name(name)
    return output

# custom FC
def make_experimental_fc(input, weight, name):
    quant_type = configs['quant_type']

    def quantize_weights(weight, quant_type):
        try:
            # build a FC node in `evaluate_qweight` mode to quantize & relayout weight
            qweight_node = custom_opset.create('FC', [Constant(weight, True)], {
                'quant_type':quant_type,
                'N' : weight.shape[0],
                'K' : weight.shape[1],
                'evaluate_qweight' : 1
            })
        except RuntimeError:
            # unsupported quant type
            return []

        # unsupported quant type
        if qweight_node.get_output_size() == 0:
            return []

        # create tensors with required shape & dtype to hold quantized weights
        output_vec = []
        for i in range(qweight_node.get_output_size()):
            ov_type = qweight_node.get_output_element_type(i)
            ov_shape = qweight_node.get_output_shape(i)
            output_vec.append(Tensor(ov_type, ov_shape))

        # evaluate_qweight
        if not qweight_node.evaluate(output_vec, [Tensor(weight)]):
            raise Exception("weight quantization failed!")

        return [Constant(w) for w in output_vec]

    quantized_weights_list = quantize_weights(weight, quant_type)

    if len(quantized_weights_list) == 0:
        return None

    return custom_opset.create('FC', [input, *quantized_weights_list] , {
        'quant_type':quant_type,
        'N' : weight.shape[0],
        'K' : weight.shape[1],
        'evaluate_qweight' : 0
    })

def make_fc(key, input, consts, name_suffix=''):
    # weight const f32 NxK
    weight = consts[f'{key}.weight']

    # ngraph requires c_contiguous numpy array as shared-mem constant
    if not weight.data.c_contiguous:
        weight = np.ascontiguousarray(weight)
    # fallbacks
    if True:
        if configs['quant_type'] == 'nncf_w8':
            weights = _make_compressed_weight_nncf(weight, key)
        elif configs['quant_type'] == '':
            weights = Constant(weight, True)
            weights.set_friendly_name(name=f'{key}.weight{name_suffix}')
        elif configs['quant_type'] == 'f16':
            weight_f16 = weight.astype(np.float16)
            weight_node = Constant(weight_f16, True)
            weight_node.set_friendly_name(name=f'{key}.weight{name_suffix}')
            weights = opset.convert(weight_node, 'f32', name=f'{key}.weight{name_suffix}.convert')
        else:
            raise Exception(f"Unknown quant type: {configs['quant_type']}")
        matmul = opset.matmul(input, weights, transpose_a=False, transpose_b=True, name=f'{key}.matmul{name_suffix}')

    # add bias
    if consts[f'{key}.bias'] is not None:
        bias = Constant(consts[f'{key}.bias'], True)
        bias.set_friendly_name(name=f'{key}.bias{name_suffix}')
        matmul = opset.add(matmul, bias, auto_broadcast='numpy', name=f'{key}.add{name_suffix}')

    return matmul

def make_combined_fc(keys, input, consts, name_suffix=''):
    weight_list = [consts[f'{key}.weight'] for key in keys]
    new_weight = np.concatenate(weight_list, axis=0)
    new_key = f'{keys[0]}kv.weight'
    if configs['quant_type'] == 'nncf_w8':
        weights = _make_compressed_weight_nncf(new_weight, new_key)
    elif configs['quant_type'] == '':
        weights = Constant(new_weight, True)
        weights.set_friendly_name(name=f'{new_key}.weight{name_suffix}')
    elif configs['quant_type'] == 'f16':
        weight_f16 = new_weight.astype(np.float16)
        weight_node = Constant(weight_f16, True)
        weight_node.set_friendly_name(name=f'{new_key}.weight{name_suffix}')
        weights = opset.convert(weight_node, 'f32', name=f'{new_key}.weight{name_suffix}.convert')
    else:
        raise Exception(f"Unknown quant type: {configs['quant_type']}")
    matmul = opset.matmul(input, weights, transpose_a=False, transpose_b=True, name=f'{new_key}.matmul{name_suffix}')

    # add bias
    for key in keys:
        assert consts[f'{key}.bias'] is None, 'there should be no bias in combined fc mode'

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
    #pow = opset.multiply(input, input, name=f'{key}.pow{name_suffix}')
    pow = opset.power(input, np.array([2], np.float32), name=f'{key}.pow{name_suffix}')
    variance = opset.reduce_mean(pow, reduction_axes=[-1], keep_dims=True, name=f'{key}.var{name_suffix}')
    add = opset.add(variance, opset.constant(epsilon, Type.f32), name=f'{key}.add{name_suffix}')
    sqrt = opset.sqrt(add, name=f'{key}.sqrt{name_suffix}')
    div = opset.power(sqrt, np.array([-1], np.float32))
    mul = opset.multiply(input, div, auto_broadcast='numpy', name=f'{key}.mul{name_suffix}')
    #div = opset.divide(input, sqrt, name=f'{key}.div{name_suffix}')
    mul2 = opset.multiply(weights, mul, auto_broadcast='numpy', name=f'{key}.mul2{name_suffix}')
    return mul2

def make_embedding(key, input, consts):
    if configs['quant_type'] != '':
        if configs['quant_type'] == 'f16':
            emb_f16 = consts[key].astype(np.float16)
            emb_node = Constant(emb_f16, True)
            emb_node.set_friendly_name(name=f'{key}')
            embed_in_const = opset.convert(emb_node, 'f32', name=f'{key}.convert')
        else:
            embed_in_const = _make_compressed_weight_nncf(consts[key], key)
    else:
        embed_in_const = Constant(consts[key], True)
        embed_in_const.set_friendly_name(name=key)
    inputs_embeds = opset.gather(embed_in_const, indices=input, axis=0)
    return inputs_embeds

def save_tokenzier(orig_model_path, ov_model_path):
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(ov_model_path)

@nameable_op
def swish(
    data: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performing Swish activation function Swish(x, beta=1.0) = x * sigmoid(x * beta)).

    :param data: Tensor with input data floating point type.
    :return: The new node which performs Swish
    """
    return _get_node_factory_opset4().create("Swish", as_nodes(data, name=name), {})
