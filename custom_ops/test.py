
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
from openvino.runtime import opset11 as opset
from openvino.runtime import opset_utils
import openvino.runtime as ovrt

if __name__ == "__main__":
    core = Core()
    ext_path = "./build/libov-cpu-llm-experimental.so"

    custom_opset = opset_utils._get_node_factory()
    custom_opset.add_extension(ext_path)

    n_layers = 28
    n_hidden = 2048
    n_head = 16
    rotary_dims = 64
    head_size = int(n_hidden / n_head)
    max_kv_len = 2048
    q_len = 10
    batch = -1

    mha_attr = {"arg_q":0,
                "arg_k":1,
                "arg_v":2,
                "arg_kv_cache":3,
                "arg_beam_table":4,
                "arg_kv_len":5,
                "arg_attn_mask":6,
                "arg_cos":7,
                "arg_sin":8,
                "layer_id":0,
                "rotary_dims":rotary_dims,
                "n_hidden":n_hidden,
                "n_head": n_head}

    q = opset.parameter([batch, q_len, n_head * head_size ], Type.f32)
    k = opset.parameter([batch, q_len, n_head * head_size ], Type.f32)
    v = opset.parameter([batch, q_len, n_head * head_size ], Type.f32)
    kv_cache = opset.parameter([2*n_layers, batch, n_head, max_kv_len, head_size], Type.f32)
    beam_table = opset.parameter([batch, max_kv_len], Type.f32)
    kv_len = opset.parameter([1], Type.i32)
    attn_mask = opset.parameter([batch, 1, 1, -1], Type.f32)
    cos_tab = opset.parameter([max_kv_len, rotary_dims//2], Type.f32)
    sin_tab = opset.parameter([max_kv_len, rotary_dims//2], Type.f32)

    mha_attr["layer_id"] = 0
    output = custom_opset.create("MultiHeadAttention", 
        [q, k, v, kv_cache, beam_table, kv_len, attn_mask, cos_tab, sin_tab], mha_attr)
    Result0 = opset.result(output)
    model = Model([Result0], [q, k, v, kv_cache, beam_table, kv_len, attn_mask, cos_tab, sin_tab], 'Model23')

    ovrt.serialize(model, "test_model.xml")
    compiled_model = core.compile_model(model, "CPU")
    
    print("compiled_model is ready")