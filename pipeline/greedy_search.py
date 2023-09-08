import utils 
import numpy as np
import pipeline.utils
import time
from openvino.runtime import Tensor, Type

def prepare_next_input(model_inputs, next_tokens):
    model_inputs['input_ids'] = np.array(next_tokens[..., np.newaxis])

    if 'attn_mask' in model_inputs:
        attention_mask = model_inputs['attn_mask']
        model_inputs['attn_mask'] = np.concatenate([attention_mask,
                                                    np.zeros([attention_mask.shape[0], 1], dtype=np.int32)], axis=-1)
    return model_inputs

def generate_greedy(model, input_ids, attention_mask, max_new_tokens, eos_token_id, pad_token_id, max_kv_len = 2048):
    first_iteration = True
    model_inputs = {}
    batch_size = input_ids.shape[0]
    kvcache_shape = [2 * model.pipeline_config.n_layers,
                     batch_size,
                     model.pipeline_config.n_head,
                     max_kv_len,
                     model.pipeline_config.head_size]
    kv_cache = Tensor(model.input("kv_cache").get_element_type(), kvcache_shape)

    # initialize "straight" beams in greedy search
    beam_table = np.zeros([batch_size, max_kv_len]).astype("int32")
    for b in range(batch_size):
        beam_table[b, :] = b

    sin_tab, cos_tab = pipeline.utils.create_sinusoidal_positions(max_kv_len, model.pipeline_config.rotary_dims)
    model_inputs = {"input_ids": input_ids,
                    "attn_mask": attention_mask,
                    "kv_cache": kv_cache,
                    "beam_table": beam_table,
                    "cos_tab": cos_tab,
                    "sin_tab": sin_tab
                    }
    latency = []
    cur_len = 0
    while True:
        time0 = time.time()
        if first_iteration:
            first_iteration = False
            outputs = model(model_inputs)
        else:
            outputs = model(model_inputs)

        logits = next(iter(outputs.values()))
        next_token_logits = logits[:, -1, :]
        
        # pre-process distribution
        next_tokens_scores = next_token_logits
        next_tokens = np.argmax(next_tokens_scores, axis=-1)
        # get next token id
        # break the loop if max length or end of text token is reached
        cur_len = cur_len + 1
        if cur_len == max_new_tokens or (next_tokens == eos_token_id).all():
            latency.append(time.time() - time0)
            break
        else:
            input_ids = np.concatenate((input_ids, next_tokens[:, None]), axis=-1)
            model_inputs = prepare_next_input(model_inputs, next_tokens)
        latency.append(time.time() - time0)

    return input_ids, latency