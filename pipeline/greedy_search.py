import utils 
import numpy as np
import pipeline.utils as utils
def prepare_next_input(model_inputs, next_tokens):
    model_inputs['input_ids'] = np.array([next_tokens])

    if 'attn_mask' in model_inputs:
        attention_mask = model_inputs['attn_mask']
        model_inputs['attn_mask'] = np.concatenate([attention_mask,
                                                    np.zeros([attention_mask.shape[0], 1], dtype=np.int32)], axis=-1)
    return model_inputs

def generate_greedy(model, input_ids, attention_mask, max_new_tokens, eos_token_id, pad_token_id):
    first_iteration = True
    model_inputs = {}
    kv_cache = np.zeros([56, 1, 16, 2048, 256]).astype("float32")
    beam_table = np.zeros([2048,1]).astype("int32")
    sin_tab, cos_tab = utils.create_sinusoidal_positions(2048, 64)
    model_inputs = {"input_ids": input_ids,
                    "attn_mask": attention_mask,
                    "kv_cache": kv_cache,
                    "beam_table": beam_table,
                    "cos_tab": cos_tab,
                    "sin_tab": sin_tab
                    }
    cur_len = 0
    while True:
        cur_input_len = len(input_ids[0])
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
        if cur_len == max_new_tokens or next_tokens == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, next_tokens[:, None]), axis=-1)
            model_inputs = prepare_next_input(model_inputs, next_tokens)

    return input_ids