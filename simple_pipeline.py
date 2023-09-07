
import numpy as np
from openvino.runtime import Core
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils

core = Core()
core.add_extension("./custom_ops/build/libov-cpu-llm-experimental.so")

np.set_printoptions(linewidth=np.inf)

def create_sinusoidal_positions(num_pos: int, dim: int):
    import torch
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq).float()
    return torch.sin(sinusoid_inp).detach().numpy(), torch.cos(sinusoid_inp).detach().numpy()

sin_tab, cos_tab = create_sinusoidal_positions(2048, 64)
print(sin_tab.shape, cos_tab.shape)
print(sin_tab)
print(cos_tab)


tokenizer = AutoTokenizer.from_pretrained("/home/llm_irs/pytorch_frontend_models/gpt-j-6b/pytorch_original/", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token_id
tokenizer.padding_side = "left"             # pad to left
tokenizer.pad_token = tokenizer.eos_token   # to avoid an error

prompt=["What's Oxygen?", "Who"]
#prompt=["Who", "What's Oxygen?"]
#prompt=["Who", "Hi"]
#prompt=["Who"]

print("Prompt: ", prompt)
inputs = tokenizer(prompt, return_tensors="np", padding=True, return_token_type_ids=False)
inputs = tokenizer(prompt, return_tensors="np", max_length=5, pad_to_max_length = True, return_token_type_ids=False)

input_ids, attn_mask = inputs["input_ids"], inputs["attention_mask"]

attn_mask = (attn_mask - 1) * np.finfo(np.float32).max

all_tokens = input_ids

print(f"input_ids = \n{input_ids}")
print(f"attn_mask = \n{attn_mask}")

model_path = "gen/gptj_6b.xml"
print(f"read_model {model_path}...")
ov_model = core.read_model(model_path)
print(f"compile_model {model_path}...")
compiled_model = core.compile_model(ov_model, "CPU")

max_kv_len = 2048
n_embd = 4096
n_layer = 28
n_head = 16
head_size = n_embd//n_head

batch_size = input_ids.shape[0]

kv_cache = np.zeros([n_layer*2, batch_size, n_head, max_kv_len, head_size], dtype=np.float32)
beam_table = np.zeros([batch_size, max_kv_len], dtype=np.int32)
for b in range(batch_size):
    beam_table[b,:] = b

first_input = {"input_ids": input_ids, 
               "kv_cache": kv_cache,
               "beam_table" : beam_table,
               "attn_mask": attn_mask,
               "cos_tab" : cos_tab,
               "sin_tab" : sin_tab,
               }

outputs = compiled_model(first_input)


firts_key = next(iter(outputs))
logits = outputs[firts_key]

print(logits.shape)

next_token_logits = logits[:, -1, :]

print(next_token_logits.shape)

next_tokens_scores = next_token_logits
next_tokens = np.argmax(next_tokens_scores, axis=-1)

print(next_tokens)

for i in range(32):
    next_tokens = next_tokens.reshape(batch_size, 1)
    first_input['input_ids'] = np.array(next_tokens)

    all_tokens = np.concatenate([all_tokens, next_tokens], axis=-1)

    # zero means valid, -np.finfo(np.float32).max means invalid(padding-part)
    attn_mask = np.concatenate([attn_mask, np.zeros([batch_size, 1], dtype=np.float32)], axis=-1)
    first_input['attn_mask'] = attn_mask

    print("input_ids=", input_ids)
    print("attn_mask=", attn_mask)
    #print("kv_cache=", kv_cache[21, 0, 0, :, 0])
    outputs = compiled_model(first_input)

    firts_key = next(iter(outputs))
    logits = outputs[firts_key]
    next_tokens = np.argmax(logits, axis=-1)


print(all_tokens)

output_text = tokenizer.batch_decode(all_tokens, skip_special_tokens=True)
for t in output_text:
    print([t])

