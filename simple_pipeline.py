
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


class OVModel:
    def __init__(self, ir_path, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        self.ir_path = ir_path

    def load(self):
        print(f"load Tokenizer from {self.tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"             # pad to left

        print(f"read_model {self.ir_path}...")
        ov_model = core.read_model(self.ir_path)
        kv_cache_shape = ov_model.input("kv_cache").partial_shape
        cos_tab_shape = ov_model.input("cos_tab").partial_shape

        # 2*n_layers, B, H, L, S
        self.n_layers = kv_cache_shape[0].get_length() // 2
        self.n_head = kv_cache_shape[2].get_length()
        self.head_size = kv_cache_shape[4].get_length()
        self.rotary_dims = cos_tab_shape[1].get_length() * 2 # assumes sin/cos table dims is half of rotary_dims

        print(f"\tn_layers={self.n_layers}, n_head={self.n_head}, head_size={self.head_size}, rotary_dims={self.rotary_dims}")
        print(f"compile_model {self.ir_path}...")
        self.compiled_model = core.compile_model(ov_model, "CPU")

    def new_kv_cache(self, batch_size, max_kv_len):
        return np.zeros([self.n_layers*2, batch_size, self.n_head, max_kv_len, self.head_size], dtype=np.float32)

    def tokenize(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True, return_token_type_ids=False)
        # inputs = self.tokenizer(prompt, return_tensors="np", max_length=5, pad_to_max_length = True, return_token_type_ids=False)
        input_ids = inputs["input_ids"]
        attn_mask = (inputs["attention_mask"] - 1) * np.finfo(np.float32).max

        return input_ids, attn_mask

    def infer(self, inputs):
        outputs = self.compiled_model(inputs)
        firts_key = next(iter(outputs))
        logits = outputs[firts_key]
        return logits
    
    def decode(self, all_tokens):
        return self.tokenizer.batch_decode(all_tokens, skip_special_tokens=True)

m1 = OVModel("gen/gptj_6b.xml", "/home/llm_irs/pytorch_frontend_models/gpt-j-6b/pytorch_original/")
m1 = OVModel("gen/dolly_v2_12b.xml", "/home/llm_irs/pytorch_frontend_models/dolly-v2-12b/pytorch_original/")
m1 = OVModel("gen/falcon_40b.xml", "/home/openvino-ci-68/falcon-40b/")

m1.load()

prompt=["What's Oxygen?", "Who"]
#prompt=["Who", "What's Oxygen?"]
#prompt=["Who", "Hi"]
#prompt=["Who"]
#prompt=["Hi"]
#prompt=["What's Oxygen?"]

print("Prompt: ", prompt)
input_ids, attn_mask = m1.tokenize(prompt)

all_tokens = input_ids

print(f"input_ids = \n{input_ids}")
print(f"attn_mask = \n{attn_mask}")

batch_size = input_ids.shape[0]
max_kv_len = 2048

beam_table = np.zeros([batch_size, max_kv_len], dtype=np.int32)
for b in range(batch_size):
    beam_table[b,:] = b

# generate cos/sin
sin_tab, cos_tab = create_sinusoidal_positions(max_kv_len, m1.rotary_dims)
print(sin_tab.shape, cos_tab.shape)

first_input = {"input_ids": input_ids, 
               "kv_cache": m1.new_kv_cache(batch_size, max_kv_len),
               "beam_table" : beam_table,
               "attn_mask": attn_mask,
               "cos_tab" : cos_tab,
               "sin_tab" : sin_tab,
               }

logits = m1.infer(first_input)

next_token_logits = logits[:, -1, :]
next_tokens = np.argmax(next_token_logits, axis=-1)
print(next_tokens)

for i in range(32):
    next_tokens = next_tokens.reshape(batch_size, 1)
    first_input['input_ids'] = np.array(next_tokens)

    all_tokens = np.concatenate([all_tokens, next_tokens], axis=-1)

    # zero means valid, -np.finfo(np.float32).max means invalid(padding-part)
    attn_mask = np.concatenate([attn_mask, np.zeros([batch_size, 1], dtype=np.float32)], axis=-1)
    first_input['attn_mask'] = attn_mask

    if False:
        print("input_ids=", input_ids)
        print("attn_mask=", attn_mask)

    logits = m1.infer(first_input)
    next_tokens = np.argmax(logits, axis=-1)


print(all_tokens)

output_text = m1.decode(all_tokens)
for t in output_text:
    print([t])

