import tqdm, sys, argparse, os
import numpy as np
from openvino.runtime import Core
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
from openvino.preprocess import PrePostProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import pipeline.utils
 
class PPL:
    def __init__(self):
        self.nll = 0
        self.cnt = 0
    
    def __call__(self, all_logits, labels):
        '''
            all_logits [seq_length, vocab_size]
            labels [seq_length]
        '''
        seq_length = all_logits.shape[0]
        for i in range(0, seq_length - 1):
            logits = all_logits[i, :]
            max_logit = np.amax(logits)
            sum_exp = np.sum(np.exp(logits - max_logit))

            # logits at time-step i is for predicting token at time-step (i+1)
            next_tok = labels[i + 1]
            log_softmax_of_tok = (logits[next_tok] - max_logit) - np.log(sum_exp)

            self.nll += -log_softmax_of_tok
            self.cnt += 1
        return np.exp(self.nll / self.cnt)

    def __str__(self):
        return f"PPL: {np.exp(self.nll / self.cnt):.2f}"

def perplexity_hf(args, text, raw_model_path):
    print("loading hf model ...")
    import torch
    raw_model = AutoModelForCausalLM.from_pretrained(raw_model_path)
    tokenizer = AutoTokenizer.from_pretrained(raw_model_path)

    print("tokenizing ...")
    inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    ppl_evaluator = PPL()

    progress_bar = tqdm.tqdm(range(0, input_ids.shape[1], 512))
    for i0 in progress_bar:
        input_ids_chunks = input_ids[:, i0:(i0+512)]
        input_ids_chunks[:, 0] = 1
        with torch.no_grad():
            result = raw_model.forward(input_ids_chunks, labels = input_ids_chunks, return_dict=True)
            #print(f"ppl = {torch.exp(result.loss)}")
            seq_len = result.logits.shape[1]
            ppl_evaluator(result.logits.numpy()[0, seq_len//2:, :], input_ids_chunks.numpy()[0, seq_len//2:])
        progress_bar.set_description(f"{ppl_evaluator}")


class OvLLMModel:
    def __init__(self, ov_model_path) -> None:
        ext_path = None
        if sys.platform == 'win32':
            ext_path = ".\\custom_ops\\build\\Release\\ov-cpu-llm-experimental.dll"
        elif sys.platform == 'linux':
            ext_path = "./custom_ops/build/libov-cpu-llm-experimental.so"
        else:
            print(f"Sample code not supported on platform: {sys.platform}")
            exit(1)

        core = Core()
        custom_opset = opset_utils._get_node_factory()
        custom_opset.add_extension(ext_path)
        core.add_extension(ext_path)

        print("Init OpenVINO model ...")
        ov_model = core.read_model(os.path.join(ov_model_path, "openvino.xml"))

        # add preprocessor for bf16 kv_cache
        self.bf16 = False
        if self.bf16:
            kv_cache_precision = Type.bf16
            ppp = PrePostProcessor(ov_model)
            for key in ov_model.inputs:
                if "kv_cache" in key.get_any_name() and kv_cache_precision != key.get_element_type():
                    ppp.input(key.get_any_name()).tensor().set_element_type(kv_cache_precision)
            ov_model = ppp.build()

        kv_cache_shape = ov_model.input("kv_cache").partial_shape
        cos_tab_shape = ov_model.input("cos_tab").partial_shape

        # 2*n_layers, B, H, L, S
        self.n_layers = kv_cache_shape[0].get_length() // 2
        self.n_head = kv_cache_shape[2].get_length()
        self.head_size = kv_cache_shape[4].get_length()
        self.rotary_dims = cos_tab_shape[1].get_length() * 2 # assumes sin/cos table dims is half of rotary_dims
        self.kv_eletype = ov_model.input("kv_cache").get_element_type()

        ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
                    "INFERENCE_PRECISION_HINT" : "bf16" if self.bf16 else "f32",
                    "CPU_DENORMALS_OPTIMIZATION" : "YES",
                    "CACHE_DIR" : None}
        self.compiled_model = core.compile_model(ov_model, "CPU", ov_config)

    def forward(self, input_ids, attention_mask, max_kv_len):
        attention_mask = (1.0 - attention_mask) * np.finfo(np.float32).min
        batch_size = input_ids.shape[0]
        kvcache_shape = [2 * self.n_layers,
                        batch_size,
                        self.n_head,
                        max_kv_len,
                        self.head_size]

        kv_cache = Tensor(self.kv_eletype, kvcache_shape)

        # initialize "straight" beams in greedy search
        beam_table = np.zeros([batch_size, max_kv_len]).astype("int32")
        for b in range(batch_size):
            beam_table[b, :] = b

        sin_tab, cos_tab = pipeline.utils.create_sinusoidal_positions(max_kv_len, self.rotary_dims)
        model_inputs = {"input_ids": input_ids,
                            "attn_mask": attention_mask,
                            "kv_cache": kv_cache,
                            "beam_table": beam_table,
                            "cos_tab": cos_tab,
                            "sin_tab": sin_tab
                            }
        return self.compiled_model(model_inputs)

    def __str__(self) -> str:
        return f"\tn_layers={self.n_layers}, n_head={self.n_head}, head_size={self.head_size}, rotary_dims={self.rotary_dims}"


def perplexity_ov(args, text, ov_model_path):
    print("loading ov model ...")
    tokenizer = AutoTokenizer.from_pretrained(ov_model_path)
    ovmodel = OvLLMModel(ov_model_path)

    print(f"tokenizing ...")
    inputs = tokenizer(text, return_tensors="np", return_token_type_ids=False)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    ppl_evaluator = PPL()

    progress_bar = tqdm.tqdm(range(0, input_ids.shape[1], 512))
    for i0 in progress_bar:
        input_ids_chunks = input_ids[:, i0:(i0+512)]
        input_ids_chunks[:, 0] = 1

        outputs = ovmodel.forward(input_ids_chunks, attention_mask[:, i0:(i0+512)], max_kv_len = 512 + 8)
        logits = next(iter(outputs.values()))
        seq_len = logits.shape[1]
        ppl_evaluator(logits[0, seq_len//2:, :], input_ids_chunks[0, seq_len//2:])

        progress_bar.set_description(f"{ppl_evaluator}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--prompt-file", type=str, default="./wikitext-2-raw/wiki.test.raw")
    parser.add_argument("-hf", type=str, default=None)
    parser.add_argument("-ov", type=str, default=None)
    args = parser.parse_args()

    with open(args.prompt_file) as f:
        text = f.read()

    if args.ov:
        perplexity_ov(args, text, args.ov)
    elif args.hf:
        perplexity_hf(args, text, args.hf)
