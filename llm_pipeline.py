import argparse
import json
import time
import numpy as np
from openvino.runtime import Core
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
from openvino.preprocess import PrePostProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline.greedy_search import generate_greedy
from pipeline.beam_search import generate_beam

class ModelConfig:
    def __init__(self, ov_model) -> None:
        kv_cache_shape = ov_model.input("kv_cache").partial_shape
        cos_tab_shape = ov_model.input("cos_tab").partial_shape

        # 2*n_layers, B, H, L, S
        self.n_layers = kv_cache_shape[0].get_length() // 2
        self.n_head = kv_cache_shape[2].get_length()
        self.head_size = kv_cache_shape[4].get_length()
        self.rotary_dims = cos_tab_shape[1].get_length() * 2 # assumes sin/cos table dims is half of rotary_dims

    def __str__(self) -> str:
        return f"\tn_layers={self.n_layers}, n_head={self.n_head}, head_size={self.head_size}, rotary_dims={self.rotary_dims}"

def prepare_next_input(model_inputs, next_tokens):
    model_inputs['input_ids'] = np.array(next_tokens)

    if 'attn_mask' in model_inputs:
        attention_mask = model_inputs['attn_mask']
        model_inputs['attn_mask'] = np.concatenate([attention_mask,
                                                    np.zeros([attention_mask.shape[0], 1], dtype=np.int32)], axis=-1)
    return model_inputs

def create_sinusoidal_positions(num_pos: int, dim: int):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos, dtype=np.float), inv_freq).astype("float32")
    return np.sin(sinusoid_inp), np.cos(sinusoid_inp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path to model file")
    parser.add_argument('-pm', '--pytorch-model', type=str, required=False,
                    help="path to pytorch model file")
    parser.add_argument('-pl', '--prompt-length', type=int, default=32, required=False,
                        help="prompt length")
    parser.add_argument('-p', '--prompt', type=str, nargs='+', required=False,
                        help="prompt")
    parser.add_argument('-al', '--answer-length', type=int,
                        default=32, help="generated token length")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    # Parse the argument
    args = parser.parse_args()

    # derive pytorch_model path
    if not args.pytorch_model:
        pm_map = {}
        pm_map["gptj_6b.xml"] = "/home/llm_irs/pytorch_frontend_models/gpt-j-6b/pytorch_original/"
        pm_map["dolly_v2_12b.xml"] = "/home/llm_irs/pytorch_frontend_models/dolly-v2-12b/pytorch_original/"
        pm_map["falcon_40b.xml"] = "/home/openvino-ci-68/falcon-40b/"
        for k in pm_map:
            if k in args.model:
                args.pytorch_model = pm_map[k]
    if not args.pytorch_model:
        raise "pytorch_model path is required for tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(args.pytorch_model, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    # initialize openvino core
    read_model_start = time.time()
    core = Core()
    ext_path = "./custom_ops/build/libov-cpu-llm-experimental.so"
    custom_opset = opset_utils._get_node_factory()
    custom_opset.add_extension(ext_path)
    core.add_extension(ext_path)
    print("Init OpenVINO model ...")
    # read the model and corresponding weights from file
    ov_model = core.read_model(args.model)

    # add preprocessor for bf16 kv_cache
    if args.bf16:
        kv_cache_precision = Type.bf16
        ppp = PrePostProcessor(ov_model)
        for key in ov_model.inputs:
            if "kv_cache" in key.get_any_name() and kv_cache_precision != key.get_element_type():
                ppp.input(key.get_any_name()).tensor().set_element_type(kv_cache_precision)
        ov_model = ppp.build()

    ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
                "INFERENCE_PRECISION_HINT" : "bf16" if args.bf16 else "f32",
                "CPU_DENORMALS_OPTIMIZATION" : "YES",
                "CACHE_DIR" : None}

    compiled_model = core.compile_model(ov_model, "CPU", ov_config)
    compiled_model.pipeline_config = ModelConfig(ov_model)

    if args.prompt:
        text = args.prompt
    else:
        prompts = {}
        with open("prompts.json") as f:
            prompts = json.load(f)
        if str(args.prompt_length) not in prompts:
            print("Prompt with length {0} is not provided in prompt.json".format(
                args.prompt_length))
            exit(-1)

        text = prompts[str(args.prompt_length)] * 2

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token_id
    tokenizer.padding_side = "left"             # pad to left

    inputs = tokenizer(text, return_tensors="np", padding=True, return_token_type_ids=False)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    attention_mask = (1.0 - attention_mask) * np.finfo(np.float32).min

    gen_sequence_start = time.time()
    print("Start generate sequence ...")
    if args.greedy:
        output_ids = generate_greedy(compiled_model, input_ids, attention_mask, 
                                    max_new_tokens=args.answer_length,
                                    eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id)
    else:
        output_ids = generate_beam(compiled_model, input_ids, attention_mask, 
                                    max_new_tokens=args.answer_length,
                                    eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id)
    gen_sequence_end = time.time()
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    gen_latency = gen_sequence_end - gen_sequence_start

    for i, out in enumerate(output_text):
        print(f"answer {i} : {[out]}")
