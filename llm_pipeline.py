import argparse
import json
import time
import numpy as np
from openvino.runtime import Core
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, serialize, opset_utils
from openvino.runtime import opset10 as opset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline.greedy_search import generate_greedy
from pipeline.beam_search import generate_beam

def prepare_next_input(model_inputs, next_tokens):
    model_inputs['input_ids'] = np.array(next_tokens)

    if 'attn_mask' in model_inputs:
        attention_mask = model_inputs['attn_mask']
        model_inputs['attn_mask'] = np.concatenate([attention_mask,
                                                    np.ones([attention_mask.shape[0], 1], dtype=np.int32)], axis=-1)
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
    parser.add_argument('-pm', '--pytorch-model', type=str, required=True,
                    help="path to pytorch model file")
    parser.add_argument('-pl', '--prompt-length', type=int, default=32, required=False,
                        help="prompt length")
    parser.add_argument('-al', '--answer-length', type=int,
                        default=32, help="generated token length")
    parser.add_argument('--greedy', type=bool, default=False)
    # Parse the argument
    args = parser.parse_args()

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

    compiled_model = core.compile_model(ov_model, "CPU")
    prompts = {}
    with open("promtps.json") as f:
        prompts = json.load(f)
    if str(args.prompt_length) not in prompts:
        print("Prompt with length {0} is not provided in prompt.json".format(
            args.prompt_length))
        exit(-1)

    text = prompts[str(args.prompt_length)]
    print("Input text: ", text)
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    gen_sequence_start = time.time()
    print("Start generate sequence ...")
    if args.greedy:
        output_ids = generate_greedy(compiled_model, input_ids, attention_mask, 
                                    max_sequence_length=args.prompt_length + args.answer_length,
                                    eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id)
    else:
        output_ids = generate_beam(compiled_model, input_ids, attention_mask, 
                                    max_sequence_length=args.prompt_length + args.answer_length,
                                    eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id)
    gen_sequence_end = time.time()
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    gen_sequence_length = len(output_ids[0]) - len(input_ids[0])
    gen_latency = gen_sequence_end - gen_sequence_start
