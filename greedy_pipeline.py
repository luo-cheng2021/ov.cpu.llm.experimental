import argparse
import json
import time
import numpy as np
from openvino.runtime import Core
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.greedy_search import process_logits

checkpoint = "llama-7b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
eos_token_id = tokenizer.eos_token_id


# this function converts text to tokens
def tokenize(text):
    """
    tokenize input text using GPT2 tokenizer

    Parameters:
      text, str - input text
    Returns:
      input_ids - np.array with input token ids
      attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model
    """

    inputs = tokenizer(text, return_tensors="np")
    return inputs["input_ids"], inputs["attention_mask"]


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation


def process_logits(cur_length, scores, eos_token_id, min_length=0):
    """
    reduce probability for padded indicies

    Parameters:
      cur_length - current length of input sequence
      scores - model output logits
      eos_token_id - index of end of string token in model vocab
      min_length - minimum length for appling postprocessing
    """
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def get_top_k_logits(scores, top_k):
    """
    perform top-k sampling

    Parameters:
      scores - model output logits
      top_k - number of elements with highest probability to select
    """
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(scores, mask=indices_to_remove,
                                 fill_value=filter_value).filled()
    return filtred_scores


def prepare_next_input(model_inputs, next_tokens):
    model_inputs['input_ids'] = np.array([[next_tokens]])

    if 'attention_mask' in model_inputs:
        attention_mask = model_inputs['attention_mask']
        model_inputs['attention_mask'] = np.concatenate([attention_mask,
                                                         np.ones(attention_mask.shape[0], 1)], dim=-1)
    return model_inputs


def generate_greedy(input_ids, attention_mask, max_sequence_length=128,
                    eos_token_id=eos_token_id, dynamic_shapes=True, engine="OV"):
    first_iteration = True
    model_inputs = {}
    output_names = []
    while True:
        cur_input_len = len(input_ids[0])
        model_input_ids = input_ids
        model_input_attention_mask = attention_mask

        if first_iteration:
            first_input = {"input_ids": model_input_ids,
                           "attention_mask": model_input_attention_mask
                           }
            outputs = compiled_model(first_input)
            logits = outputs['logits']
            next_token_logits = logits[:, -1, :]
            first_iteration = False
        else:
            outputs = compiled_model(model_inputs)
        # pre-process distribution
        next_tokens_scores = next_token_logits
        next_tokens = np.argmax(next_tokens_scores, axis=-1)
        # get next token id
        # break the loop if max length or end of text token is reached
        if cur_input_len == max_sequence_length or next_tokens == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, next_tokens), axis=-1)
            model_inputs = prepare_next_input(model_inputs, next_tokens)

    return input_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="path to model file")
    parser.add_argument('-pl', '--prompt-length', type=int, default=32, required=False,
                        help="prompt length")
    parser.add_argument('-al', '--answer-length', type=int,
                        default=32, help="generated token length")
    # Parse the argument
    args = parser.parse_args()

    # initialize openvino core
    read_model_start = time.time()
    core = Core()
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
    input_ids, attention_mask = tokenize(text)

    gen_sequence_start = time.time()
    print("Start generate sequence ...")

    output_ids = generate_greedy(input_ids, attention_mask, max_sequence_length=args.prompt_length + args.answer_length,
                                 eos_token_id=eos_token_id, dynamic_shapes=True, engine=args.engine)
    gen_sequence_end = time.time()
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    gen_sequence_length = len(output_ids[0]) - len(input_ids[0])
    gen_latency = gen_sequence_end - gen_sequence_start
