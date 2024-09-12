
from openvino.runtime import opset11 as opset
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize
import numpy as np
import openvino as ov
from openvino.runtime.passes import Manager, Matcher, MatcherPass, WrapType, AnyInput
from openvino.runtime.utils import replace_node
import tqdm
import pickle, sys, time
from datasets import load_dataset
import argparse

from . import utils
from . import llm


def to_min_max_model(model):
    new_results = []
    fc_names = []
    def pattern_replacement():
        act = AnyInput()
        wei = AnyInput()
        matmul = WrapType("opset8.MatMul", [act.output(0), wei.output(0)])

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            pvm = matcher.get_pattern_value_map()

            const_weight = pvm[wei].get_node()
            if const_weight.get_type_name() == "Convert":
                const_weight = const_weight.input_value(0).get_node()

            if const_weight.get_type_name() != "Constant":
                print("\t skipped: ", root.get_friendly_name(), " the weight type: ", const_weight.get_type_name())
                return False

            #print(root, pvm[act].get_node())
            act_rank = len(pvm[act].get_partial_shape())
            axes =  [i for i in range(act_rank-1)]

            the_min = opset.reduce_min(pvm[act], reduction_axes = axes)
            the_max = opset.reduce_max(pvm[act], reduction_axes = axes)
            
            the_min.get_output_tensor(0).set_names({root.get_friendly_name() + "_min"})
            the_max.get_output_tensor(0).set_names({root.get_friendly_name() + "_max"})

            new_results.append(opset.result(the_min))
            new_results.append(opset.result(the_max))
            fc_names.append(root.get_friendly_name())
            return False

        return Matcher(matmul, "SimpleReplacement"), callback
    
    manager = Manager()
    manager.register_pass(MatcherPass(*pattern_replacement()))
    manager.run_passes(model)

    model.add_results(new_results)
    return fc_names

class OVLLMSmoothQuantCalib(llm.OVLLM):
    def _patch_model(self, model):
        self.fc_names = to_min_max_model(model)
        return model

    def _generate(self,
                  model,    # ompiled_model,
                  kv_cache, # [2 * n_layers, max_kv_len, batch_size * beam_size, n_head, head_size]
                  eos_token_id,
                  pad_token_id,
                  input_ids,
                  attention_mask,
                  beam_size,
                  max_new_tokens,
                  max_kv_len,
                  streamer,
                  continuation):
        batch_size = input_ids.shape[0]

        #print("input_ids=", input_ids.shape, " max_kv_len=", max_kv_len)
        # initialize "straight" beams in greedy search
        beam_table = np.zeros([batch_size, max_kv_len]).astype("int32")

        sin_tab, cos_tab = utils.create_sinusoidal_positions(max_kv_len, self.pipeline_config.rotary_dims, self._get_rotary_base())
        model_inputs = {"input_ids": input_ids[:, :self.token_limit],
                        "attn_mask": attention_mask[:, :self.token_limit],
                        "kv_cache": kv_cache,
                        "beam_table": beam_table,
                        "cos_tab": cos_tab,
                        "sin_tab": sin_tab,
                        }

        # for k in model_inputs:
        #     print(f" {k} = ", model_inputs[k].shape, " ")
        self.outputs = model(model_inputs)

        return None, None


def get_fc_observations(model_path, dataset_path, seq_len = 512, num_samples = 512, device="CPU", use_cache = True):
    prec = "f32"
    ovllm = OVLLMSmoothQuantCalib(model_path, prec)
    ovllm.token_limit = seq_len

    fc_observations = {}

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm.tqdm(range(num_samples)):
        # ovllm.generate(dataset[i]["text"], new_token_length=1)
        
        prompt = dataset[i]["text"]
        ovllm.generate(prompt, new_token_length=1)

        for fc_name in ovllm.fc_names:
            the_min = ovllm.outputs[fc_name + "_min"].data
            the_max = ovllm.outputs[fc_name + "_max"].data
            if not (fc_name in fc_observations):
                fc_observations[fc_name] = {"min": the_min, "max": the_max}
            else:
                fc_observations[fc_name]["min"] = np.minimum(the_min, fc_observations[fc_name]["min"])
                fc_observations[fc_name]["max"] = np.maximum(the_max, fc_observations[fc_name]["max"])


    return fc_observations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-path", type=str, default="./pile-rnd512.val.jsonl.zst", help="location of the calibration dataset, we use the validation set of the Pile dataset")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="raw openvino IR (OVModelForCausalLM) export by optimum-cli")
    parser.add_argument("-uc", '--use_cache', type=int, default=1)
    parser.add_argument("act_minmax_path", type=str, help="target pickle file for storing calibration result",
                        default="act_scales/llama-2-7b.pickle")

    args = parser.parse_args()

    print(f"calibrating {args.model_path} on {args.dataset_path} ...")
    fc_observations = get_fc_observations(
        model_path = args.model_path,
        dataset_path = args.dataset_path,
        use_cache=bool(args.use_cache)
    )

    print(f"saving fc_observations to {args.act_minmax_path}...")
    with open(args.act_minmax_path, 'wb') as handle:
        pickle.dump(fc_observations, handle)

    print("Done.")