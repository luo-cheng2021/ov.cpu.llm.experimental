
from openvino.runtime import opset8 as opset
from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape, op, serialize, save_model
import numpy as np
import openvino as ov
from openvino.runtime.passes import Manager, Matcher, MatcherPass, WrapType, AnyInput
from openvino.runtime.utils import replace_node
import tqdm
import pickle, sys, time, argparse
import os, yaml
from collections import OrderedDict

def get_fc_weight(node):
    if node.get_type_name() != "MatMul":
        return None

    act_out = node.input_value(0)

    # weight matrix: [OC, IC]
    const_weight = node.input_value(1).get_node()
    if const_weight.get_type_name() == "Convert":
        const_weight = const_weight.input_value(0).get_node()
    assert(const_weight.get_type_name() == "Constant")

    weight = const_weight.data.astype(np.float32)

    act_rank = len(act_out.get_partial_shape())
    IC = act_out.get_partial_shape().get_dimension(act_rank-1).get_length()
    assert(IC == weight.shape[1])
    return weight


class LayerConfig:
    def __init__(self, file_name):
        self.file_name = file_name
        self.dict_obj = None
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding="utf-8") as fr:
                self.dict_obj = yaml.safe_load(fr)
        if self.dict_obj is None:
            self.dict_obj = {}

    def save(self):
        with open(self.file_name, "w", encoding='utf-8') as file:
            yaml.safe_dump(self.dict_obj, file, sort_keys=False, width=4096)

    def __getitem__(self, layername):
        if not (layername in self.dict_obj.keys()):
            self.dict_obj[layername] = {}
        return self.dict_obj[layername]

    def pop(self, layername):
        self.dict_obj.pop(layername)

def to_smooth_quant_model(model, fc_observations, config: LayerConfig):
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    END = "\033[0m"

    cfg_rules = config['rules']
    if len(cfg_rules) == 0:
        # initialize rules, rules will be used to initialize layer configs
        cfg_rules['q_proj'] = 'SW'
        cfg_rules['k_proj'] = 'SW'
        cfg_rules['v_proj'] = 'SW'
        cfg_rules['gate_proj'] = 'SW'
        cfg_rules['up_proj'] = 'SW'
        cfg_rules['down_proj'] = 'W'
        cfg_rules['o_proj'] = 'AW'
        # from chatglm3-6b
        cfg_rules['self_attention.query_key_value'] = 'SAW'
        cfg_rules['self_attention.dense'] = 'SAW'
        cfg_rules['mlp.dense_h_to_4h'] = 'SAW'
        cfg_rules['mlp.dense_4h_to_h'] = 'SAW'
        cfg_rules['transformer.output_layer.matmul'] = 'SAW'

        cfg_rules['alpha'] = 0.8
        cfg_rules['outlier_rel_thr'] = 10
        cfg_rules['outlier_abs_thr'] = 30
        cfg_rules['act_quant_sym'] = False
    print(f"{GREEN}Using rules: {cfg_rules}{END}")

    config['outlier_layers'].clear()

    # Simple: for Extensions. Without any classes and inheritance.
    def pattern_replacement():
        act = AnyInput()
        wei = AnyInput()
        matmul = WrapType("opset8.MatMul", [act.output(0), wei.output(0)])

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            pvm = matcher.get_pattern_value_map()
            transpose_a = False
            transpose_b = True

            if not root.get_friendly_name() in fc_observations:
                return False

            # collect all FCs sharing same activation
            fc_nodes = []
            X_min = None
            X_max = None
            X_absmax = None
            W_absmax = None
            per_token_quant = False

            # all FC shares same source input will be quantized together
            # with same scale or not
            cfg_key = root.get_friendly_name()
            cfg = config[cfg_key]
            if 'quantize' not in cfg:
                # initialize  cfg['quantize']  with rules
                cfg['quantize'] = None
                for rules_kw in cfg_rules:
                    if rules_kw in cfg_key:
                        cfg['quantize'] = cfg_rules[rules_kw]

            if cfg['quantize'] is None:
                quant_weight = False
                quant_activation = False
                smooth_activation = False
            else:
                quant_weight = 'W' in cfg['quantize']
                quant_activation = 'A' in cfg['quantize']
                smooth_activation = 'S' in cfg['quantize']

            quant_flag = ""
            quant_flag += 'W' if quant_weight else '_'
            quant_flag += 'S' if smooth_activation else '_'
            quant_flag += 'A' if quant_activation else '_'

            if 'fc' not in cfg:
                cfg['fc'] = []

            for iput in pvm[act].get_target_inputs():
                fc_node = iput.get_node()
                fc_weight = get_fc_weight(fc_node)
                if fc_weight is not None:
                    tag = fc_node.get_friendly_name()
                    #if "mlp.down_proj" in root.get_friendly_name():
                    #    #quant_activation = False
                    #    per_token_quant = True
                    #    pass
                    if tag not in cfg['fc']:
                        cfg['fc'].append(tag)

                    # merge all observations
                    if X_min is None:
                        X_min = fc_observations[tag]["min"]
                        X_max = fc_observations[tag]["max"]
                        X_absmax = fc_observations[tag]["absmax"]
                        W_absmax = abs(fc_weight).max(0).clip(min=1e-5)
                    else:
                        X_min = np.minimum(X_min, fc_observations[tag]["min"])
                        X_max = np.maximum(X_max, fc_observations[tag]["max"])
                        X_absmax = np.maximum(X_absmax, fc_observations[tag]["absmax"])
                        W_absmax = np.maximum(W_absmax, abs(fc_weight).max(0).clip(min=1e-5))
                    fc_nodes.append((fc_node, fc_weight))

            cfg['fc_observations'] = f"{X_min.min():.2f} ~ {X_max.max():.2f}"

            # get outliers:
            X_absmax = X_absmax.clip(min=1e-5)
            X_maxabs_thr = max(cfg_rules['outlier_rel_thr'] * X_absmax.mean(), cfg_rules['outlier_abs_thr'])
            X_outliers = X_absmax[X_absmax > X_maxabs_thr].flatten()
            if X_outliers.size > 0:
                for fc_node, weight in fc_nodes:
                    name = fc_node.get_friendly_name()
                    config['outlier_layers'][name] = f"{quant_flag}  {X_min.min():.2f} ~ {X_max.max():.2f} outliers={X_outliers}"

            if len(fc_nodes) == 0:
                config.pop(cfg_key)
                return

            if not quant_weight:
                print(f"{RED}SKIPPED {[fc_node.get_friendly_name() for fc_node, _ in fc_nodes]} {END}")
                return

            # s : each IC channel of weight matrix * s
            # use big alpha, for bigger outlier |X|, so x_scale can scale it down further

            per_channel_alpha = (X_absmax * 0 + cfg_rules['alpha'])
            px = pow(X_absmax, per_channel_alpha)
            pw = pow(W_absmax, 1 - per_channel_alpha)
            # pw = px
            smoothquant_w_scales = (px / pw).clip(min=1e-5)
            smoothquant_x_scales = 1/smoothquant_w_scales

            # [OC, IC] * [IC]
            if smooth_activation:
                x_min_per_tensor = (X_min * smoothquant_x_scales).min()
                x_max_per_tensor = (X_max * smoothquant_x_scales).max()
                node_act = opset.multiply(pvm[act], op.Constant(smoothquant_x_scales))
            elif quant_activation:
                x_min_per_tensor = (X_min * smoothquant_x_scales).min()
                x_max_per_tensor = (X_max * smoothquant_x_scales).max()
                # symmetrical quantization has lower accuracy than asymmetrical
                if cfg_rules['act_quant_sym']:
                    absmax = max(abs(x_min_per_tensor), abs(x_max_per_tensor))
                    x_min_per_tensor = -absmax
                    x_max_per_tensor = absmax

                act_smoothed = opset.multiply(pvm[act], op.Constant(smoothquant_x_scales))

                levels = np.int32(256)
                if per_token_quant:
                    # per-token dynamic, need special impl
                    absmax_per_token = opset.reduce_max(opset.absolute(act_smoothed), [0, 1], keep_dims = True)
                    input_low = opset.negative(absmax_per_token)
                    output_low = opset.negative(absmax_per_token)
                    input_high = absmax_per_token
                    output_high = absmax_per_token
                else:
                    # per-tensor static (easier for impl to speed-up)
                    input_low = np.array(x_min_per_tensor, dtype=np.float32)
                    input_high = np.array(x_max_per_tensor, dtype=np.float32)
                    output_low = np.array(x_min_per_tensor, dtype=np.float32)
                    output_high = np.array(x_max_per_tensor, dtype=np.float32)

                node_act = opset.fake_quantize(act_smoothed,
                                            input_low,
                                            input_high,
                                            output_low,
                                            output_high,
                                            levels)
            else:
                node_act = pvm[act]
                x_min_per_tensor = X_min.min()
                x_max_per_tensor = X_max.max()


            info = f"{quant_flag} [x:{smoothquant_x_scales.min():.2f}~{smoothquant_x_scales.max():.2f} w:{smoothquant_w_scales.min():.2f}~{smoothquant_w_scales.max():.2f}]  {X_min.min():.2f}~{X_max.max():.2f} =>  {x_min_per_tensor:.2f}~{x_max_per_tensor:.2f} mean:{X_absmax.mean():.3f}  big:{X_outliers}"
            print(info)
            cfg['info'] = info

            for fc_node, weight in fc_nodes:
                # quantize weight to INT8 on per-OC basis (per-tensor is not enough)
                if quant_activation or smooth_activation:
                    weight = weight * smoothquant_w_scales
                w_deq_scales = abs(weight).max(1, keepdims=True) / 127
                weight_quant = (weight / w_deq_scales).round().astype(np.int8)
                w_deq = opset.multiply(opset.convert(op.Constant(weight_quant), Type.f32), op.Constant(w_deq_scales))

                name = fc_node.get_friendly_name()

                new_matmul = opset.matmul(node_act, w_deq, transpose_a, transpose_b, name = name)
                replace_node(fc_node, new_matmul)

                #if outlier_result is not None:
                #    new_matmul = opset.add(new_matmul, outlier_result, name="AddOutlier")
                print(f"\t {new_matmul.get_friendly_name()}")

            return True
        return Matcher(matmul, "SimpleReplacement"), callback
    manager = Manager()
    manager.register_pass(MatcherPass(*pattern_replacement()))
    manager.run_passes(model)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--xml_path", type=str, required=True, help="raw openvino IR XML file")
    parser.add_argument("-s", "--act_scales_path", type=str, required=True, help="target pickle file storing calibration result",
                        default="act_scales/llama-2-7b.pickle")
    parser.add_argument("-p", "--prompt", type=str, default="What's oxygen?")
    parser.add_argument("-c", "--config", type=str, default="sq_config.yaml")
    parser.add_argument("output_xml_path", type=str, help="target openvino IR XML")

    args = parser.parse_args()

    handle = open(args.act_scales_path, 'rb')
    with handle:
        fc_observations = pickle.load(handle)

    print("=== absmax of activations observed ===")
    for fc_name in fc_observations:
        the_min = fc_observations[fc_name]["min"]
        the_max = fc_observations[fc_name]["max"]

        the_absmax = np.maximum(np.abs(the_min), np.abs(the_max))
        fc_observations[fc_name]["absmax"] = the_absmax

        mean_absmax = max(0.5, np.mean(the_absmax))
        outlier_idx = (the_absmax > 20 * mean_absmax)
        if (outlier_idx.sum() > 0):
            print(fc_name, mean_absmax, the_absmax[outlier_idx])


    # initialize openvino core
    core = Core()
    # read the model and corresponding weights from file
    ov_model = core.read_model(args.xml_path)

    config = LayerConfig(args.config)

    to_smooth_quant_model(ov_model, fc_observations, config = config)

    print(f"saving updated sq-config to {args.config} ...", end="")
    config.save()
    print(f"Done")

    print(f"saving smooth-quantized IR to {args.output_xml_path} ...", end="")
    save_model(ov_model, args.output_xml_path, True)
    print(f"Done")
