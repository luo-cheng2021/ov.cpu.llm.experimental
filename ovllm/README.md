## ovllm

## Help
```bash
# Setup Demo Environment
pip3 install -r requirements.txt
pip3 install -e .

# convert orginal model into OpenVINO ovllm IR:
python3 -m ovllm.export.llama -m meta-llama/Llama-2-7b-hf -o gen/llama-2-7b/ --quant_type f16
python3 -m ovllm.export.llama -m meta-llama/Llama-2-13b-hf -o gen/llama-2-13b/ --quant_type f16
python3 -m ovllm.export.gptj -m EleutherAI/gpt-j-6b -o gen/gpt-j-6b/ --quant_type f16

# smooth-quantize the model : calibartion
python -m ovllm.sq_calibration  -m ./gen/llama-2-7b/f16/ Llama2-7b-ovllm.pickle
python -m ovllm.sq_calibration  -m ./gen/llama-2-13b/f16/ Llama2-13b-ovllm.pickle
python -m ovllm.sq_calibration  -m ./gen/chatglm3-6b/f16/ chatglm3-6b-ovllm.pickle
python -m ovllm.sq_calibration  -m ./gen/gpt-j-6b/f16 gpt-j-6b-ovllm.pickle

# smooth-quantize the model : sq_quant
python -m ovllm.sq_quant -m=./gen/llama-2-7b/f16/openvino_model.xml  -s Llama2-7b-chat-ovllm.pickle -c=sq_config_llama2_7b.yaml gen/llama-2-7b/SQ/openvino_model.xml 
python -m ovllm.sq_quant -m=./gen/llama-2-13b/f16/openvino_model.xml -s Llama2-13b-ovllm.pickle -c=sq_config_llama2_13b.yaml  gen/llama-2-13b/SQ/openvino_model.xml
python -m ovllm.sq_quant -m=./gen/chatglm3-6b/f16/openvino_model.xml -s chatglm3-6b-ovllm.pickle -c=sq_config_chatglm3_6b.yaml  gen/chatglm3-6b/SQ/openvino_model.xml
python -m ovllm.sq_quant -m=./gen/gpt-j-6b/f16/openvino_model.xml -s gpt-j-6b-ovllm.pickle -c=sq_config_gptj_6b.yaml  gen/gpt-j-6b/SQ/openvino_model.xml

# Edit sq_config_llama2_7b.yaml and run above command again

# copy other files
cp gen/llama-2-7b/f16/*.json ./gen/llama-2-7b/SQ/
cp gen/llama-2-7b/f16/*.model ./gen/llama-2-7b/SQ/

# greedy search:  f32/bf16 
python -m ovllm -m ./gen/llama-2-7b-chat/f16/ -p "What's Oxygen?" -r 3 -bs 4 --prec bf16 --numa 0 1 2 3 4 5
python -m ovllm -m ./gen/llama-2-7b-chat/f16/ -pl 32x1024 -r 3 -bs 4 --prec bf16 --numa 0 1 2 3 4 5

numactl -N1 -m1 python -m ovllm -m ./gen/chatglm3-6b/f16/ -p "What's Oxygen?" -r 3 -bs 1 --prec bf16

numactl -N1 -m1 python -m ovllm -m ./gen/chatglm3-6b/f16/ -pl 32x128 -al 8 -r 3 -bs 0 --prec bf16
[ 2024.4.0-16554-9c9778aba39-luocheng/mha_fusion_bhls]  [32x1,  128+7]  3197.6ms = [2754.0ms  1487.3tok/s] + [68.5ms + (62.5ms x 6)  505.4tok/s] + [0.3ms]
[ 2024.4.0-16554-9c9778aba39-luocheng/mha_fusion_bhls]  [32x4,   64+8]  2233.9ms = [1359.1ms  1506.9tok/s] + [133.1ms + (123.4ms x 6)  293.0tok/s] + [1.0ms]
[ 2024.4.0-16554-9c9778aba39-luocheng/mha_fusion_bhls]  [32x4,   64+8]  1890.2ms = [1085.8ms  1886.1tok/s] + [121.7ms + (113.6ms x 6)  318.7tok/s] + [1.2ms] # SQ

numactl -N1 -m1 python -m ovllm -m ./gen/llama-2-7b/f16/ -pl 32x128 -al 8 -r 3 -bs 0 --prec bf16
[ 2024.4.0-16554-9c9778aba39-luocheng/mha_fusion_bhls]  [32x1,  128+7]  2265.0ms = [1748.5ms  2342.6tok/s] + [75.9ms + (73.4ms x 6)  433.9tok/s] + [0.2ms]
[ 2024.4.0-16554-9c9778aba39-luocheng/mha_fusion_bhls]  [32x4,   64+8]  1726.9ms = [882.3ms  2321.3tok/s] + [133.6ms + (118.4ms x 6)  303.4tok/s] + [0.9ms]
[ 2024.4.0-16554-9c9778aba39-luocheng/mha_fusion_bhls]  [32x4,   64+8]  1143.2ms = [588.5ms  3480.0tok/s] + [81.9ms + (78.7ms x 6)  462.0tok/s] + [0.5ms] # SQ + LLM_DQ=3

# test lambada_openai accuracy using lm-evaluation-harness
#   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./ov.cpu.llm.experimental/gen/llama-2-7b/SQ/,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.7336|±  |0.0062|
|              |       |none  |     0|perplexity|↓  |3.4366|±  |0.0675|

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./ov.cpu.llm.experimental/gen/llama-2-13b/SQ/,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.7638|±  |0.0059|
|              |       |none  |     0|perplexity|↓  |3.0610|±  |0.0565|

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./gen/chatglm3-6b/SQ,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.6068|±  |0.0068|
|              |       |none  |     0|perplexity|↓  |9.1129|±  |0.4398|

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./gen/gpt-j-6b/f16,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.6806|±  |0.0065|
|              |       |none  |     0|perplexity|↓  |4.1319|±  |0.0895|

```