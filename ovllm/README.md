## ovllm

## Help
```bash
# Setup Demo Environment
pip3 install -r requirements.txt
pip3 install -e .

# convert orginal model into OpenVINO ovllm IR:
python -m ovllm.export.llama --quant_type=f16

# smooth-quantize the model : calibartion
python -m ovllm.sq_calibration  -m ./gen/llama-2-7b/f16/ Llama2-7b-ovllm.pickle
python -m ovllm.sq_calibration  -m ./gen/llama-2-13b/f16/ Llama2-13b-ovllm.pickle
python -m ovllm.sq_calibration  -m ./gen/chatglm3-6b/f16/ chatglm3-6b-ovllm.pickle

# smooth-quantize the model : sq_quant
python -m ovllm.sq_quant -m=./gen/llama-2-7b/f16/openvino_model.xml  -s Llama2-7b-chat-ovllm.pickle -c=sq_config_llama2_7b.yaml gen/llama-2-7b/SQ/openvino_model.xml 
python -m ovllm.sq_quant -m=./gen/llama-2-13b/f16/openvino_model.xml -s Llama2-13b-ovllm.pickle -c=sq_config_llama2_13b.yaml  gen/llama-2-13b/SQ/openvino_model.xml
python -m ovllm.sq_quant -m=./gen/chatglm3-6b/f16/openvino_model.xml -s chatglm3-6b-ovllm.pickle -c=sq_config_chatglm3_6b.yaml  gen/chatglm3-6b/SQ/openvino_model.xml

# Edit sq_config_llama2_7b.yaml and run above command again

# copy other files
cp gen/llama-2-7b/f16/*.json ./gen/llama-2-7b/SQ/
cp gen/llama-2-7b/f16/*.model ./gen/llama-2-7b/SQ/

# greedy search:  f32/bf16 
python -m ovllm -m ./gen/llama-2-7b-chat/f16/ -p "What's Oxygen?" -r 3 -bs 4 --prec bf16 --numa 0 1 2 3 4 5
python -m ovllm -m ./gen/llama-2-7b-chat/f16/ -pl 32x1024 -r 3 -bs 4 --prec bf16 --numa 0 1 2 3 4 5

# test lambada_openai accuracy using lm-evaluation-harness
#   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./ov.cpu.llm.experimental/gen/llama-2-7b-chat/f16/,nbatch=1

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./ov.cpu.llm.experimental/gen/llama-2-7b/SQ/,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.7359|±  |0.0061|
|              |       |none  |     0|perplexity|↓  |3.4486|±  |0.0677|

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./ov.cpu.llm.experimental/gen/llama-2-13b/SQ/,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.7615|±  |0.0059|
|              |       |none  |     0|perplexity|↓  |3.0723|±  |0.0566|

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./gen/chatglm3-6b/f16,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.6113|±  |0.0068|
|              |       |none  |     0|perplexity|↓  |8.5563|±  |0.4104|

python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./gen/chatglm3-6b/SQ,nbatch=1
|    Tasks     |Version|Filter|n-shot|  Metric  |   |Value |   |Stderr|
|--------------|------:|------|-----:|----------|---|-----:|---|-----:|
|lambada_openai|      1|none  |     0|acc       |↑  |0.6123|±  |0.0068|
|              |       |none  |     0|perplexity|↓  |8.6018|±  |0.4122|

```