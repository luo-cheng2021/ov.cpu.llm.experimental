## ovllm

## Help
```bash
# Setup Demo Environment
pip3 install -r requirements.txt
pip3 install -e .

# convert orginal model into OpenVINO ovllm IR:
python -m ovllm.export.llama --quant_type=f16

# smooth-quantize the model
python -m ovllm.sq_calibration  -m ./gen/llama-2-7b-chat/f16/ Llama2-7b-chat-ovllm.pickle

python -m ovllm.sq_quant -m=./gen/llama-2-7b-chat/f16/openvino_model.xml  -s Llama2-7b-chat-ovllm.pickle gen/llama-2-7b-chat/SQ/openvino_model.xml -c sq_config_llama2_7b.yaml
# Edit sq_config_llama2_7b.yaml and run above command again


# greedy search:  f32/bf16 
python -m ovllm -m ./gen/llama-2-7b-chat/f16/ -p "What's Oxygen?" -r 3 -bs 4 --prec bf16 --numa 0 1 2 3 4 5
python -m ovllm -m ./gen/llama-2-7b-chat/f16/ -pl 32x1024 -r 3 -bs 4 --prec bf16 --numa 0 1 2 3 4 5

# test lambada_openai accuracy using lm-evaluation-harness
#   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
python -m ovllm.lm_eval --model ovllm --tasks lambada_openai --model_args path=./ov.cpu.llm.experimental/gen/llama-2-7b-chat/f16/,nbatch=1
```