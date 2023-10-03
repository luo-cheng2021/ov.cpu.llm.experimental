# ov.cpu.llm.experimental
This repo demonstrates a LLM optimization method by custom-ops for OpenVINO. In order to inference the LLM efficiently, this repo introduces a new Op called `MHA` and re-construct the LLM based on this new-ops.

## 1.1. Build Dependency on Linux
You could refer to [build_linux](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md) for more details. Please set the install dir for openvino. Note, please make sure the gcc version is at least 11.2.

### Build OpenVINO
```bash
git clone https://github.com/usstq/openvino.git -b vnode-lc
cd openvino && git submodule update --init --recursive 
python3 -m pip install -U pip 
python3 -m pip install -r ./src/bindings/python/src/compatibility/openvino/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/wheel/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/requirements.txt
mkdir build && cd build
cmake -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<ov install dir> ..

# if want to run the model on multiple numa nodes, use the following
# cmake -DENABLE_INTEL_GPU=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DTHREADING=OMP -DCMAKE_INSTALL_PREFIX=<ov install dir> ..
make --jobs=$(nproc --all)
make install
cd <ov install dir>/tools/
python3 -m pip install  openvino*.whl

```
### Build Custom Ops Library
Please Do Reminder to enable the customized OpenVINO environment for this repo
```bash
source <ov install dir>/setupvars.sh
cd custom_ops
mkdir build && cd build
cmake ..
make -j8
# custom_ops/build/libov-cpu-llm-experimental.so
```

## 1.2. Build Dependency on Windows
You could refer to [build_windows](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_windows.md) for more details. Please set the install dir for openvino. Note, please make sure the MSVC version is at least Visual Studio 16 2019.

### Build OpenVINO
```bash
git clone https://github.com/usstq/openvino.git -b vnode-lc
cd openvino && git submodule update --init --recursive
python3 -m pip install -U pip
python3 -m pip install -r ./src/bindings/python/src/compatibility/openvino/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/wheel/requirements-dev.txt
python3 -m pip install -r ./src/bindings/python/requirements.txt
mkdir build && cd build
cmake -G "Visual Studio 16 2019" -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<ov install dir> ..

# if want to run the model on multiple numa nodes, use the following
# cmake -G "Visual Studio 16 2019" -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DTHREADING=OMP -DCMAKE_INSTALL_PREFIX=<ov install dir> ..
cmake --build . --config Release --verbose -j8
cmake --install .
cd <ov install dir>/tools/
python3 -m pip install  openvino*.whl
```
### Build Custom Ops Library
Please Do Reminder to enable the customized OpenVINO environment for this repo
```bash
<ov install dir>/setupvars.bat
cd custom_ops
mkdir build && cd build
cmake --build . --config Release --verbose -j8 ..
# custom_ops\\build\\Release\\ov-cpu-llm-experimental.dll
```

## 2. Setup Demo Environment
Install python env
```bash
pip3 install -r requirements.txt
pip3 install -e .
```

## 3. Model Conversion
convert orginal model into OpenVINO FP32 IR:

```bash
python models/gptj.py
python models/gptneox.py
python models/falcon.py
python models/llama.py
python models/chatglm2.py
```
convert orginal model into OpenVINO INT8 IR with weight compression:
```bash
python models/gptj.py --compressed_weight=true
python models/gptneox.py --compressed_weight=true
python models/falcon.py --compressed_weight=true
python models/llama.py --compressed_weight=true
python models/chatglm2.py --compressed_weight=true
```

## 4. Run Pipeline

```bash
# greedy search:  f32/bf16 
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3 --greedy
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3 --greedy --bf16
# beam search:  f32/bf16 
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b/ -p "What's Oxygen?" -r 3 --bf16
# specific input token length (support multiple langth, multiple round)
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b/ -pl 32 512 1024 2016 8192 -r 3 --bf16
# run on all numa nodes
python llm_pipeline.py -m ./gen/falcon_40b -bs 1 --bf16 -pl 8000
```

# Quantization with experimental FC node

Inspired by excellent project [llama.cpp](https://github.com/ggerganov/llama.cpp), we use following quantization methods: 
  - Weights are quantized off-line
  - Activations are quantized dynamically at runtime

| quant_type    |  description |
| ---------     |     -------  |
| `F16`         | FP16 weight format |
| `Q8_C`        | per-output channel symmetric weight-quantization |
| `Q4_C`        | per-output channel asymmetric weight-quantization |
| `Q8_0`, `Q4_0`| llama.cpp style per-32 weights symmetric weight-quantization |
| `Q4_1`        | llama.cpp style per-32 weights asymmetric weight-quantization |

> Note
>  - asymmetric quantization improves accuracy (PPL) at lower quantization bits, so Q4_C uses asymmetric quantization (with integer zero-point which has higher accuracy than non-integer zero-point)

## performance/accuracy report

RPL: i9-13900K + dua-chanel DDR5 @ 4800MT/s (~70GB/s)

```bash
# performance
numactl -C0-15  python llm_pipeline.py -m ./gen/llama-2-7b-chat/Q8_0/ -p "I am retail store manager with new ice cream flavor Super Sweet White Coffee. Can you generate a twitter post to promote it?" -r 1 --greedy -al 32

# perplexity
# download wikitext-2-raw from :
#   https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip?ref=salesforce-research
#   https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
python ./llm_perplexity.py -f=./wikitext-2-raw/wiki.test.raw -ov ./gen/llama-2-7b-chat/F16/
```


| Model    | Measure          | F32   | F16     |  Q8_0    |  Q4_1  |  Q4_0  |  Q8_C  |   Q4_C |
| -------- | -------          |-------|  -------|  ------- |------- |------- |------- |------- |
| Llama-7B | bin file size    | 26G   | 13G     |   6.7G   | 4.0G   | 3.6G   |  6.3G  |   3.3G |
|          | ms/tok @ 8 Pcore | 383   | 196     |   107    |  69    |  64    |  99    |    57  |
|          |  perplexity      |  7.49 | 7.49    |   7.49   |  7.79  |  7.81  |  7.50  | 10.33  |

