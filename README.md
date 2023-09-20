# ov.cpu.llm.experimental
This repo demonstrates a LLM optimization method by custom-ops for OpenVINO. In order to inference the LLM efficiently, this repo introduces a new Op called `MHA` and re-construct the LLM based on this new-ops.

## Build OpenVINO and Custom Ops on Linux
You could refer to [build_linux](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_linux.md) for more details. Please set the install dir for openvino. Note, please make sure the gcc version is at least 11.2.

### Build Customized OpenVINO
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
### Build OpenVINO Custom Ops library
Please Do Reminder to enable the customized OpenVINO environment for this repo
#### Enable OpenVINO Environment
```bash
source <ov install dir>/setupvars.sh
```
#### Build custom ops library
```bash
cd custom_ops
mkdir build && cd build
cmake ..
make -j8
# custom_ops/build/libov-cpu-llm-experimental.so
```

## Build OpenVINO and Custom Ops on Windows
You could refer to [build_windows](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build_windows.md) for more details. Please set the install dir for openvino. Note, please make sure the MSVC version is at least Visual Studio 16 2019.

### Build Customized OpenVINO
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
### Build OpenVINO Custom Ops library
Please Do Reminder to enable the customized OpenVINO environment for this repo
#### Enable OpenVINO Environment
```bash
source <ov install dir>/setupvars.sh
```
#### Build OpenVINO custom ops
```bash
<ov install dir>/setupvars.bat
cd custom_ops
mkdir build && cd build
cmake --build . --config Release --verbose -j8 ..
# custom_ops\\build\\Release\\ov-cpu-llm-experimental.dll
```

## Setup Demo Environment
```
Install python env
```bash
pip3 install -r requirements.txt
pip3 install -e .
```
## Model Conversion

convert orginal model into OpenVINO IR:

```bash
python models/gptj.py
python models/gptneox.py
python models/falcon.py
python models/llama.py
python models/chatglm2.py
```
convert orginal model into OpenVINO IR with INT8 weight compression:
```bash
python models/gptj.py --compressed_weight=true
python models/gptneox.py --compressed_weight=true
python models/falcon.py --compressed_weight=true
python models/llama.py --compressed_weight=true
python models/chatglm2.py --compressed_weight=true
```

## Run Pipeline

```bash
# greedy search:  f32/bf16 
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3 --greedy
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3 --greedy --bf16
# beam search:  f32/bf16 
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3 --bf16
# specific input token length (support multiple langth, multiple round)
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -pl 32 512 1024 2016 8192 -r 3 --bf16
# run on all numa nodes
python llm_pipeline.py -m ./gen/falcon_40b.xml -bs 1 --bf16 -pl 8000
```
