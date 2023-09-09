# ov.cpu.llm.experimental

## build dependency
OpenVINO custom ops : 
```bash
cd custom_ops
mkdir build
cmake ..
make -j8
# custom_ops/build/libov-cpu-llm-experimental.so
```

some function implemented in pybind11 :
```bash
pip install -e .
```
## model convert

convert orginal model into OpenVINO IR:

```bash
python models/gptj.py
python models/gptneox.py
python models/falcon.py
python models/llama.py
```

## run pipeline

```bash
# greedy search:  f32/bf16 
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3 --greedy
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3 --greedy --bf16
# beam search:  f32/bf16 
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -p "What's Oxygen?" -r 3 --bf16
# specific input token length (support multiple langth, multiple round)
numactl -N 0 --membind=0  python llm_pipeline.py -m ./gen/gptj_6b.xml -pl 32 512 1024 2016 8192 -r 3 --bf16

```
