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
## gptj_6b

convert orginal model into OpenVINO IR:

```bash
python models/gptj.py
```