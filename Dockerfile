from ubuntu:22.04

## Install dependencies
USER root
RUN apt-get update && apt-get install -y \
	git cmake build-essential python3-venv python3-pip \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone https://github.com/usstq/openvino.git --single-branch -b vnode-lc \
	&& cd openvino && git submodule update --init --recursive \
	&& chmod +x install_build_dependencies.sh && ./install_build_dependencies.sh \
	&& python3 -m venv /root/openvino_venv && . /root/openvino_venv/bin/activate \
	&& python3 -m pip install -U pip \
	&& python3 -m pip install -r ./src/bindings/python/src/compatibility/openvino/requirements-dev.txt \
	&& python3 -m pip install -r ./src/bindings/python/wheel/requirements-dev.txt \
	&& python3 -m pip install -r ./src/bindings/python/requirements.txt

## Build OpenVINO
WORKDIR /tmp/openvino
RUN . /root/openvino_venv/bin/activate \
	&& mkdir build && cd build \
	&& cmake -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_GNA=OFF -DENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX=/opt/openvino -DTHREADING=OMP -DENABLE_WHEEL=ON \
		-DENABLE_CLANG_FORMAT=OFF -DENABLE_CPPLINT=OFF -DBUILD_TESTING=OFF .. \
	&& make -j $(nproc --all) \
	&& make install \
	&& python3 -m pip install wheels/*.whl \
	&& cd /tmp && rm -rf openvino

## Prepare LLM test python environment (1/2)
WORKDIR /root
COPY ./requirements.txt /tmp/requirements.txt
RUN python3 -m venv /root/llm_ov_venv && . /root/llm_ov_venv/bin/activate \
	&& python3 -m pip install /opt/openvino/tools/*.whl \
	&& python3 -m pip install -U pip \
	&& python3 -m pip install -r /tmp/requirements.txt

## Build custom LLM ops library
COPY ./custom_ops /root/llm-openvino/custom_ops
WORKDIR /root/llm-openvino/custom_ops
RUN /bin/bash -c ". /root/llm_ov_venv/bin/activate && source /opt/openvino/setupvars.sh \
	&& mkdir build && cd build \
	&& cmake .. && make -j$(nproc --all)"

## Prepare LLM test python environment (2/2)
COPY . /root/llm-openvino
WORKDIR /root/llm-openvino
RUN . /root/llm_ov_venv/bin/activate \
	&& python3 -m pip install -e .

RUN mkdir /cache
ENV HF_HOME /cache

COPY ./entry_point.sh /entry_point.sh
ENTRYPOINT ["/bin/bash", "/entry_point.sh"]
CMD ["/bin/bash"]


