#!/bin/bash -ex
COMMAND=$@
source /root/llm_ov_venv/bin/activate
source /opt/openvino/setupvars.sh

exec $COMMAND
