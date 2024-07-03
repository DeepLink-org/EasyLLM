#!/bin/bash
set -x -e -o pipefail

export NCCL_DEBUG=WARN
CONFIG_PATH=$1
T=`date +%m%d%H%M`

export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

ROOT=/root/workspace/Test/deeplink.framework/dipu/mmlab_pack/EasyLLM/
# DS=/ailab_internlm_data/yaoyongqiang/DeepSpeed
DS=/root/workspace/Test/deeplink.framework/dipu/mmlab_pack/DeepSpeed
export PYTHONPATH=$DS:$ROOT/llm/models:$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
# Please export them, if you use ceph to load or save model

export ASCEND_RUNTIME=1

export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0

echo "START TIME: $(date)"

MASTER_ADDR=127.0.0.1
MASTER_PORT=6002

NNODES=${NNODES:-'1'}
NODE_RANK=${NODE_RANK:-'0'}
GPUS_PER_NODE=${2:-'1'}

DISTRIBUTED_ARGS="python -m torch.distributed.launch  --nnodes=$NNODES  --node_rank=$NODE_RANK  --master_addr=$MASTER_ADDR  --nproc_per_node=$GPUS_PER_NODE  --master_port=$MASTER_PORT"


export LAUNCHER="$DISTRIBUTED_ARGS"

export CMD=" \
    $ROOT/llm/runners/base_llm_runner.py \
    --config $CONFIG_PATH \
    --launcher torch
    "

echo $CMD

set -x 
bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"

