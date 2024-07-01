#!/bin/bash
set -x -e -o pipefail

export NCCL_DEBUG=WARN
CONFIG_PATH=$1
T=`date +%m%d%H%M`

export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

ROOT=/deeplink_afs/yangbo1/easyllm
export PYTHONPATH=$ROOT/llm/models:$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
# Please export them, if you use ceph to load or save model

export ASCEND_RUNTIME=1

echo "START TIME: $(date)"

# pip install peft
# pip install tiktoken
# pip install tensorboardX
# pip install varname
# pip install pandas
MASTER_ADDR=127.0.0.1
MASTER_PORT=6002

NNODES=${NNODES:-'1'}
NODE_RANK=${NODE_RANK:-'0'}
GPUS_PER_NODE=${2:-'1'}

DISTRIBUTED_ARGS="python -m torch.distributed.launch  --nnodes=$NNODES  --node_rank=$NODE_RANK  --master_addr=$MASTER_ADDR  --nproc_per_node=$GPUS_PER_NODE  --master_port=$MASTER_PORT"


export LAUNCHER="$DISTRIBUTED_ARGS"


# export LAUNCHER="python -u -m torch.distributed.run \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#     --rdzv_backend c10d \
#     --max_restarts 0 \
#     --tee 3 \
#     "

export CMD=" \
    $ROOT/llm/runners/ascend_runner.py \
    --config $CONFIG_PATH \
    --launcher torch
    "

echo $CMD

bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"


