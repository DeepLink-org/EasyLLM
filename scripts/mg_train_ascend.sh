#!/bin/bash
set -x -e -o pipefail

export NCCL_DEBUG=WARN
T=`date +%m%d%H%M`

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
export COMBINED_ENABLE=1

ROOT=/data/yujinbiao/easyllm
export PYTHONPATH=/data/yujinbiao/easyllm/llm/models:$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
# Please export them, if you use ceph to load or save model

export ASCEND_RUNTIME=1

echo "START TIME: $(date)"

# for slurm
# export CMD=" python \
#     $ROOT/llm/runners/base_llm_runner.py \
#     --config $2 \
#     --launcher slurm" 

# echo $CMD

# srun --partition=$1 --mpi=pmi2 -N 1 -n8 --gres=gpu:8 --quotatype=spot --cpus-per-task=16  bash -c "$CMD"  2>&1 | tee -a mg_train_log.txt

# echo "END TIME: $(date)"

# for torch
# so processes know who to talk to
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # for multi node

pip install peft
pip install tiktoken
pip install tensorboardX
pip install varname
MASTER_ADDR=127.0.0.1
MASTER_PORT=6002

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
export LAUNCHER="torchrun \
    $DISTRIBUTED_ARGS \
    "

export CMD=" \
    $ROOT/llm/runners/ascend_runner.py \
    --config $1 \
    --launcher torch
    "

echo $CMD

bash -c "$LAUNCHER $CMD"  2>&1 | tee /data/yujinbiao/logs/llama_7b_ascend_debug.txt

echo "END TIME: $(date)"



