# export DIPU_DUMP_OP_ARGS=2
bash scripts/mg_train_ascend.sh configs/megatron/custom_llama_7b_sft.yaml 16 2>&1 | tee llama_7b_sft.log    
# bash scripts/mg_train_ascend.sh configs/megatron/custom.yaml 1 2>&1 | tee llama_lora.log    