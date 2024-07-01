# export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
bash scripts/mg_train_ascend.sh configs/ascend/my_llama_7b_ascend.yaml 16 2>&1 | tee logs/llama_7b_ascend_debug.txt