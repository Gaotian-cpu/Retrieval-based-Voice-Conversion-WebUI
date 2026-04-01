#!/bin/bash

# 🔥 彻底关闭所有警告（包括 NNPACK，100% 有效）
export PYTORCH_DISABLE_NNPACK=1
export TF_CPP_MIN_LOG_LEVEL=3
export KMP_AFFINITY=noverbose
export MKL_SERVICE_FORCE_INTEL=1
export PYTHONWARNINGS="ignore"

CWD=`pwd`

RVC_ROOT="/root/Project/RVC-WebUI"
PYTHON_ROOT="$RVC_ROOT/.venv"
PYTHON_BIN="$PYTHON_ROOT/bin/python3"
TRAIN_SCRIPT="$RVC_ROOT/rvc_train3.py"
DATASET_DIR="$CWD/audio_jiejie"
RESULT_DIR="$CWD/output_rvc"
LOGS_DIR="$RESULT_DIR/my_logs"
CKPT_DIR="$RESULT_DIR/my_weights"

"$PYTHON_BIN" "$TRAIN_SCRIPT" \
        --exp_name test_jiejie \
        --dataset_dir "$DATASET_DIR" \
        --sr 48k \
        --version v2 \
        --total_epoch 50 \
        --batch_size 32 \
        --f0_method "rmvpe" \
        --log_root "$LOGS_DIR" \
        --save_dir "$CKPT_DIR" \
	--if_cache_data_in_gpu 0 \
	--save_every_weights 0 \
	--if_latest 0 
