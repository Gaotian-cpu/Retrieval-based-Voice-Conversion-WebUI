#!/bin/bash

# ============================================================
# RVC Single Inference Script
# 单音频声音转换 - 所有参数完整暴露
# ============================================================

# 环境配置
CWD=`pwd`
RVC_ROOT="/root/Project/RVC-WebUI"
PYTHON_VENV="$RVC_ROOT/.venv"
PYTHON="$PYTHON_VENV/bin/python3"
INFER_PY="$RVC_ROOT/rvc_inference.py"
ASSETS_ROOT="$CWD/output_rvc"

# ==================== 核心参数 ====================
# 模型文件
MODEL_PATH="$ASSETS_ROOT/my_weights/test_jiejie.pth"

# 索引文件
INDEX_PATH="$ASSETS_ROOT/my_logs/test_jiejie/added_IVF256_Flat_nprobe_1_test_jiejie_v2.index"

# 输入音频
AUDIO_PATH="$CWD/wangdamao-jinwanbuxiangshui.flac"
OUTPUT_PATH="$ASSETS_ROOT/inference/test_output.wav"

# 变调：0=不变，12=升八度，-12=降八度
TRANSPOSE=0

# 音高提取算法：pm / harvest / crepe / rmvpe
F0_METHOD="rmvpe"

# ==================== 高级参数 ====================
RESAMPLE_SR=0
RMS_MIX_RATE=0.25
PROTECT=0.33
FILTER_RADIUS=3
INDEX_RATE=0.75

# ============================================================
# 执行命令（所有参数齐全）
# ============================================================
$PYTHON $INFER_PY \
  --model_path "$MODEL_PATH" \
  --mode single \
  --transpose "$TRANSPOSE" \
  --audio_path "$AUDIO_PATH" \
  --output_path "$OUTPUT_PATH" \
  --index_path "$INDEX_PATH" \
  --f0_method "$F0_METHOD" \
  --resample_sr "$RESAMPLE_SR" \
  --rms_mix_rate "$RMS_MIX_RATE" \
  --protect "$PROTECT" \
  --filter_radius "$FILTER_RADIUS" \
  --index_rate "$INDEX_RATE"

echo "✅ Single inference done!"
