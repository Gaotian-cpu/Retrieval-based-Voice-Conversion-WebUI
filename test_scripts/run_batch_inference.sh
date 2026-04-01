#!/bin/bash

# ============================================================
# RVC Batch Inference Script
# 批量声音转换 - 所有参数完整暴露
# ============================================================

# 环境配置
PYTHON="python3"
INFER_PY="./rvc_infer.py"

# ==================== 核心参数 ====================
# 模型文件
MODEL_PATH="./output_rvc/my_weights/test_jiejie.pth"

# 索引文件
INDEX_PATH="./output_rvc/my_logs/test_jiejie/added_IVF256_Flat_nprobe_1_test_jiejie_v2.index"

# 批量输入/输出
INPUT_DIR="./input_audios"
OUTPUT_DIR="./output_results"
EXPORT_FORMAT="wav"  # wav / flac / mp3 / m4a

# 变调
TRANSPOSE=0

# 音高提取算法
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
  --mode batch \
  --transpose "$TRANSPOSE" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --export_format "$EXPORT_FORMAT" \
  --index_path "$INDEX_PATH" \
  --f0_method "$F0_METHOD" \
  --resample_sr "$RESAMPLE_SR" \
  --rms_mix_rate "$RMS_MIX_RATE" \
  --protect "$PROTECT" \
  --filter_radius "$FILTER_RADIUS" \
  --index_rate "$INDEX_RATE"

echo "✅ Batch inference done!"
