#!/bin/bash

# 评估脚本
set -e

# 配置
ORIGINAL_MODEL="/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B"
COMPRESSED_MODEL="/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_CLUSTER_8"
OUTPUT_DIR="/root/fsas/zhanghongyu/SMoE/qwen/evaluation_results"
TASKS="hellaswag,winogrande,arc_easy,arc_challenge,piqa"
BATCH_SIZE=8

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "🚀 开始评估模型..."
echo "原始模型: $ORIGINAL_MODEL"
echo "压缩模型: $COMPRESSED_MODEL"
echo "任务: $TASKS"

# 进入lm-evaluation-harness目录
cd /root/SMoE/lm-evaluation-harness

echo ""
echo "📊 评估压缩后的模型..."
python -m lm_eval --model hf \
    --model_args pretrained=$COMPRESSED_MODEL,trust_remote_code=True,torch_dtype=auto \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --output_path $OUTPUT_DIR/compressed_model_results.json

echo ""
echo "📊 评估原始模型（对比基线）..."
python -m lm_eval --model hf \
    --model_args pretrained=$ORIGINAL_MODEL,trust_remote_code=True,torch_dtype=auto \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --device cuda \
    --output_path $OUTPUT_DIR/original_model_results.json

echo ""
echo "✅ 评估完成！结果保存在: $OUTPUT_DIR"
echo ""
echo "📈 查看结果:"
echo "压缩模型: $OUTPUT_DIR/compressed_model_results.json"
echo "原始模型: $OUTPUT_DIR/original_model_results.json"