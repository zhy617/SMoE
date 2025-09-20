#!/bin/bash
# filepath: /root/SMoE/script/download_data.sh

echo "🚀 下载评估数据集..."

# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 激活虚拟环境
source .venv/bin/activate

# 或者使用 lm_eval 下载
cd libs/lm-evaluation-harness
python -m lm_eval --model hf \
    --model_args pretrained=gpt2 \
    --tasks boolq,rte,hellaswag \
    --limit 1 \
    # --download_only
cd ../..

echo "✅ 数据集下载完成！"