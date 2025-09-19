# 自定义参数：2层一组，平均30个专家，范围10-50
python src/qwen/analysis/calculate_redundancy.py --allocate_experts \
    --group_size 3 \
    --target_avg_experts 30 \
    --min_experts 10 \
    --max_experts 50