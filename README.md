# SMoE
> [Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging](https://arxiv.org/abs/2506.23266)

以下为自用，暂时仅仅支持 Qwen1.5-MoE-A2.7B
## Venv
```bash
apt-get update
apt install python3.10-venv
python3 -m venv .venv

source .venv/bin/activate
```
### Troubleshooting
```bash
# 重建虚拟环境（终极方案）
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Install
```bash
bash script/init.sh
pip install -r requirements.txt
# pip install git+https://github.com/huggingface/transformers
git submodule update --init lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Model and Data
```bash
bash script/download_models.sh

# bash script/download_data.sh

# output: data/qwen/wikitext_calibration.json
bash script/prepare_data.sh # it takes 1-2 mins
```

## Run
### Qwen1.5-MoE-A2.7B
```bash
# calculate expert similarity and frequency
# intermediate result dir: fsas/zhanghongyu/SMoE/qwen/hidden_states_cache
# analysis result dir: fsas/zhanghongyu/SMoE/qwen/analysis
bash script/qwen/calculate_simi_freq.sh
```

```bash
# kmeans cluster
# input: fsas/zhanghongyu/SMoE/qwen/analysis/similarity_results
# output: fsas/zhanghongyu/SMoE/qwen/analysis/kmeans_clusters_{N}
bash script/qwen/kmeans_cluster.sh
```

```bash
# merge experts
# input: fsas/zhanghongyu/SMoE/qwen/analysis/kmeans_clusters_{N}
#        fsas/zhanghongyu/SMoE/qwen/analysis
# output: fsas/zhanghongyu/SMoE/qwen/merged_models
bash script/qwen/merge_expert.sh
```

## evaluate
```bash
# evaluate merged model
# input: fsas/zhanghongyu/SMoE/qwen/merged_models
# output: fsas/zhanghongyu/SMoE/qwen/eval_results
bash script/qwen/evaluate_benchmark.sh
# change the "CLUSTER_N" in the script/code to evaluate different merged models
```


## Structure
```
/root/fsas/zhanghongyu/SMoE/  (项目根目录)
├── data/                       # 原始数据集
│   └── wikitext_calibration.json
├── base_models/                # 原始 HF 模型
│   └── Qwen1.5-MoE-A2.7B-Chat/
├── shared_cache/               # [关键] 存放提取的中间特征 (以此避免重复计算)
│   └── Qwen1.5-MoE-A2.7B_wikitext_128/  (命名规则: 模型名_数据名_采样数)
│       ├── hidden_states_layer_0.pt
│       ├── router_logits_layer_0.pt
│       └── activation_frequency.pt
└── experiments/                # [核心] 具体实验流水线
    └── exp_20251125_clustering_test/  (按 实验名 或 时间戳 分隔)
        ├── config.json                # 记录本次实验的所有参数
        ├── analysis/                  # 聚类与分析结果
        │   ├── kmeans_k30/            # 聚类中心
        │   └── similarity_matrix.pt
        ├── checkpoints/               # [保存] 合并后的模型权重
        │   ├── expert_svd_router_avg_k30/
        │   └── expert_avg_router_avg_k30/
        └── evaluation/                # [新增] 测试 Pipeline 的结果
            ├── expert_svd_router_avg_k30/
            │   ├── entropy_analysis.json  # 路由熵
            │   ├── load_balance.png       # 负载均衡图
            │   └── perplexity.log         # PPL 测试
            └── summary_report.csv         # 所有变体的对比汇总
```

## GPU Memory Consumption
### Qwen1.5-MoE-A2.7B
|  | calculate_simi_freq | kmeans_cluster | merge_expert (N=30) | merge_expert (N=60) |evaluate (N=30) |
|-------|---------------------|----------------|---------------------|------------------|-------------------|
| GPU Memory (GB) |                     |                |     |    110G            |                  |
| 4090(48G)   | 1 | 1 | 4 | 1 |


| cluster number                    | "mmlu,winogrande,arc_easy,arc_challenge"|
|--------------------------|-----------------------------------------|
| original model         | 32G                                     |
| cluster N=30            | 20G                                     |
