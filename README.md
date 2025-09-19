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

```bash
# change the config json
bash script/qwen/update_config.sh
```


## evaluate
```bash
# evaluate merged model
# input: fsas/zhanghongyu/SMoE/qwen/merged_models
# output: fsas/zhanghongyu/SMoE/qwen/eval_results
bash script/qwen/evaluate_benchmark.sh
# change the "CLUSTER_N" in the script/code to evaluate different merged models
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
