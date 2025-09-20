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
```

```bash
git submodule update --init libs/transformers
cd libs/transformers
pip install -e .
cd ../..
```

```bash
git submodule update --init libs/lm-evaluation-harness
cd libs/lm-evaluation-harness
pip install -e .
cd ../..
```



## Model and Data
```bash
bash script/download_models.sh

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
# calculate redundancy and allocate experts
# output: fsas/zhanghongyu/SMoE/qwen/analysis/redundancy.json
bash script/qwen/calculate_redundancy.sh    
```

```bash
# kmeans cluster
# input: fsas/zhanghongyu/SMoE/qwen/analysis/similarity_results
# output: fsas/zhanghongyu/SMoE/qwen/analysis/kmeans_clusters_{N}
# note: change the allocation_file!!!
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


## Evaluate
```bash
# download the evaluation datasets
bash script/download_datasets.sh
```

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

## Result
||ARC_c ↑|ARC_e ↑|MMLU ↑|WinoG ↑|BoolQ ↑|HellaS ↑|RTE  ↑|
|---|---|---|---|---|---|---|---|
|(paper)Qwen1.5-MoE-A2.7B-Chat|0.40|0.71|0.60|0.66|0.81|0.59|0.74|
|(self)Qwen1.5-MoE-A2.7B-Chat|0.397|0.709|0.6016|0.6540|
|(paper)Cluster 45|0.37|0.69|0.53|0.66|0.80|0.56|0.76
|(self)Cluster 45|0.3400|0.6110|0.3884|0.6100|
|(self)Adaptive 45(2 groups)|0.3540|0.6410|0.3588|0.6040|
|(self)Adaptive 45(3 groups)|0.3870|0.6840|0.4778|0.6180|0.7380|0.4750|0.5993|
|(paper)Cluster 30|0.32|0.58|0.38|0.58|0.51|0.46|0.57|
|(self)Cluster 30|0.2390|0.4320|0.2331|0.5280|
|(self)Adaptive 30(2 groups)|0.2650|0.5090|0.2305|0.5390|
|(self)Adaptive 30(3 groups)|0.2470|0.4310|0.2316|0.5360|0.4880|0.3550|0.5235|