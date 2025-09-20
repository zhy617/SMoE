hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_30,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------|------:|------|-----:|--------|---|-----:|---|-----:|
|boolq    |      2|none  |     0|acc     |↑  |0.4880|±  |0.0158|
|hellaswag|      1|none  |     0|acc     |↑  |0.3550|±  |0.0151|
|         |       |none  |     0|acc_norm|↑  |0.4410|±  |0.0157|
|rte      |      1|none  |     0|acc     |↑  |0.5235|±  |0.0301|