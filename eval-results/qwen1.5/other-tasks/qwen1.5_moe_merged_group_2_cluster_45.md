hf (pretrained=/root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_cluster_45,trust_remote_code=True), gen_kwargs: (None), limit: 1000.0, num_fewshot: 0, batch_size: 4
|  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------|------:|------|-----:|--------|---|-----:|---|-----:|
|boolq    |      2|none  |     0|acc     |↑  |0.6730|±  |0.0148|
|hellaswag|      1|none  |     0|acc     |↑  |0.4610|±  |0.0158|
|         |       |none  |     0|acc_norm|↑  |0.6090|±  |0.0154|
|rte      |      1|none  |     0|acc     |↑  |0.6245|±  |0.0291|