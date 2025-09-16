# Timeline
## 9.12
### Done
- 阅读论文，学习并理解论文中的算法细节，重温奇异值分解
- 搭建环境，安装部分依赖（完整依赖待进一步补充）
- 下载 Qwen1.5-MoE-A2.7B 模型权重
- 从 wikitext 中抽取部分数据，保存为 jsonl 格式
- 尝试实现hook函数，强制路由到指定专家 / 统计激活频率 （实现中，有点困难）

### Questions
- 论文中提了嘴 Adaptive Expert Clustering，但没有具体的细节，想问这部分是怎么实现的。比如统一合并的层数（2层还是else）；多层合并后的专家怎么处理（比如第0层和第1层的专家合并后，结果放在哪一层）；Adaptive 方法中每层的压缩比怎么确定（论文中说按照每层的diversity 来确定，但没有给具体量化标准）。

- 我现在要计算激活频率和相似度，需要实现hook函数。有没有实现过类似的hook函数？我不太清楚怎么实现，现在是按照 gemini 的思路来写的，刚刚copy过来，没有测试过。

  代码在 `src/qwen/analysis` 中

- 有没有关于项目结构的建议？因为我现在是按照 gemini 的项目结构来组织的，但不确定是否适合这个项目。

  现在还只是针对 qwen 1.5 的实验，后续可能会有更多模型和数据集，想问下项目结构方面有没有建议。

## 9.13
### Done
- 参考 plh 师兄的建议，修改了方法，先 prefill 一次，保存每层的 hidden states，然后再手动执行到指定层，计算激活频率和相似度
- 翻阅 transformers.models.qwen2_moe.modeling_qwen2_moe，理解 Qwen2MoeDecoderLayer 的实现，并仿照 forward 方法，手动实现了 attention 和 mlp 的计算
- 成功保存了每层的 hidden states
- 计算激活频率和相似度 (实现中，较简单)

### Questions
- None

## 9.14
### Done
- 完成了激活频率和相似度的计算
- 实现了 kmeans 聚类
- 实现了专家合并：SVD 
- 保存了合并后的模型

### Questions
- None


## 9.15
### Done
- 实现了模型的压缩
```
# 路径
path = /root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_CLUSTER_8
# name
name = qwen1.5_moe_merged_svd_CLUSTER_8
# info
path = /root/fsas/zhanghongyu/SMoE/qwen/merged_models/qwen1.5_moe_merged_svd_CLUSTER_8/merge_info.json
```
- 修复了bug
  - 激活专家数由 2 -> 4

- 完成了评测脚本
- 评测了原始模型(Qwen1.5-MoE-A2.7B, 60 experts)和合并后的模型(Qwen1.5-MoE-A2.7B, cluster 30)，发现相去甚远

  结果见 `eval-results/qwen/qwen1.5_moe_merged_svd_cluster_30.md`
- 为了验证合并代码的正确性，尝试了 cluster 60，发现结果与原始模型几乎一致，代表合并代码大概率是正确的

  结果见 `eval-results/qwen/qwen1.5_moe_merged_svd_cluster_60.md`


## 9.16
### Done
- 昨天发现 30 个专家的模型效果与论文差距较大。发现论文中测试的是 Qwen1.5-MoE-A2.7B-Chat，故重新下载了该模型，并进行了合并和评测，测试了 45 和 30 个专家的模型

  结果见 `eval-results/qwen/qwen1.5_moe_merged_svd_cluster_45.md` 和 `eval-results/qwen/qwen1.5_moe_merged_svd_cluster_30.md`

||ARC_c ↑|ARC_e ↑|MMLU ↑|WinoG ↑|
|---|---|---|---|---|
|(paper)Qwen1.5-MoE-A2.7B-Chat|0.40|0.71|0.53|0.66|
|(self)Qwen1.5-MoE-A2.7B-Chat|0.397|0.709|0.6016|0.6540|
|(paper)Cluster 45|0.37|0.69|0.53|0.66|
|(self)Cluster 45|0.3400|0.6110|0.3884|0.6100|
|(paper)Cluster 30|0.32|0.58|0.38|0.58|
|(self)Cluster 30|0.2390|0.4320|0.2331|0.5280|

### Questions
- 我已经实现了单层的专家合并，实测后发现与论文结果差距较大。请问多层合并具体应该怎么实现。我现在困惑的是
  - 多层合并后，专家权重应该放在哪一层？（比如第0层和第1层的专家合并后，结果放在第0层还是第1层？）
  - 多层合并后，路由器应该怎么处理？（比如第0层和第1层的专家合并后，假设专家放在第0层，那第0层的路由怎么确定？第1层的路由又怎么确定？）

或者可能是我理解错了，实际上还是单层的合并，只是每层的合并数量不一样？那怎么在相邻两层中调配专家数量？