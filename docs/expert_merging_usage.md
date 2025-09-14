# 专家合并使用指南

## 概述
本文档说明如何进行专家合并以及如何使用合并后的模型。

## 运行专家合并

```bash
cd /root/SMoE
bash script/qwen/merge_expert.sh
```

## 关于 Tokenizer

### 为什么不保存 Tokenizer？
1. **避免兼容性问题**：Tokenizer 在不同环境下可能有兼容性问题
2. **减少依赖**：专家合并主要关注模型权重，不涉及词汇表变化
3. **简化流程**：避免 tokenizer 加载失败导致整个流程中断

### 如何使用合并后的模型？

#### 方法1：使用原始模型的 Tokenizer
```python
from transformers import AutoTokenizer, Qwen2MoeForCausalLM

# 加载原始模型的分词器
tokenizer = AutoTokenizer.from_pretrained(
    "/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B",
    trust_remote_code=True
)

# 加载合并后的模型
merged_model = Qwen2MoeForCausalLM.from_pretrained(
    "/path/to/merged/model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 正常使用
inputs = tokenizer("你好，世界！", return_tensors="pt")
outputs = merged_model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### 方法2：手动复制 Tokenizer 文件
如果你需要一个完整的模型包，可以手动复制 tokenizer 文件：

```bash
# 从原始模型目录复制 tokenizer 文件到合并模型目录
cp /root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B/tokenizer* /path/to/merged/model/
cp /root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B/vocab.json /path/to/merged/model/  # 如果存在
cp /root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B/merges.txt /path/to/merged/model/  # 如果存在
```

## 合并配置

### 可配置参数
在 `merge_experts.py` 的 `main()` 函数中可以配置：

- `MODEL_PATH`: 原始模型路径
- `CLUSTER_DIR`: 聚类结果目录
- `RESULT_DIR`: 分析结果目录
- `OUTPUT_DIR`: 输出目录
- `TARGET_LAYERS`: 要合并的层列表
- `MERGING_METHOD`: 合并方法 ("svd" 或 "frequency")

### 示例配置
```python
# 合并所有MoE层
TARGET_LAYERS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

# 只合并部分层
TARGET_LAYERS = [1, 3, 5]

# 使用不同的合并方法
MERGING_METHOD = "frequency"  # 基于激活频率
MERGING_METHOD = "svd"        # 基于SVD子空间对齐
```

## 验证合并结果

### 检查模型大小
```python
import torch
from transformers import Qwen2MoeForCausalLM

# 加载模型
model = Qwen2MoeForCausalLM.from_pretrained("/path/to/merged/model")

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 检查MoE层的专家数量
for i, layer in enumerate(model.model.layers):
    if hasattr(layer.mlp, 'experts'):
        num_experts = len(layer.mlp.experts)
        print(f"Layer {i}: {num_experts} experts")
```

### 简单推理测试
```python
# 简单的文本生成测试
inputs = tokenizer("请介绍一下人工智能", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_length=200, 
        do_sample=True, 
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", result)
```

## 故障排除

### 常见问题
1. **内存不足**：使用 `device_map="cpu"` 在CPU上进行合并
2. **文件缺失**：确保聚类结果和激活频率文件存在
3. **路径错误**：检查所有路径配置是否正确

### 日志信息
合并过程会生成详细的日志信息，包括：
- 每层的专家数量变化
- 压缩比统计
- 错误和警告信息

关注这些信息可以帮助诊断问题。