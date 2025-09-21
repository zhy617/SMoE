import torch
import os
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, Qwen2Tokenizer
from transformers import PreTrainedTokenizerFast
from trl.trainer.sft_trainer import SFTTrainer

from typing import cast, Dict, Any, List

# change hf endpoint
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 配置参数
# ------------------------------------------------------------------------------------
# !!! 重要: 将此路径替换为您压缩后、包含自定义代码的模型文件夹路径
CLUSTER_N = 45  # 聚类数量
MODEL_NAME = "qwen1.5_moe_merged_svd_cluster_" + str(CLUSTER_N)
base_model_path = "/root/fsas/zhanghongyu/SMoE/qwen/merged_models" + "/" + MODEL_NAME
# 使用一个公开的高质量数据集
dataset_name = "HuggingFaceH4/ultrafeedback_binarized" 
# 微调后新模型的保存路径
output_dir = "/root/fsas/zhanghongyu/SMoE/qwen/finetuned_models" + "/" + MODEL_NAME
os.makedirs(output_dir, exist_ok=True)

# 2. 加载并预处理数据集
# ------------------------------------------------------------------------------------
# UltraFeedback 数据集需要特殊处理
print("--- 正在加载和预处理数据集 ---")
dataset = load_dataset(dataset_name, split="train_sft")

assert isinstance(dataset, Dataset)

# 显示原始数据集信息
print(f"--- 原始数据集列名: {dataset.column_names} ---")
print(f"--- 原始数据集样本数: {len(dataset)} ---")

# 🎯 限制数据集大小进行测试
TEST_SAMPLES = 100  # 先用100个样本测试
print(f"--- 🧪 测试模式：限制数据集为 {TEST_SAMPLES} 个样本 ---")
dataset = dataset.select(range(min(TEST_SAMPLES, len(dataset))))
print(f"--- 测试数据集样本数: {len(dataset)} ---")

# 预处理函数：只保留 messages 列
def preprocess_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    只保留 messages 列，移除其他所有列
    """
    return {"messages": example["messages"]}

# 应用预处理，只保留 messages 列
print("--- 开始预处理数据集，只保留 messages 列 ---")
dataset = dataset.map(
    preprocess_dataset, 
    remove_columns=[col for col in dataset.column_names if col != "messages"]
)

print(f"--- 预处理后数据集列名: {dataset.column_names} ---")

# 打印一个样本查看格式
print("--- 样本示例 ---")
sample = dataset[0]
print(f"Messages 类型: {type(sample['messages'])}")
print(f"Messages 内容: {sample['messages'][:2]}...")  # 只显示前2条消息

# 3. 加载模型和Tokenizer
# ------------------------------------------------------------------------------------
# Qwen的Tokenizer需要设置pad_token
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
assert isinstance(tokenizer.eos_token, str), "eos_token should be a string"
tokenizer.pad_token = tokenizer.eos_token

# 加载模型，因为是恢复性微调，我们不需要量化，直接用bfloat16加载
# trust_remote_code=True 是必须的，因为它会加载您文件夹中的自定义模型代码
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True, 
)
# 对于全参数微调，建议启用梯度检查点以节省显存
model.gradient_checkpointing_enable()

# 4. 配置训练参数 (TrainingArguments) - 针对测试优化
# ------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,        # 保持最小batch size
    gradient_accumulation_steps=2,        # 🎯 减少累积步数：2而不是8
    learning_rate=5e-6,                   # 全参数微调的低学习率
    num_train_epochs=1,                   # 只训练1轮
    max_steps=10,                         # 🎯 限制最大步数，用于快速测试
    lr_scheduler_type="cosine",
    logging_steps=2,                      # 🎯 更频繁的日志记录
    save_strategy="steps",               
    save_steps=5,                         # 🎯 更频繁的保存，测试保存功能
    save_total_limit=2,                 
    fp16=False, 
    bf16=True,                          
    max_grad_norm=0.3,
    warmup_ratio=0.1,                     # 🎯 增加warmup比例
    group_by_length=False,                # 🎯 关闭分组以简化处理
    report_to="tensorboard",
    dataloader_num_workers=0,             # 🎯 减少worker数量
    remove_unused_columns=False,          # 🎯 保留所有列以避免潜在问题
)

print("--- 训练配置 ---")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Max steps: {training_args.max_steps}")
print(f"有效 batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# 5. 初始化 SFTTrainer 并开始训练
# ------------------------------------------------------------------------------------
print("--- 初始化 SFTTrainer ---")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("--- 开始全参数微调 ---")
print(f"💡 Tensorboard 日志目录: {output_dir}")
print(f"💡 启动命令: tensorboard --logdir={output_dir} --port=6006 --bind_all")
print(f"🎯 测试模式：只训练 {training_args.max_steps} 步")

try:
    trainer.train()
    print("--- ✅ 微调完成 ---")
except Exception as e:
    print(f"--- ❌ 训练出错: {e} ---")
    raise

# 6. 保存最终模型
# ------------------------------------------------------------------------------------
print("--- 开始保存最终模型 ---")
try:
    final_model_path = os.path.join(output_dir, "test_model")  # 🎯 保存到子目录
    trainer.save_model(final_model_path)
    print(f"--- ✅ 完整模型已保存至: {final_model_path} ---")
except Exception as e:
    print(f"--- ❌ 保存出错: {e} ---")
    raise

print("--- 🎉 测试完成！---")
print("如果一切正常，可以调整以下参数进行正式训练：")
print("1. 增加 TEST_SAMPLES 到更大的数值")
print("2. 移除或增加 max_steps 限制")
print("3. 调整 gradient_accumulation_steps 到 8")
print("4. 调整 save_steps 到 200")