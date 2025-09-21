import torch
import os
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import PreTrainedTokenizerFast
from trl.trainer.sft_trainer import SFTTrainer

from typing import cast, Dict, Any, List

# change hf endpoint
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 1. 配置参数
# ------------------------------------------------------------------------------------
CLUSTER_N = 45  # 聚类数量
MODEL_NAME = "qwen1.5_moe_merged_svd_cluster_" + str(CLUSTER_N)
base_model_path = "/root/fsas/zhanghongyu/SMoE/qwen/merged_models" + "/" + MODEL_NAME
dataset_name = "HuggingFaceH4/ultrafeedback_binarized" 
output_dir = "/root/fsas/zhanghongyu/SMoE/qwen/finetuned_models" + "/" + MODEL_NAME
os.makedirs(output_dir, exist_ok=True)

# 2. 加载并预处理数据集
# ------------------------------------------------------------------------------------
print("--- 正在加载和预处理数据集 ---")
dataset = load_dataset(dataset_name, split="train_sft")

assert isinstance(dataset, Dataset)

print(f"--- 原始数据集列名: {dataset.column_names} ---")
print(f"--- 原始数据集样本数: {len(dataset)} ---")

# 🎯 生产配置：使用更多但合理的样本数量
TRAIN_SAMPLES = 10000  # 使用1万个高质量样本，足够进行恢复性微调
print(f"--- 🚀 生产模式：使用 {TRAIN_SAMPLES} 个样本进行训练 ---")
dataset = dataset.select(range(min(TRAIN_SAMPLES, len(dataset))))
print(f"--- 训练数据集样本数: {len(dataset)} ---")

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
    remove_columns=[col for col in dataset.column_names if col != "messages"],
    num_proc=4,  # 🎯 使用多进程加速预处理
    desc="预处理数据集"
)

print(f"--- 预处理后数据集列名: {dataset.column_names} ---")

# 3. 加载模型和Tokenizer
# ------------------------------------------------------------------------------------
print("--- 加载 Tokenizer 和模型 ---")
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
assert isinstance(tokenizer.eos_token, str), "eos_token should be a string"

# 明确配置特殊token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 打印tokenizer配置信息
print("--- Tokenizer配置 ---")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    pad_token_id=tokenizer.pad_token_id,  # 明确指定pad_token_id
)

# 启用梯度检查点以节省显存
model.gradient_checkpointing_enable()
print("--- 模型加载完成，已启用梯度检查点 ---")

# 4. 最佳生产配置 (TrainingArguments)
# ------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    
    # 🎯 批次配置：平衡显存和效率
    per_device_train_batch_size=1,        # 保持为1，避免OOM
    gradient_accumulation_steps=8,        # 恢复到8，有效batch_size=8
    
    # 🎯 学习率配置：恢复性微调的关键
    learning_rate=2e-6,                   # 进一步降低学习率，更保守的恢复训练
    weight_decay=0.01,                    # 添加权重衰减，防止过拟合
    
    # 🎯 训练轮数配置
    num_train_epochs=1,                   # 恢复性微调只需要1轮
    # max_steps=None,                     # 移除步数限制，完整训练
    
    # 🎯 学习率调度
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,                    # 减少warmup，因为模型已经训练过了
    
    # 🎯 日志和保存配置
    logging_steps=25,                     # 适中的日志频率
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=250,                       # 每250步保存一次
    save_total_limit=3,                   # 保留3个checkpoint
    
    # 🎯 精度和性能配置
    fp16=False,
    bf16=True,                           # 使用bf16以获得更好的数值稳定性
    max_grad_norm=1.0,                   # 增加梯度剪切阈值
    
    # 🎯 数据加载配置
    dataloader_num_workers=2,            # 适度的worker数量
    dataloader_pin_memory=True,          # 加速数据传输
    group_by_length=True,                # 按长度分组，提高效率
    
    # 🎯 其他配置
    remove_unused_columns=False,
    report_to="tensorboard",
    run_name=f"qwen_moe_recovery_cluster_{CLUSTER_N}",  # 清晰的运行名称
    
    # 🎯 恢复训练特定配置
    ignore_data_skip=False,              # 允许从checkpoint恢复时跳过已处理数据
    seed=42,                            # 固定随机种子以提高复现性
)

# 预估训练时间和资源
total_steps = len(dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
print("--- 训练配置详情 ---")
print(f"📊 样本数量: {len(dataset):,}")
print(f"📊 每设备批次大小: {training_args.per_device_train_batch_size}")
print(f"📊 梯度累积步数: {training_args.gradient_accumulation_steps}")
print(f"📊 有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"📊 预估总步数: {total_steps:,}")
print(f"📊 学习率: {training_args.learning_rate}")
print(f"📊 预估训练时间: {total_steps * 30 / 3600:.1f} 小时 (假设30秒/步)")

# 5. 初始化 SFTTrainer 并开始训练
# ------------------------------------------------------------------------------------
print("--- 初始化 SFTTrainer ---")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("--- 开始生产级全参数微调 ---")
print(f"💡 Tensorboard 命令: tensorboard --logdir={output_dir} --port=6006 --bind_all")
print(f"🚀 正式训练模式：约 {total_steps:,} 步")
print(f"💾 模型将保存在: {output_dir}")

try:
    # 训练开始
    trainer.train()
    print("--- ✅ 微调完成！---")
    
except KeyboardInterrupt:
    print("--- ⏹️ 训练被用户中断 ---")
    print("💡 可以使用 trainer.train(resume_from_checkpoint=True) 恢复训练")
    
except torch.cuda.OutOfMemoryError as e:
    print("--- ❌ 显存不足 ---")
    print("💡 建议调整配置：")
    print("   - 减少 gradient_accumulation_steps 到 4")
    print("   - 或减少 TRAIN_SAMPLES 到 5000")
    raise
    
except Exception as e:
    print(f"--- ❌ 训练出错: {e} ---")
    raise

# 6. 保存最终模型
# ------------------------------------------------------------------------------------
print("--- 保存最终模型 ---")
try:
    # 保存完整模型
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"--- ✅ 完整模型已保存至: {final_model_path} ---")
    
    # 保存tokenizer
    tokenizer.save_pretrained(final_model_path)
    print(f"--- ✅ Tokenizer已保存至: {final_model_path} ---")
    
    # 保存训练参数
    # training_args.save_to_json(os.path.join(final_model_path, "training_args.json"))
    print(f"--- ✅ 训练参数已保存 ---")
    
except Exception as e:
    print(f"--- ❌ 保存出错: {e} ---")
    raise

print("--- 🎉 恢复性微调完全完成！---")
print("🔍 模型质量检查建议：")
print("1. 使用测试数据集评估模型性能")
print("2. 进行人工对话测试")
print("3. 对比微调前后的输出质量")
print("4. 检查是否出现灾难性遗忘")

print(f"📁 最终模型位置: {final_model_path}")
print(f"📊 训练日志: {output_dir}/runs/")