import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

# 1. 配置参数
# ------------------------------------------------------------------------------------
# !!! 重要: 将此路径替换为您压缩后、包含自定义代码的模型文件夹路径
base_model_path = "path/to/your/compressed/model_with_custom_code" 
# 使用一个公开的高质量小数据集作为示例
dataset_name = "mlabonne/guanaco-llama2-1k" 
# 微调后新模型的保存路径
output_dir = "./qwen-moe-finetuned-recovered"

# 2. 加载数据集
# ------------------------------------------------------------------------------------
dataset = load_dataset(dataset_name, split="train")

# 3. 加载模型和Tokenizer
# ------------------------------------------------------------------------------------
# Qwen的Tokenizer需要设置pad_token
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载模型，因为是恢复性微调，我们不需要量化，直接用bfloat16加载
# trust_remote_code=True 是必须的，因为它会加载您文件夹中的自定义模型代码
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True, 
)

# 4. 配置LoRA参数 (PEFT)
# ------------------------------------------------------------------------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # 对于Qwen2-MoE, 目标模块通常是这些。可以先只用q_proj, v_proj测试
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. 配置训练参数 (TrainingArguments)
# ------------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,        # 根据您的显存调整
    gradient_accumulation_steps=4,      # 实际 batch_size = 1 * 4 = 4
    learning_rate=2e-5,                 # !!! 恢复性微调的关键：低学习率
    num_train_epochs=1,                 # !!! 恢复性微调的关键：只训练1轮
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=50,
    fp16=False, # 如果你的GPU支持bf16, 就用bf16
    bf16=True,  # 如果是A100/H100或40系显卡，强烈建议开启
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    report_to="tensorboard",
)

# 6. 初始化 SFTTrainer 并开始训练
# ------------------------------------------------------------------------------------
# SFTTrainer可以自动处理数据格式化
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # Guanaco数据集的列名是'text'
    max_seq_length=1024,        # 根据您的显存和需求调整
    tokenizer=tokenizer,
    args=training_args,
    packing=True, # 将多个短样本打包成一个长样本，提升效率
)

print("--- 开始微调 ---")
trainer.train()
print("--- 微调完成 ---")


# 7. 合并LoRA权重并保存完整模型
# ------------------------------------------------------------------------------------
print("--- 开始合并 LoRA 权重 ---")
# 首先，卸载 trainer, 释放显存
del trainer
torch.cuda.empty_cache()

# 加载训练好的基座模型和适配器
# 注意：这里需要重新加载，因为SFTTrainer会修改模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
# SFTTrainer 默认将adapter保存在 output_dir/checkpoint-xxx/ 下最新的文件夹
# 或者在训练结束后，adapter会保存在trainer.state.best_model_checkpoint
from peft import PeftModel
# 这里假设最后一个checkpoint是最好的
last_checkpoint = training_args.output_dir + "/checkpoint-..." # 替换为最新的checkpoint文件夹
peft_model = PeftModel.from_pretrained(base_model, last_checkpoint)


# 合并权重
merged_model = peft_model.merge_and_unload()
print("--- 合并完成 ---")

# 保存合并后的完整模型
final_model_path = output_dir + "/final_merged_model"
merged_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"--- 完整模型已保存至: {final_model_path} ---")