from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen1.5-MoE-A2.7B"
# save_dir = "/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B"  # 自定义保存路径

# # Download tokenizer
# print("Downloading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir, mirror="tuna")

# # Download model
# print("Downloading model...")
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_dir, mirror="tuna")

# print("Download complete.")

model_name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
save_dir = "/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B-Chat"  # 自定义保存路径

# Download tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=save_dir, mirror="tuna")

# Download model
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=save_dir, mirror="tuna")