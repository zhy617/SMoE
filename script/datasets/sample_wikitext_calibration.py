import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 或 "https://huggingface.co.cn" 或 tuna 镜像

from datasets import load_dataset
from transformers import AutoTokenizer, Qwen2TokenizerFast
import random

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
TOKENIZER_CACHE = "/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B"
SAMPLE_SIZE = 128
MAX_LENGTH = 2048
CACHE_DIR = "/root/fsas/dataset/wikitext"
SAVE_PATH = "/root/SMoE/data/wikitext_calibration.json"

def main() -> None:
    # 加载 wikitext-2-raw-v1 训练集
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=CACHE_DIR)
    # print(f"Loaded {len(ds)} samples.")

    # 随机采样
    samples = random.sample(list(ds), SAMPLE_SIZE)

    # 加载分词器
    tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=TOKENIZER_CACHE)

    calibration_data = []
    for sample in samples:
        text = sample["text"]
        # 编码为 token id
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # 截断或拼接到2048长度
        if len(tokens) > MAX_LENGTH:
            tokens = tokens[:MAX_LENGTH]
        else:
            # 如果不足2048，可以拼接自身或其他样本
            while len(tokens) < MAX_LENGTH:
                tokens += tokens[:(MAX_LENGTH - len(tokens))]
            tokens = tokens[:MAX_LENGTH]
        calibration_data.append(tokens)

    print(f"Processed {len(calibration_data)} calibration samples.")
    # 保存为 json 格式
    with open(SAVE_PATH, "w") as f:
        import json
        json.dump(calibration_data, f)
    print(f"Calibration data saved to {SAVE_PATH}.")

if __name__ == "__main__":
    main()