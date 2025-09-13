from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
from transformers import Qwen2MoeForCausalLM
from typing import cast

from .direct_expert_similarity import (
    generate_and_save_hidden_states,
    analyze_similarity_from_saved_states
)

# ... config ...
BASE_HIDDEN_STATES_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/hidden_states_cache"
SAMPLE_INPUT_FILE = "/root/SMoE/data/qwen/wikitext_calibration.json"
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
MODEL_DIR = "/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B"
SAMPLE_SIZE = 128
MAX_LENGTH = 2048

def main() -> None:
    # ... 加载模型和数据 ...
    print("Loading model...")
    model = cast(Qwen2MoeForCausalLM, AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ))

    # --- 加载数据 ---
    print(f"Loading calibration data from {SAMPLE_INPUT_FILE}...")
    with open(SAMPLE_INPUT_FILE, 'r') as f:
        calibration_data = json.load(f)
    num_samples = len(calibration_data)
    print(f"Found {num_samples} samples.")

    # --- 步骤 1: 为所有样本生成并保存中间结果 ---
    print("\n" + "="*20 + " Step 1: Generating and saving hidden states for all samples " + "="*20)
    for i in tqdm(range(num_samples), desc="Generating hidden states"):
        sample_save_dir = os.path.join(BASE_HIDDEN_STATES_DIR, f"sample_{i}")
        
        if not os.path.exists(sample_save_dir) or not os.listdir(sample_save_dir):
            sample_input_ids = torch.tensor([calibration_data[i]], dtype=torch.long)
            generate_and_save_hidden_states(model, sample_input_ids, sample_save_dir)
        # else:
        #     print(f"Skipping sample {i}, already processed.")

if __name__ == "__main__":
    main()