from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
from transformers import Qwen2MoeForCausalLM
from typing import cast

from .direct_expert_similarity import (
    generate_and_save_hidden_states,
    get_expert_activation_from_saved_states,
    calculate_expert_similarity_matrix,
)

# ... config ...
BASE_HIDDEN_STATES_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/hidden_states_cache"
RESULT_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/analysis_results"
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

    # --- 步骤 2: 从所有样本的保存结果进行分析并聚合 ---
    print("\n" + "="*20 + " Step 2: Analyzing all samples and calculating similarity " + "="*20)
    
    # 定义要分析的MoE层 (Qwen1.5-MoE-A2.7B的MoE层通常在奇数层)
    layers_to_analyze = list(range(24))  # 你可以按需修改这个列表
    
    # 为每一层初始化一个累加器
    aggregated_similarity = {layer_idx: torch.zeros(60, 60) for layer_idx in layers_to_analyze}  # Qwen1.5-MoE有60个专家
    layer_sample_counts = {layer_idx: 0 for layer_idx in layers_to_analyze}  # 记录每层成功处理的样本数
    
    for i in tqdm(range(num_samples), desc="Analyzing samples"):
        sample_save_dir = os.path.join(BASE_HIDDEN_STATES_DIR, f"sample_{i}")
        
        for layer_idx in layers_to_analyze:
            try:
                # 获取专家激活
                expert_activations = get_expert_activation_from_saved_states(
                    model, 
                    sample_save_dir, 
                    target_moe_layer_idx=layer_idx
                )
                
                # 计算相似度矩阵 (根据论文公式)
                similarity_matrix = calculate_expert_similarity_matrix(expert_activations)
                
                # 将结果累加到对应层的累加器上
                aggregated_similarity[layer_idx] += similarity_matrix.cpu()
                layer_sample_counts[layer_idx] += 1
                
            except Exception as e:
                print(f"Warning: Failed to process sample {i} for layer {layer_idx}: {e}")
                continue
    
    # --- 步骤 3: 计算平均相似度并保存结果 ---
    print("\n" + "="*20 + " Step 3: Computing average similarity and saving results " + "="*20)
    results_dir = os.path.join(RESULT_DIR, "similarity_results")
    os.makedirs(results_dir, exist_ok=True)
    
    for layer_idx, total_sim in aggregated_similarity.items():
        sample_count = layer_sample_counts[layer_idx]
        if sample_count > 0:
            avg_sim_matrix = total_sim / sample_count
            print(f"\n--- Average Similarity Matrix for MoE Layer {layer_idx} (based on {sample_count} samples) ---")
            torch.set_printoptions(precision=4, sci_mode=False)
            print(avg_sim_matrix)
            torch.set_printoptions(profile="default")
            
            # 保存相似度矩阵到文件
            result_path = os.path.join(results_dir, f"avg_similarity_matrix_layer_{layer_idx}.pt")
            torch.save(avg_sim_matrix, result_path)
            print(f"Saved average similarity matrix for layer {layer_idx} to {result_path}")
        else:
            print(f"Warning: No valid samples processed for layer {layer_idx}")
    
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    main()