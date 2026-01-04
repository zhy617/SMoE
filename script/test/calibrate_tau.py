import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import json
import os
import sys
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock, Qwen2MoeMLP, Qwen2MoeForCausalLM, Qwen2MoeDecoderLayer
from typing import cast, List, Dict, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# print(project_root)

from src import config

# --- 1. 配置 ---
# 从 evaluate_benchmark.sh 中提取的路径信息
MODEL_NAME = "Qwen/expert_svd_router_avg_k45"
MODEL_PATH = "/root/fsas/zhanghongyu/LAMoE/models/Qwen/expert_svd_router_avg_k45"
OUTPUT_DIR = os.path.join(config.EVALUATE_DIR, "calibration_results")

# 验证的配置
VALIDATION_DATASET = "wikitext"
VALIDATION_SUBSET = "wikitext-2-raw-v1"
NUM_VALIDATION_SAMPLES = 256
SEQUENCE_LENGTH = config.MAX_LENGTH
BATCH_SIZE = 4 # 根据您的 GPU 显存调整
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 搜索范围
TAU_RANGE = np.arange(1.9, 2.1, 0.01).tolist() # 例如 [0.0, 0.1, 0.2, ..., 1.5]

# --- 2. 辅助函数和类 ---
class LogitAdjustmentHook:
    """
    一个 PyTorch Hook 类，用于在 router 的 forward pass 后动态调整 logits。
    """
    def __init__(self, num_experts, device):
        self.tau = 0.0
        self.log_freqs = torch.zeros(num_experts, device=device)

    def set_params(self, tau, expert_freqs):
        self.tau = tau
        # 避免 log(0)
        self.log_freqs = torch.log(expert_freqs + 1e-9).to(DEVICE)

    def __call__(self, module, input, output):
        # output 是 router 的输出，通常是一个元组 (router_logits, ...)
        original_logits = output
        
        # 应用 Logit Adjustment
        adjusted_logits = original_logits + self.tau * self.log_freqs
        
        return adjusted_logits

def prepare_dataset(dataset_name, subset, num_samples, tokenizer, seq_length):
    """加载并预处理数据集"""
    dataset = load_dataset(
        path=dataset_name, 
        name=subset, 
        split="test", 
        cache_dir=config.DATASET_CACHE_DIR,
    )
    text_list = [item['text'] for item in dataset.select(range(num_samples))]
    
    all_tokens = []
    for text in text_list:
        if text:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            
    # 将所有文本拼接后，按固定长度切块
    token_chunks = []
    for i in range(0, len(all_tokens), seq_length):
        chunk = all_tokens[i:i+seq_length]
        if len(chunk) == seq_length:
            token_chunks.append({"input_ids": chunk})
            
    return token_chunks

# --- 3. 主逻辑 ---

def main():
    print("🚀 开始 Post-hoc Calibration 流程...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 步骤 1: 加载模型和分词器 ---
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = cast(Qwen2MoeForCausalLM, AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    ))
    model.eval()
    
    try:
        num_experts = model.config.num_experts
        print(f"✅ 从模型配置中获取专家数量: {num_experts}")
    except AttributeError:
        print("❌ 错误: 无法从模型配置中获取专家数量。")
        return

    # --- 步骤 2: 加载每层独立的专家频率 ---
    print(f"\n📊 从 '{config.FREQ_RESULT_DIR}' 加载各层独立的专家激活频率...")
    
    layer_specific_freqs: Dict[int, torch.Tensor] = {}
    layers_to_process = config.TARGET_LAYERS
    if not layers_to_process:
        print("❌ 错误: config.TARGET_LAYERS 为空，请指定要分析的层。")
        return
        
    print(f"将为以下层加载频率: {layers_to_process}")

    for layer_idx in layers_to_process:
        freq_file_path = os.path.join(config.FREQ_RESULT_DIR, f"activation_frequency_layer_{layer_idx}.pt")
        
        if not os.path.exists(freq_file_path):
            print(f"⚠️ 警告: 频率文件不存在，跳过层 {layer_idx}: {freq_file_path}")
            continue

        freq_data = torch.load(freq_file_path, map_location=DEVICE)
        
        if 'activation_counts' not in freq_data:
            print(f"⚠️ 警告: 文件 '{freq_file_path}' 中没有找到 'activation_counts' 键，跳过层 {layer_idx}。")
            continue
            
        layer_counts = freq_data['activation_counts']
        
        if layer_counts.shape[0] != num_experts:
            print(f"❌ 错误: 第 {layer_idx} 层的频率张量维度 ({layer_counts.shape[0]}) 与模型专家数量 ({num_experts}) 不匹配。")
            return
        
        if layer_counts.sum() > 0:
            layer_specific_freqs[layer_idx] = layer_counts.float() / layer_counts.sum()
        else:
            print(f"⚠️ 警告: 第 {layer_idx} 层总激活计数为 0，将无法使用。")

    if not layer_specific_freqs:
        print("❌ 错误: 未能成功加载任何层的有效频率数据。请检查 FREQ_RESULT_DIR 路径和文件内容。")
        return
    
    print(f"✅ 成功为 {len(layer_specific_freqs)} 个层加载了频率数据。")

    # --- 步骤 3: 为每个目标层注册 Hook ---
    hooks: List[Tuple[int, LogitAdjustmentHook]] = []
    handles = []
    print("\n🔧 为每个目标 MoE 层注册 Hook...")
    for i, layer in enumerate(model.model.layers):
        # 只为成功加载了频率的层注册hook
        if i in layer_specific_freqs:
            try:
                router_module = cast(Qwen2MoeDecoderLayer, layer).mlp.gate
                hook = LogitAdjustmentHook(num_experts, DEVICE)
                handle = router_module.register_forward_hook(hook)
                
                hooks.append((i, hook)) # 保存层索引和hook实例
                handles.append(handle)
                print(f"  - 已在第 {i} 层注册 Hook。")
            except AttributeError:
                print(f"⚠️ 警告: 无法在第 {i} 层找到 'mlp.gate'，跳过该层。")
    
    if not handles:
        print("❌ 错误: 未能成功注册任何 Hook。")
        return

    # --- 步骤 4: 准备验证集 ---
    validation_data = prepare_dataset(
        VALIDATION_DATASET, VALIDATION_SUBSET, NUM_VALIDATION_SAMPLES, tokenizer, SEQUENCE_LENGTH
    )
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE)

    # --- 步骤 5: Grid Search tau 并评估 PPL 和 Entropy ---
    print(f"\n🔍 开始在验证集上搜索最佳 tau，范围: {TAU_RANGE}")
    results = []

    for tau in TAU_RANGE:
        # 为每个hook设置其对应层的频率和当前的tau
        for layer_idx, hook in hooks:
            hook.set_params(tau, layer_specific_freqs[layer_idx])
        
        total_loss = 0
        total_tokens = 0
        total_entropy = 0
        
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc=f"评估 Tau={tau:.2f}"):
                input_ids = torch.stack(batch["input_ids"]).to(DEVICE)
                labels = input_ids.clone()
                
                outputs = model(input_ids, labels=labels, output_router_logits=True)
                
                # 计算 Perplexity
                loss = outputs.loss
                total_loss += loss.item() * (input_ids.size(0) * input_ids.size(1))
                total_tokens += input_ids.size(0) * input_ids.size(1)

                # 计算 Router Entropy
                # outputs.router_logits 是一个元组，每个元素对应一层MoE的输出
                # 每个元素又是 (router_logits, expert_indices) 的元组
                if outputs.router_logits:
                    all_router_logits = torch.cat([l[0] for l in outputs.router_logits], dim=0)
                    probs = F.softmax(all_router_logits, dim=-1)
                    log_probs = F.log_softmax(all_router_logits, dim=-1)
                    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
                    total_entropy += entropy.item()

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        avg_entropy = total_entropy / len(validation_loader) if len(validation_loader) > 0 else 0
        
        print(f"Tau: {tau:.2f} -> Perplexity: {perplexity:.4f}, Router Entropy: {avg_entropy:.4f}")
        results.append({"tau": tau, "perplexity": perplexity, "router_entropy": avg_entropy})

    # --- 步骤 6: 选定最佳 Tau 并保存结果 ---
    for handle in handles:
        handle.remove() # 清理所有 hooks
    print("\n✅ 所有 Hook 已被移除。")

    best_result = min(results, key=lambda x: x["perplexity"])
    print("\n🎉 校准完成！")
    print(f"最佳 Tau: {best_result['tau']:.2f}")
    print(f"最低 Perplexity: {best_result['perplexity']:.4f}")
    print(f"对应的 Router Entropy: {best_result['router_entropy']:.4f}")

    # 保存详细结果
    output_file = os.path.join(OUTPUT_DIR, "calibration_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"详细结果已保存至: {output_file}")
    
    print("\n下一步建议:")
    print(f"使用选定的 tau = {best_result['tau']:.2f} 参数，在你的 `evaluate_benchmark.sh` 中进行一次完整的评估。")


if __name__ == "__main__":
    main()