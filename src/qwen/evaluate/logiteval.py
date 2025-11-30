import torch
import torch.nn.functional as F
from transformers import Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer, Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
from typing import List, Tuple, Dict, cast, Optional, Union
import os

# 计算logit信息熵
def compute_logit_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

# 逐层读取logit，计算信息熵
# ~/fsas/zhanghongyu/LAMoE/tensor_cache/Qwen/expert_svd_router_avg_k30_wikitext_128/hidden_states_cache/sample_0/router_logits_layer_0.pt
# 返回每层的平均信息熵
def evaluate_model_logit_entropy(
    logits_path: str
) -> Dict[int, float]:
    layer_entropy: Dict[int, float] = {}
    for layer in TARGET_LAYERS:
        for sample_idx in range(SAMPLE_SIZE):
            layer_logit_file = os.path.join(
                logits_path,
                f"sample_{sample_idx}",
                f"router_logits_layer_{layer}.pt"
            )
            layer_logits = torch.load(layer_logit_file)  # shape: (seq_len, vocab_size)
            entropy = compute_logit_entropy(layer_logits)  # shape: (seq_len,)
            avg_entropy = entropy.mean().item()
            if layer not in layer_entropy:
                layer_entropy[layer] = 0.0
            layer_entropy[layer] += avg_entropy / SAMPLE_SIZE
    return layer_entropy


from ...config import (
    HIDDEN_STATES_DIR,
    TARGET_LAYERS,
    SAMPLE_SIZE,
    EVALUATE_DIR,
)

# test tmp logit entropy evaluation
if __name__ == "__main__":
    print(HIDDEN_STATES_DIR)
    output_dir = os.path.join(EVALUATE_DIR, "logit_entropy")
    os.makedirs(output_dir, exist_ok=True)
    layer_entropy = evaluate_model_logit_entropy(HIDDEN_STATES_DIR)

    entropy_mean = sum(layer_entropy.values()) / len(layer_entropy) if layer_entropy else 0.0
    print(f"\n=== Logit Entropy across layers ===")
    print(f"Mean Logit Entropy across layers: {entropy_mean:.4f}\n")

    for layer, entropy in layer_entropy.items():
        print(f"Layer {layer}: Average Logit Entropy = {entropy:.4f}")

    # === 保存为 JSON 和 PT 文件 ===
    # 1) JSON：便于后续脚本 / 可视化使用
    import json
    json_path = os.path.join(output_dir, "logit_entropy.json")
    with open(json_path, "w") as f:
        json.dump(layer_entropy, f, indent=2, sort_keys=True)
    print(f"[Saved] logit entropy JSON -> {json_path}")

    # 2) PT：便于用 torch.load 直接读
    pt_path = os.path.join(output_dir, "logit_entropy.pt")
    torch.save(layer_entropy, pt_path)
    print(f"[Saved] logit entropy PT -> {pt_path}")
    
