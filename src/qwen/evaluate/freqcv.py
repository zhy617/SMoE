import os
from typing import Dict

import torch

from ...config import (
    HIDDEN_STATES_DIR,   # 例如: .../tensor_cache/Qwen/..._wikitext_128/hidden_states_cache
    TARGET_LAYERS,
    EVALUATE_DIR,
)


def compute_coefficient_of_variation(counts: torch.Tensor) -> float:
    """
    计算变异系数（Coefficient of Variation, CV） = std / mean
    counts: 形状为 (num_experts,) 的激活次数向量
    """
    counts = counts.float()
    mean = counts.mean()
    std = counts.std(unbiased=False)  # population std
    if mean == 0:
        return 0.0 # TODO: or float('inf')?
    return (std / mean).item()


def evaluate_activation_cv(
    activation_result_dir: str,
) -> Dict[int, float]:
    """
    从 activation_frequency_results 目录读取每层的激活频率文件，
    计算每层专家激活次数的变异系数。

    activation_result_dir 形如：
    ~/fsas/zhanghongyu/LAMoE/tensor_cache/Qwen/Qwen1.5-MoE-A2.7B_wikitext_128/activation_frequency_results
    其中每个文件名为: activation_frequency_layer_{layer}.pt
    文件内容为一个 dict:
        {
            'activation_counts': Tensor[num_experts],
            'activation_percentages': Tensor[num_experts],
            'total_samples': int,
        }
    """
    layer_cv: Dict[int, float] = {}

    for layer in TARGET_LAYERS:
        freq_file = os.path.join(
            activation_result_dir,
            f"activation_frequency_layer_{layer}.pt",
        )
        if not os.path.exists(freq_file):
            print(f"Warning: activation file not found for layer {layer}: {freq_file}")
            continue

        data = torch.load(freq_file, map_location="cpu")
        # 优先使用 'activation_counts'
        if "activation_counts" not in data:
            print(f"Warning: 'activation_counts' not in file {freq_file}, skip layer {layer}")
            continue

        counts: torch.Tensor = data["activation_counts"]
        cv = compute_coefficient_of_variation(counts)
        layer_cv[layer] = cv

    return layer_cv


if __name__ == "__main__":
    # HIDDEN_STATES_DIR 形如:
    # /root/fsas/zhanghongyu/LAMoE/tensor_cache/Qwen/Qwen1.5-MoE-A2.7B_wikitext_128/hidden_states_cache
    # 把 "hidden_states_cache" 替换为 "activation_frequency_results"
    parent_dir = os.path.dirname(HIDDEN_STATES_DIR)
    activation_result_dir = os.path.join(parent_dir, "activation_frequency_results")

    print(f"Activation frequency dir: {activation_result_dir}")

    layer_cv = evaluate_activation_cv(activation_result_dir)

    cv_mean = sum(layer_cv.values()) / len(layer_cv) if layer_cv else 0.0
    print(f"\n=== Coefficient of Variation (CV) across layers ===")
    print(f"Mean CV across layers: {cv_mean:.6f}\n")

    for layer, cv in sorted(layer_cv.items()):

        print(f"Layer {layer}: Coefficient of Variation (std/mean) = {cv:.6f}")

    # === 保存为 JSON 和 PT 文件 ===
    import json

    output_dir = os.path.join(EVALUATE_DIR, "activation_cv")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "activation_cv.json")
    with open(json_path, "w") as f:
        json.dump(layer_cv, f, indent=2, sort_keys=True)
    print(f"[Saved] activation CV JSON -> {json_path}")

    pt_path = os.path.join(output_dir, "activation_cv.pt")
    torch.save(layer_cv, pt_path)
    print(f"[Saved] activation CV PT -> {pt_path}")