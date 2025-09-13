import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from typing import List, Tuple, Dict
import os

def generate_and_save_hidden_states(
    model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    save_dir: str
) -> None:
    """
    执行一次前向传播，并将每一层的输出 hidden_states 保存到磁盘。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存输入
    torch.save(input_ids.cpu(), os.path.join(save_dir, "input_ids.pt"))

    with torch.no_grad():
        # 获取词嵌入
        hidden_states = model.model.embed_tokens(input_ids.to(model.device))
        torch.save(hidden_states.cpu(), os.path.join(save_dir, "hidden_states_embedding.pt"))

        # 逐层执行并保存
        for i, layer in enumerate(model.model.layers):
            print(f"  - Processing and saving output of layer {i}...")
            layer_outputs = layer(hidden_states, use_cache=False)
            hidden_states = layer_outputs[0]
            torch.save(hidden_states.cpu(), os.path.join(save_dir, f"hidden_states_layer_{i}.pt"))
    
    print(f"\nAll intermediate hidden states saved to '{save_dir}'.")


def analyze_similarity_from_saved_states(
    model: Qwen2ForCausalLM, # 仍然需要模型来访问专家权重
    saved_states_dir: str,
    target_moe_layer_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从磁盘加载预存的 hidden_states，并执行专家相似度分析。
    """
    print(f"\n--- Analyzing from saved states for layer {target_moe_layer_idx} ---")
    
    # --- 1. 加载目标层之前的 hidden_states ---
    # 我们需要进入MoE块的输入，它是在前一层输出的基础上经过一个LayerNorm得到的
    # 因此，我们加载 target_moe_layer_idx - 1 层的输出
    if target_moe_layer_idx > 0:
        pre_layer_path = os.path.join(saved_states_dir, f"hidden_states_layer_{target_moe_layer_idx - 1}.pt")
        hidden_states = torch.load(pre_layer_path).to(model.device)
    else: # 如果目标是第0层
        embedding_path = os.path.join(saved_states_dir, "hidden_states_embedding.pt")
        hidden_states = torch.load(embedding_path).to(model.device)

    # --- 2. 手动执行 LayerNorm ---
    target_layer = model.model.layers[target_moe_layer_idx]
    pre_moe_hidden_states = target_layer.post_attention_layernorm(hidden_states)
    print(f"  - Shape of loaded and normed hidden states: {pre_moe_hidden_states.shape}")

    # --- 3. 手动遍历专家 (与之前相同) ---
    moe_block = target_layer.mlp
    num_experts = len(moe_block.experts)
    expert_outputs: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(num_experts):
            expert = moe_block.experts[i]
            output = expert(pre_moe_hidden_states)
            expert_outputs.append(output)

    stacked_expert_outputs = torch.stack(expert_outputs)

    # --- 4. 计算相似度 (与之前相同) ---
    expert_representations = stacked_expert_outputs.mean(dim=[1, 2])
    normalized_representations = F.normalize(expert_representations, p=2, dim=1)
    similarity_matrix = torch.matmul(normalized_representations, normalized_representations.T)

    print("\n--- Final Result: Expert Similarity Matrix ---")
    torch.set_printoptions(precision=4, sci_mode=False)
    print(similarity_matrix)
    torch.set_printoptions(profile="default")

    return similarity_matrix, stacked_expert_outputs
```

### 更新后的主脚本 `run_qwen_analysis.py`

```python
# filepath: scripts/analysis/run_qwen_analysis.py
# ... imports ...
from src.qwen.analysis.direct_expert_similarity import (
    generate_and_save_hidden_states,
    analyze_similarity_from_saved_states
)

# ... config ...
HIDDEN_STATES_SAVE_DIR = "/root/SMoE/data/hidden_states_cache/sample_0"

def main():
    # ... 加载模型和数据 ...
    # sample_input_ids = ...

    # --- 步骤 1: 生成并保存中间结果 (如果尚未保存) ---
    # 这一步只需要运行一次
    if not os.path.exists(HIDDEN_STATES_SAVE_DIR) or not os.listdir(HIDDEN_STATES_SAVE_DIR):
        print("="*20 + " Generating and saving hidden states... " + "="*20)
        generate_and_save_hidden_states(model, sample_input_ids, HIDDEN_STATES_SAVE_DIR)
    else:
        print(f"Hidden states already found in '{HIDDEN_STATES_SAVE_DIR}'. Skipping generation.")

    # --- 步骤 2: 从保存的结果进行分析 ---
    # 这一步可以反复运行，分析不同的层，速度很快
    print("\n" + "="*20 + " Analyzing Layer 1 from saved states... " + "="*20)
    analyze_similarity_from_saved_states(model, HIDDEN_STATES_SAVE_DIR, target_moe_layer_idx=1)
    
    print("\n" + "="*20 + " Analyzing Layer 3 from saved states... " + "="*20)
    analyze_similarity_from_saved_states(model, HIDDEN_STATES_SAVE_DIR, target_moe_layer_idx=3)
    
    # 你甚至可以在这里卸载模型，然后只用CPU进行其他分析（如果不需要访问专家权重）

if __name__ == "__main__":
    main()
```

这个重构后的工作流更加强大和高效，完全体现了你提出的优化思路。# filepath: /root/SMoE/src/qwen/analysis/direct_expert_similarity.py
import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from typing import List, Tuple, Dict
import os

def generate_and_save_hidden_states(
    model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    save_dir: str
) -> None:
    """
    执行一次前向传播，并将每一层的输出 hidden_states 保存到磁盘。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存输入
    torch.save(input_ids.cpu(), os.path.join(save_dir, "input_ids.pt"))

    with torch.no_grad():
        # 获取词嵌入
        hidden_states = model.model.embed_tokens(input_ids.to(model.device))
        torch.save(hidden_states.cpu(), os.path.join(save_dir, "hidden_states_embedding.pt"))

        # 逐层执行并保存
        for i, layer in enumerate(model.model.layers):
            print(f"  - Processing and saving output of layer {i}...")
            layer_outputs = layer(hidden_states, use_cache=False)
            hidden_states = layer_outputs[0]
            torch.save(hidden_states.cpu(), os.path.join(save_dir, f"hidden_states_layer_{i}.pt"))
    
    print(f"\nAll intermediate hidden states saved to '{save_dir}'.")


def analyze_similarity_from_saved_states(
    model: Qwen2ForCausalLM, # 仍然需要模型来访问专家权重
    saved_states_dir: str,
    target_moe_layer_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从磁盘加载预存的 hidden_states，并执行专家相似度分析。
    """
    print(f"\n--- Analyzing from saved states for layer {target_moe_layer_idx} ---")
    
    # --- 1. 加载目标层之前的 hidden_states ---
    # 我们需要进入MoE块的输入，它是在前一层输出的基础上经过一个LayerNorm得到的
    # 因此，我们加载 target_moe_layer_idx - 1 层的输出
    if target_moe_layer_idx > 0:
        pre_layer_path = os.path.join(saved_states_dir, f"hidden_states_layer_{target_moe_layer_idx - 1}.pt")
        hidden_states = torch.load(pre_layer_path).to(model.device)
    else: # 如果目标是第0层
        embedding_path = os.path.join(saved_states_dir, "hidden_states_embedding.pt")
        hidden_states = torch.load(embedding_path).to(model.device)

    # --- 2. 手动执行 LayerNorm ---
    target_layer = model.model.layers[target_moe_layer_idx]
    pre_moe_hidden_states = target_layer.post_attention_layernorm(hidden_states)
    print(f"  - Shape of loaded and normed hidden states: {pre_moe_hidden_states.shape}")

    # --- 3. 手动遍历专家 (与之前相同) ---
    moe_block = target_layer.mlp
    num_experts = len(moe_block.experts)
    expert_outputs: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(num_experts):
            expert = moe_block.experts[i]
            output = expert(pre_moe_hidden_states)
            expert_outputs.append(output)

    stacked_expert_outputs = torch.stack(expert_outputs)

    # --- 4. 计算相似度 (与之前相同) ---
    expert_representations = stacked_expert_outputs.mean(dim=[1, 2])
    normalized_representations = F.normalize(expert_representations, p=2, dim=1)
    similarity_matrix = torch.matmul(normalized_representations, normalized_representations.T)

    print("\n--- Final Result: Expert Similarity Matrix ---")
    torch.set_printoptions(precision=4, sci_mode=False)
    print(similarity_matrix)
    torch.set_printoptions(profile="default")

    return similarity_matrix, stacked_expert_outputs
```

### 更新后的主脚本 `run_qwen_analysis.py`

```python
# filepath: scripts/analysis/run_qwen_analysis.py
# ... imports ...
from src.qwen.analysis.direct_expert_similarity import (
    generate_and_save_hidden_states,
    analyze_similarity_from_saved_states
)

# ... config ...
HIDDEN_STATES_SAVE_DIR = "/root/SMoE/data/hidden_states_cache/sample_0"

def main():
    # ... 加载模型和数据 ...
    # sample_input_ids = ...

    # --- 步骤 1: 生成并保存中间结果 (如果尚未保存) ---
    # 这一步只需要运行一次
    if not os.path.exists(HIDDEN_STATES_SAVE_DIR) or not os.listdir(HIDDEN_STATES_SAVE_DIR):
        print("="*20 + " Generating and saving hidden states... " + "="*20)
        generate_and_save_hidden_states(model, sample_input_ids, HIDDEN_STATES_SAVE_DIR)
    else:
        print(f"Hidden states already found in '{HIDDEN_STATES_SAVE_DIR}'. Skipping generation.")

    # --- 步骤 2: 从保存的结果进行分析 ---
    # 这一步可以反复运行，分析不同的层，速度很快
    print("\n" + "="*20 + " Analyzing Layer 1 from saved states... " + "="*20)
    analyze_similarity_from_saved_states(model, HIDDEN_STATES_SAVE_DIR, target_moe_layer_idx=1)
    
    print("\n" + "="*20 + " Analyzing Layer 3 from saved states... " + "="*20)
    analyze_similarity_from_saved_states(model, HIDDEN_STATES_SAVE_DIR, target_moe_layer_idx=3)
    
    # 你甚至可以在这里卸载模型，然后只用CPU进行其他分析（如果不需要访问专家权重）

if __name__ == "__main__":
    main()
```

这个重构后的工作流更加强大和高效，完全体现了你提出的优化思路。