import torch
import torch.nn.functional as F
from transformers import Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
from typing import List, Tuple, Dict, cast, Optional, Union
import os

def generate_and_save_hidden_states(
    model: Qwen2MoeForCausalLM,
    input_ids: torch.Tensor,
    save_dir: str
) -> None:
    """
    input:
        model: 已加载的 Qwen2MoeForCausalLM 模型。Qwen1.5-MoE-A2.7B
        input_ids: 输入的 token IDs，形状为 (1, seq_len)。
        save_dir: 保存 hidden_states 的目录路径。
    执行一次前向传播，并将每一层的输出 hidden_states 保存到磁盘。
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存输入
    torch.save(input_ids.cpu(), os.path.join(save_dir, "input_ids.pt"))

    # --- 新增：手动创建 position_ids ---
    batch_size, seq_length = input_ids.shape
    device = input_ids.device
    # position_ids 应该在模型所在的设备上
    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=model.device).unsqueeze(0)

    with torch.no_grad():
        # 获取词嵌入
        hidden_states: torch.Tensor = model.model.embed_tokens(input_ids.to(model.device))
        # torch.save(hidden_states.cpu(), os.path.join(save_dir, "hidden_states_embedding.pt"))

        position_embeddings: tuple[torch.Tensor, torch.Tensor] = model.model.rotary_emb(hidden_states, position_ids)  

        # 逐层执行并保存
        for i, layer_module in enumerate(model.model.layers):
            layer = cast(Qwen2MoeDecoderLayer, layer_module)
            print(f"  - Processing layer {i}...")

            # 1. Attention Block
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            # --- 修改：在调用 self_attn 时传入 position_ids ---
            hidden_states, _ = layer.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False
            )
            hidden_states = residual + hidden_states

            # --- 在这里保存中间结果！---
            # Fully Connected
            # 这个 hidden_states 是 Attention 之后、进入 post_attention_layernorm 之前的状态
            torch.save(hidden_states.cpu(), os.path.join(save_dir, f"hidden_states_after_attn_layer_{i}.pt"))

            # 2. MLP/MoE Block (为了给下一层提供正确的输入，我们必须完成本层的计算)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            mlp_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]= layer.mlp(hidden_states)
            if isinstance(mlp_output, tuple):
                hidden_states, router_logits = mlp_output  
                torch.save(router_logits.cpu(), os.path.join(save_dir, f"router_logits_layer_{i}.pt"))
            else:
                router_logits = None
            
            hidden_states = residual + hidden_states
    
    print(f"\nAll intermediate hidden states saved to '{save_dir}'.")



def analyze_similarity_from_saved_states(
    model: Qwen2MoeForCausalLM, # 仍然需要模型来访问专家权重
    saved_states_dir: str,
    target_moe_layer_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从磁盘加载预存的 hidden_states，并执行专家相似度分析。
    """
    print(f"\n--- Analyzing from saved states for layer {target_moe_layer_idx} ---")
    
    # --- 1. 加载目标层之前的 hidden_states ---
    # 这是进入 post_attention_layernorm 的正确输入
    state_path = os.path.join(saved_states_dir, f"hidden_states_after_attn_layer_{target_moe_layer_idx}.pt")
    hidden_states: torch.Tensor = torch.load(state_path).to(model.device)

    # --- 2. 手动执行 LayerNorm ---
    target_layer = cast(Qwen2MoeDecoderLayer, model.model.layers[target_moe_layer_idx])
    pre_moe_hidden_states: torch.Tensor = target_layer.post_attention_layernorm(hidden_states)
    print(f"  - Shape of loaded and normed hidden states: {pre_moe_hidden_states.shape}")

    # --- 3. 手动遍历专家 (与之前相同) ---
    moe_block = target_layer.mlp
    if "Moe" not in moe_block.__class__.__name__:
        raise TypeError(f"Layer {target_moe_layer_idx} is not a MoE layer.")

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