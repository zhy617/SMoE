import torch
import torch.nn.functional as F
from transformers import Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer, Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
from typing import List, Tuple, Dict, cast, Optional, Union
import os

from ...config import (
    CURRENT_CLUSTER_N
)


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

            layer_device = next(layer.parameters()).device
            hidden_states = hidden_states.to(layer_device)
            # position_embeddings = position_embeddings.to(layer_device)

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



def get_expert_activation_from_saved_states(
    model: Qwen2MoeForCausalLM, # 仍然需要模型来访问专家权重
    saved_states_dir: str,
    target_moe_layer_idx: int
) -> torch.Tensor:
    """
    input:
        model: 已加载的 Qwen2MoeForCausalLM 模型。Qwen1.5-MoE-A2.7B
        saved_states_dir: 之前保存 hidden_states 的目录路径。
        target_moe_layer_idx: 目标 MoE 层的索引。
    output:
        expert_activations: 形状为 (num_experts, seq_len, hidden_size) 的张量，包含每个专家对输入序列的激活输出。
    读取之前保存的 hidden_states，手动执行 LayerNorm 和专家前向传播，返回专家激活。
    单个样本的分析。
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
    if isinstance(moe_block, Qwen2MoeMLP):
        raise ValueError(f"Layer {target_moe_layer_idx} is not a MoE layer.")

    num_experts = CURRENT_CLUSTER_N
    expert_outputs: List[torch.Tensor] = []

    with torch.no_grad():
        for i in range(num_experts):
            expert = moe_block.experts[i]
            output = expert(pre_moe_hidden_states)
            expert_outputs.append(output)

    stacked_expert_outputs = torch.stack(expert_outputs)
    return stacked_expert_outputs


def calculate_expert_similarity_matrix(expert_activations: torch.Tensor) -> torch.Tensor:
    """
    根据论文公式计算专家间的相似度矩阵：
    Sim(Ei, Ej) = (1/m) * Σ(Ei(xl) · Ej(xl) / (||Ei(xl)|| * ||Ej(xl)||))
    
    Args:
        expert_activations: 形状为 (num_experts, batch_size, seq_len, hidden_dim) 的专家激活张量
        
    Returns:
        similarity_matrix: 形状为 (num_experts, num_experts) 的相似度矩阵
    """
    num_experts = CURRENT_CLUSTER_N
    batch_size, seq_len, hidden_dim = expert_activations.shape[1], expert_activations.shape[2], expert_activations.shape[3]
    
    # 将专家激活重塑为 (num_experts, m, hidden_dim)，其中 m = batch_size * seq_len
    # 这样每个 token 对应公式中的一个 x_l
    expert_activations_reshaped = expert_activations.view(num_experts, batch_size * seq_len, hidden_dim)
    m = expert_activations_reshaped.shape[1]  # 总 token 数
    
    print(f"  - Computing similarity for {num_experts} experts across {m} tokens")
    
    # 初始化相似度矩阵
    similarity_matrix = torch.zeros(num_experts, num_experts, device=expert_activations.device)
    
    # 计算每一对专家的相似度
    for i in range(num_experts):
        for j in range(num_experts):
            if i == j:
                # 专家与自身的相似度为1
                similarity_matrix[i, j] = 1.0
            else:
                # Ei(xl) 和 Ej(xl) 分别是专家i和j对所有token的输出
                expert_i_outputs = expert_activations_reshaped[i]  # (m, hidden_dim)
                expert_j_outputs = expert_activations_reshaped[j]  # (m, hidden_dim)
                
                # 计算每个token的点积：Ei(xl) · Ej(xl)
                dot_products = torch.sum(expert_i_outputs * expert_j_outputs, dim=1)  # (m,)
                
                # 计算每个token的L2范数：||Ei(xl)||, ||Ej(xl)||
                norms_i = torch.norm(expert_i_outputs, p=2, dim=1)  # (m,)
                norms_j = torch.norm(expert_j_outputs, p=2, dim=1)  # (m,)
                
                # 避免除零，添加小的epsilon
                epsilon = 1e-8
                cosine_similarities = dot_products / (norms_i * norms_j + epsilon)  # (m,)
                
                # 计算平均余弦相似度：(1/m) * Σ cosine_similarity
                avg_cosine_similarity = torch.mean(cosine_similarities)
                similarity_matrix[i, j] = avg_cosine_similarity
    
    print("\n--- Expert Similarity Matrix (Average Cosine Similarity) ---")
    torch.set_printoptions(precision=4, sci_mode=False)
    # print(similarity_matrix)
    torch.set_printoptions(profile="default")
    
    return similarity_matrix


def calculate_expert_activation_frequency(
    saved_states_dir: str,
    target_moe_layer_idx: int,
    top_k: int = 4
) -> Optional[torch.Tensor]:
    """
    从保存的router logits计算专家激活频率。
    
    Args:
        saved_states_dir: 保存hidden states和router logits的目录
        target_moe_layer_idx: 目标MoE层的索引
        top_k: 每个token激活的专家数量 (对于Qwen1.5-MoE通常是4)
        
    Returns:
        activation_counts: 形状为(num_experts,)的张量，记录每个专家被激活的次数
    """
    print(f"\n--- Calculating expert activation frequency for layer {target_moe_layer_idx} ---")
    
    # 加载router logits
    router_logits_path = os.path.join(saved_states_dir, f"router_logits_layer_{target_moe_layer_idx}.pt")
    
    if not os.path.exists(router_logits_path):
        print(f"Warning: Router logits file not found at {router_logits_path}")
        return None
        
    router_logits: torch.Tensor = torch.load(router_logits_path)  # 形状: (batch_size, seq_len, num_experts)
    
    print(f"  - Router logits shape: {router_logits.shape}")
    
    # 获取top-k专家的索引
    top_k_indices = torch.topk(router_logits, k=top_k, dim=-1).indices  # (batch_size, seq_len, top_k)
    
    # 展平为一维，统计每个专家被选中的次数
    flat_indices = top_k_indices.flatten()  # (batch_size * seq_len * top_k,)
    num_experts = CURRENT_CLUSTER_N
    
    # 统计激活频率
    activation_counts = torch.bincount(flat_indices, minlength=num_experts)
    
    print(f"  - Total tokens: {router_logits.shape[0] * router_logits.shape[1]}")
    print(f"  - Expert activation counts shape: {activation_counts.shape}")
    print(f"  - Top 10 most activated experts: {torch.topk(activation_counts, k=10)}")
    
    return activation_counts