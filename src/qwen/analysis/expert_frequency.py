import torch
from collections import Counter
from .hook_utils import MultiGPUHook

def get_expert_frequency(model, dataloader, num_top_k=2):
    """
    Calculates the activation frequency of each expert in a Qwen2Moe model.

    Args:
        model: The loaded Qwen2Moe model.
        dataloader: Dataloader providing calibration data.
        num_top_k (int): The number of experts to consider per token (e.g., 2 for Qwen-MoE).

    Returns:
        A dictionary where keys are layer names and values are Counter objects
        with expert indices and their activation counts.
    """
    # Qwen2-MoE's router is part of the Qwen2MoeBlock
    hook = MultiGPUHook(model, "Qwen2MoeBlock")
    hook.register()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.to(next(model.parameters()).device)
            # output_router_logits=True is crucial
            model(input_ids, output_router_logits=True)

    router_logits_by_layer = hook.get_outputs()
    hook.remove()

    frequencies = {}
    for layer_name, all_logits in router_logits_by_layer.items():
        # all_logits shape: [num_tokens, num_experts]
        # Get the indices of the top-k experts for each token
        top_k_indices = torch.topk(all_logits, k=num_top_k, dim=-1).indices
        
        # Flatten the indices and count occurrences
        expert_counts = Counter(top_k_indices.flatten().tolist())
        frequencies[layer_name] = expert_counts
        print(f"Frequencies for {layer_name}: {expert_counts}")

    return frequencies