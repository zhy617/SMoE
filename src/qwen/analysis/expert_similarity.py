import torch
import torch.nn.functional as F
from .hook_utils import MultiGPUHook

def _force_expert_hook(expert_to_force_idx):
    """A hook factory to create a hook that forces a specific expert."""
    def hook(module, input, output):
        # input is a tuple, input[0] is the hidden state
        # output is the router logits tensor
        # Create a new tensor to avoid in-place modification errors
        new_logits = torch.full_like(output, -1e9)
        # Force all tokens to select the target expert
        new_logits[:, :, expert_to_force_idx] = 0
        return new_logits
    return hook

def get_expert_similarity_matrix(model, dataloader, target_expert_layer="up_proj"):
    """
    Calculates the output similarity matrix between experts by forcing activation.

    Args:
        model: The loaded Qwen2Moe model.
        dataloader: Dataloader providing calibration data.
        target_expert_layer (str): The expert's sub-layer to hook for output capture
                                   (e.g., 'up_proj', 'down_proj').

    Returns:
        A similarity matrix (Tensor).
    """
    moe_blocks = [m for m in model.modules() if "Qwen2MoeBlock" in m.__class__.__name__]
    if not moe_blocks:
        raise ValueError("No Qwen2MoeBlock found in the model.")
    
    # We will analyze the first MoE block for simplicity
    target_moe_block = moe_blocks[0]
    num_experts = len(target_moe_block.experts)
    print(f"Analyzing first MoE block with {num_experts} experts.")

    # Hook to capture the output of a specific layer within each expert MLP
    # The module name will be like 'model.layers.1.mlp.experts.0.up_proj'
    output_hook = MultiGPUHook(model, "Qwen2Mlp")

    expert_outputs = []
    model.eval()

    for i in range(num_experts):
        print(f"Processing expert {i}...")
        
        # Register a temporary hook on the router to force expert i
        router_hook_handle = target_moe_block.router.register_forward_hook(_force_expert_hook(i))
        
        # Register hooks on the expert sub-layers to capture their output
        output_hook.register()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch.to(next(model.parameters()).device)
                model(input_ids)
        
        # Clean up hooks for the next iteration
        router_hook_handle.remove()
        
        # Gather outputs for expert i
        captured_outputs = output_hook.get_outputs()
        
        # We need to find the output of the correct expert's sub-layer
        # The name will contain `experts.{i}.{target_expert_layer}`
        key_fragment = f"experts.{i}.{target_expert_layer}"
        
        expert_i_output = None
        for name, output_tensor in captured_outputs.items():
            if key_fragment in name:
                expert_i_output = output_tensor
                break
        
        if expert_i_output is not None:
            # Average over the token dimension to get a single vector per expert
            expert_outputs.append(expert_i_output.mean(dim=0))
        else:
            print(f"Warning: Could not find output for expert {i}")

    output_hook.remove()

    if not expert_outputs:
        print("No expert outputs were captured.")
        return None

    # Stack all expert representations into a single tensor
    expert_representations = torch.stack(expert_outputs) # [num_experts, hidden_dim]
    
    # Normalize and compute cosine similarity matrix
    expert_representations = F.normalize(expert_representations, p=2, dim=1)
    similarity_matrix = torch.matmul(expert_representations, expert_representations.T)
    
    print("Expert Similarity Matrix:")
    print(similarity_matrix)
    return similarity_matrix