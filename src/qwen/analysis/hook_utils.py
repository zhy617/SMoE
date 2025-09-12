import torch
from collections import defaultdict
from transformers import Qwen2ForCausalLM

class MultiGPUHook:
    """
    A helper class to manage forward hooks on a model distributed across multiple GPUs.
    It automatically gathers the hooked outputs to the CPU.
    """
    def __init__(self, model: Qwen2ForCausalLM, module_class_name: str) -> None:
        self.model = model
        self.module_class_name = module_class_name
        self.handles = []
        self.outputs = defaultdict(list)
        self._find_modules()

    def _find_modules(self) -> None:
        """Finds all modules of the specified class in the model."""
        self.target_modules = []
        for name, module in self.model.named_modules():
            if self.module_class_name in module.__class__.__name__:
                self.target_modules.append((name, module))
        print(f"Found {len(self.target_modules)} modules of type '{self.module_class_name}'.")

    def _hook_fn(self, name):
        """The hook function factory."""
        def hook(module, input, output):
            # Move data to CPU to aggregate from all devices
            # Detach to avoid holding onto the computation graph
            if isinstance(output, torch.Tensor):
                self.outputs[name].append(output.detach().cpu())
            # For router outputs or other tuple/dict outputs
            elif hasattr(output, "router_logits"):
                 self.outputs[name].append(output.router_logits.detach().cpu())
            else:
                # Handle other complex output structures if necessary
                print(f"Warning: Unsupported output type {type(output)} in hook for {name}")

        return hook

    def register(self):
        """Registers the forward hook on all target modules."""
        self.clear()
        for name, module in self.target_modules:
            handle = module.register_forward_hook(self._hook_fn(name))
            self.handles.append(handle)

    def get_outputs(self, clear_on_get=True):
        """Returns the concatenated outputs from all hooks."""
        # Concatenate tensors from all forward passes and all GPUs
        gathered = {name: torch.cat(tensors, dim=0) for name, tensors in self.outputs.items()}
        if clear_on_get:
            self.clear()
        return gathered

    def clear(self):
        """Clears the stored outputs."""
        self.outputs.clear()

    def remove(self):
        """Removes all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []