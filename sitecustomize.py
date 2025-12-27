# filepath: /root/LAMoE/sitecustomize.py
import torch
from torch import nn
# from transformers.models.qwen2_moe

try:
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
except Exception:
    Qwen2MoeSparseMoeBlock = None

if Qwen2MoeSparseMoeBlock is not None:
    _orig_init = Qwen2MoeSparseMoeBlock.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        cfg = getattr(self, "config", None)
        need_bias = getattr(cfg, "apply_logit_adjustment", False)
        gate = getattr(self, "gate", None)
        if need_bias and gate is not None and getattr(gate, "bias", None) is None:
            gate.register_parameter(
                "bias",
                nn.Parameter(torch.zeros(gate.out_features, dtype=gate.weight.dtype, device=gate.weight.device)),
            )

    Qwen2MoeSparseMoeBlock.__init__ = _patched_init