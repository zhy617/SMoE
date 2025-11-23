import torch
import torch.nn.functional as F
from transformers import Qwen2MoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer, Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
from typing import List, Tuple, Dict, cast, Optional, Union
import os



