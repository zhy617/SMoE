import os

CLUSTER_N = 30  # 聚类目标数量
SAMPLE_SIZE = 128
MAX_LENGTH = 2048

EXPERT_MERGING_METHOD = "svd"
LOGIT_MERGING_METHOD = "avg"  # "avg" 或 "max" 或 "none"
ROUTER_MERGING_METHOD = "avg"  # "avg" 或 "svd"

# --- 模型配置 ---
BASE_MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
BASE_MODEL_PATH = os.path.join("/root/fsas/models", BASE_MODEL_NAME)

# workspace 路径
WORKSPACE_DIR = "/root/SMoE" # 可自定义

SAMPLE_INPUT_FILE = os.path.join(WORKSPACE_DIR, "data/qwen/wikitext_calibration.json")

# --- 中间结果保存路径 ---
BASE_INTER_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/" # 可自定义

# --- 分析结果保存路径 ---
# 包括 activation_frequency_results, kmeans_clusters_XX 
# redundancy_results  similarity_results 
ANALYSIS_DIR = os.path.join(BASE_INTER_DIR, "analysis_results")

# --- 具体中间结果路径 ---
# 包括 router_logits_layer_0.pt, hidden_states_after_attn_layer_0.pt 等文件
HIDDEN_STATES_DIR = os.path.join(ANALYSIS_DIR, "hidden_states_cache") 


CLUSTER_DIR = os.path.join(ANALYSIS_DIR, f"kmeans_clusters_{CLUSTER_N}")
FREQ_RESULT_DIR = os.path.join(ANALYSIS_DIR, "activation_frequency_results")
OUTPUT_MODEL_DIR = os.path.join(BASE_INTER_DIR, "merged_models")
SIMILARITY_RESULT_DIR = os.path.join(ANALYSIS_DIR, "similarity_results")


# --- 其他配置 ---
# Qwen1.5-MoE-A2.7B 共有 24 个 transformer 层
TARGET_LAYERS = list(range(24))


