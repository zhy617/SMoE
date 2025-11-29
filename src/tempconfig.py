import os

# =========================================================
# 1. 核心超参数 (手动修改这里)
# =========================================================
CLUSTER_N = 30          # 聚类数
SAMPLE_SIZE = 128       # 采样样本数
MAX_LENGTH = 2048

# 合并策略配置
EXPERT_MERGING_METHOD = "svd"
ROUTER_MERGING_METHOD = "avg"   # "avg", "svd"
LOGIT_MERGING_METHOD = "avg"    # "avg", "max", "none"

# 基础模型信息
FAMILY_NAME = "Qwen"    # [关键] 家族名称 (Qwen, Mixtral)
BASE_MODEL_FULL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
BASE_MODEL_SHORT_NAME = "Qwen1.5-MoE-A2.7B-Chat"

# 数据集
DATASET_NAME = "wikitext"

# =========================================================
# 2. 项目根目录结构
# =========================================================
# 项目根路径
ROOT_DIR = "/root/fsas/zhanghongyu/LAMoE"

# 原始数据文件
SAMPLE_INPUT_FILE = os.path.join(ROOT_DIR, "data", f"{DATASET_NAME}_calibration.json")

# 三大核心目录 (基于 Family 隔离)
MODELS_ROOT = os.path.join(ROOT_DIR, "models", FAMILY_NAME)
CACHE_ROOT = os.path.join(ROOT_DIR, "tensor_cache", FAMILY_NAME)
EVAL_ROOT = os.path.join(ROOT_DIR, "evaluation", FAMILY_NAME)

# =========================================================
# 3. 模型路径 (Models)
# =========================================================
# [A] Base Model 路径 (读取用)
# 注意：你需要确保 models/Qwen/ 下有这个文件夹（或者是软链接）
BASE_MODEL_PATH = os.path.join(MODELS_ROOT, BASE_MODEL_SHORT_NAME)

# [B] Merged Model 路径 (写入用)
# 自动生成文件夹名：expert_svd_router_avg_k30
MERGED_TAG = f"expert_{EXPERT_MERGING_METHOD}_router_{ROUTER_MERGING_METHOD}_k{CLUSTER_N}"
MERGED_MODEL_DIR = os.path.join(MODELS_ROOT, MERGED_TAG)

# =========================================================
# 4. 中间缓存路径 (Tensor Cache)
# =========================================================
# 这里我们需要定义两个路径：一个是 Base 的(读聚类中心)，一个是 Merged 的(存Logits)

# [A] Base Model 的缓存 (存放聚类中心、原始 Hidden States)
# 文件夹名: Qwen1.5-MoE-A2.7B-Chat_wikitext_128
BASE_CACHE_DIR = os.path.join(CACHE_ROOT, f"{BASE_MODEL_SHORT_NAME}_{DATASET_NAME}_{SAMPLE_SIZE}")

# 具体文件路径 (方便直接调用)
BASE_HIDDEN_STATES_FILE = os.path.join(BASE_CACHE_DIR, "hidden_states.pt")
BASE_ROUTER_LOGITS_FILE = os.path.join(BASE_CACHE_DIR, "router_logits.pt")

# [关键] 聚类中心 (始终在 Base 的缓存里)
CLUSTERING_CENTER_FILE = os.path.join(BASE_CACHE_DIR, "clustering_centers", f"kmeans_k{CLUSTER_N}.pt")


# [B] Merged Model 的缓存 (存放合并后的 Logits，用于分析信息熵)
# 文件夹名: expert_svd_router_avg_k30_wikitext_128
MERGED_CACHE_DIR = os.path.join(CACHE_ROOT, f"{MERGED_TAG}_{DATASET_NAME}_{SAMPLE_SIZE}")

# 具体文件路径
MERGED_HIDDEN_STATES_FILE = os.path.join(MERGED_CACHE_DIR, "hidden_states.pt")
MERGED_ROUTER_LOGITS_FILE = os.path.join(MERGED_CACHE_DIR, "router_logits.pt")

# =========================================================
# 5. 分析结果路径 (Evaluation)
# =========================================================
# 存放 entropy.json, load_balance.png 等
ANALYSIS_RESULT_DIR = os.path.join(EVAL_ROOT, MERGED_TAG)


# =========================================================
# 6. 其他配置
# =========================================================
TARGET_LAYERS = list(range(24))

# 简单检查目录是否存在的辅助代码 (可选)
def ensure_dirs():
    """在脚本开头调用此函数，确保所有父文件夹都存在"""
    for p in [MERGED_MODEL_DIR, BASE_CACHE_DIR, MERGED_CACHE_DIR, ANALYSIS_RESULT_DIR]:
        os.makedirs(p, exist_ok=True)
    # 聚类中心的子文件夹单独建一下
    os.makedirs(os.path.dirname(CLUSTERING_CENTER_FILE), exist_ok=True)