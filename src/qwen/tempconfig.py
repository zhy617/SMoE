import os
from datetime import datetime

class Config:
    def __init__(self, 
                 root_dir="/root/fsas/zhanghongyu/LAMoE/qwen/",
                 base_model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat",
                 data_name="wikitext",
                 sample_size=128,
                 base_model_path="/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B-Chat",
                 experiment_name="default_exp"):
        
        # --- 基础配置 ---
        self.root_dir = root_dir
        self.base_model_name = base_model_name.split("/")[-1] # 提取短名
        self.data_name = data_name
        self.sample_size = sample_size
        
        # 自动生成实验时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"

        # --- 1. 静态资源路径 ---
        self.data_dir = os.path.join(root_dir, "data")
        self.model_dir = os.path.join(root_dir, "base_models")
        self.input_file = os.path.join(self.data_dir, f"{data_name}_calibration.json")

        # --- 2. 共享缓存路径 (Shared Cache) ---
        # 逻辑：如果是同一个模型、同一个数据、同样的样本数，中间态可以复用
        cache_folder_name = f"{self.base_model_name}_{self.data_name}_{self.sample_size}"
        self.cache_dir = os.path.join(root_dir, "shared_cache", cache_folder_name)
        self.hidden_states_dir = os.path.join(self.cache_dir, "hidden_states")
        
        # --- 3. 实验工作区 (Workspace) ---
        self.exp_dir = os.path.join(root_dir, "experiments", self.experiment_id)
        self.analysis_dir = os.path.join(self.exp_dir, "analysis")
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        self.eval_dir = os.path.join(self.exp_dir, "evaluation")

        # 自动创建目录
        self._create_dirs()

    def _create_dirs(self):
        """初始化时自动创建主要文件夹"""
        for p in [self.cache_dir, self.hidden_states_dir, 
                  self.analysis_dir, self.ckpt_dir, self.eval_dir]:
            os.makedirs(p, exist_ok=True)

    # --- 动态路径生成器 ---
    
    def get_ckpt_path(self, expert_method, router_method, k_clusters):
        """根据合并策略生成唯一的模型保存路径"""
        dir_name = f"expert_{expert_method}_router_{router_method}_k{k_clusters}"
        return os.path.join(self.ckpt_dir, dir_name)

    def get_eval_path(self, expert_method, router_method, k_clusters):
        """根据合并策略生成唯一的评估结果路径"""
        dir_name = f"expert_{expert_method}_router_{router_method}_k{k_clusters}"
        path = os.path.join(self.eval_dir, dir_name)
        os.makedirs(path, exist_ok=True)
        return path

    def get_cluster_path(self, k_clusters):
        """获取聚类结果路径"""
        return os.path.join(self.analysis_dir, f"kmeans_clusters_k{k_clusters}")

# --- 使用示例 ---

# 初始化配置
cfg = Config(
    root_dir="/root/fsas/zhanghongyu/LAMoE/qwen/",
    base_model_path="/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B-Chat",
    experiment_name="test_entropy_idea",
    sample_size=128
)

print(f"中间状态缓存 (可复用): {cfg.hidden_states_dir}")
print(f"本次实验根目录: {cfg.exp_dir}")

# 假设我们在流水线中：
expert_method = "svd"
router_method = "avg"
k = 30

# 1. 获取模型保存路径
save_path = cfg.get_ckpt_path(expert_method, router_method, k)
print(f"模型将保存在: {save_path}")

# 2. 获取评估结果保存路径 (存放 entropy, load balance 等)
eval_output_path = cfg.get_eval_path(expert_method, router_method, k)
print(f"评估结果将保存在: {eval_output_path}")

# 3. 生成具体文件路径
entropy_file = os.path.join(eval_output_path, "router_entropy.json")