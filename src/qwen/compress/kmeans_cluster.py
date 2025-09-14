import torch
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any, Optional
import json
import os

def load_similarity_matrix(file_path: str) -> torch.Tensor:
    """加载相似度矩阵文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Similarity matrix file not found: {file_path}")
    
    similarity_matrix: torch.Tensor = torch.load(file_path, map_location='cpu')
    print(f"Loaded similarity matrix with shape: {similarity_matrix.shape}")
    return similarity_matrix

def cluster_experts_kmeans(
    similarity_matrix: torch.Tensor, 
    n_clusters: int, 
    random_state: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    使用K-means++对专家进行聚类
    
    Args:
        similarity_matrix: 专家相似度矩阵 (n_experts, n_experts)
        n_clusters: 目标聚类数量
        random_state: 随机种子
        
    Returns:
        cluster_labels: 聚类标签 (n_experts,)
        cluster_info: 聚类信息字典
    """
    # 将相似度矩阵转换为距离矩阵 (1 - similarity)
    distance_matrix: npt.NDArray[np.float32] = 1.0 - similarity_matrix.numpy()
    
    # 使用K-means++进行聚类
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=random_state,
        n_init=10
    )
    
    # 直接对距离矩阵进行聚类
    cluster_labels: npt.NDArray[np.int32] = kmeans.fit_predict(distance_matrix)
    
    # 收集聚类信息
    cluster_info: dict[str, Any] = {
        'n_clusters': n_clusters,
        'n_experts': len(cluster_labels),
        'inertia': kmeans.inertia_,
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'cluster_sizes': [],
        'expert_assignments': {}
    }
    
    # 强制标注
    cluster_sizes: List[int] = cluster_info['cluster_sizes']
    expert_assignments: Dict[str, List[int]] = cluster_info['expert_assignments']

    # 统计每个聚类的大小和专家分配
    for cluster_id in range(n_clusters):
        experts_in_cluster = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_sizes.append(len(experts_in_cluster))
        expert_assignments[f'cluster_{cluster_id}'] = experts_in_cluster
    
    return cluster_labels, cluster_info

def save_clustering_results(
    cluster_labels: np.ndarray,
    cluster_info: Dict[str, Any],
    layer_idx: int,
    output_dir: str
) -> None:
    """保存聚类结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存聚类标签
    labels_path = os.path.join(output_dir, f"cluster_labels_layer_{layer_idx}.pt")
    torch.save(torch.from_numpy(cluster_labels), labels_path)
    
    # 保存聚类信息
    info_path = os.path.join(output_dir, f"cluster_info_layer_{layer_idx}.json")
    with open(info_path, 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    print(f"Clustering results saved:")
    print(f"  - Labels: {labels_path}")
    print(f"  - Info: {info_path}")

def cluster_layer_experts(
    result_dir: str,
    layer_idx: int,
    n_clusters: int,
    output_dir: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    对指定层的专家进行聚类
    
    Args:
        result_dir: 分析结果目录
        layer_idx: 层索引
        n_clusters: 目标聚类数量
        output_dir: 输出目录，默认为result_dir
        
    Returns:
        cluster_labels: 聚类标签
        cluster_info: 聚类信息
    """
    if output_dir is None:
        output_dir = result_dir
    
    # 构建相似度矩阵文件路径
    similarity_file = os.path.join(result_dir, f"avg_similarity_matrix_layer_{layer_idx}.pt")
    
    # 加载相似度矩阵
    similarity_matrix = load_similarity_matrix(similarity_file)
    
    # 执行聚类
    print(f"\nClustering experts for layer {layer_idx}...")
    print(f"Number of experts: {similarity_matrix.shape[0]}")
    print(f"Target clusters: {n_clusters}")
    
    cluster_labels, cluster_info = cluster_experts_kmeans(similarity_matrix, n_clusters)
    
    # 显示聚类结果
    print(f"\nClustering completed!")
    print(f"Cluster sizes: {cluster_info['cluster_sizes']}")
    for cluster_id, experts in cluster_info['expert_assignments'].items():
        print(f"  {cluster_id}: {len(experts)} experts")
    
    # 保存结果
    save_clustering_results(cluster_labels, cluster_info, layer_idx, output_dir)
    
    return cluster_labels, cluster_info

def main():
    """示例使用"""
    # 配置参数
    RESULT_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/analysis_results/similarity_results"
    LAYER_IDX = list(range(24))  # 要聚类的层
    N_CLUSTERS = 8  # 目标专家数量
    OUTPUT_DIR = f"/root/fsas/zhanghongyu/SMoE/qwen/analysis_results/kmeans_clusters_{N_CLUSTERS}"
    for layer in LAYER_IDX:
        try:
            # 执行聚类
            cluster_labels, cluster_info = cluster_layer_experts(
                result_dir=RESULT_DIR,
                layer_idx=layer,
                n_clusters=N_CLUSTERS,
                output_dir=OUTPUT_DIR
            )
            
            print(f"\n✅ Successfully clustered layer {LAYER_IDX} experts into {N_CLUSTERS} clusters")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()