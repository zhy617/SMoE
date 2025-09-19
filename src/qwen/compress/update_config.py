import json
import os
import glob
import argparse
from typing import Dict, List

def scan_cluster_results(cluster_dir: str) -> Dict[int, int]:
    """
    扫描聚类结果目录，从每个JSON文件中提取每层的专家数量。

    Args:
        cluster_dir: 包含 `cluster_info_layer_*.json` 文件的目录。

    Returns:
        一个字典，将层索引映射到该层的专家数量。
    """
    print(f"🔍 正在扫描目录: {cluster_dir}")
    layer_expert_counts: Dict[int, int] = {}
    
    # 使用glob查找所有符合模式的JSON文件
    json_files = glob.glob(os.path.join(cluster_dir, "cluster_info_layer_*.json"))

    if not json_files:
        raise FileNotFoundError(f"在 '{cluster_dir}' 中没有找到 'cluster_info_layer_*.json' 文件。")

    print(f"   找到了 {len(json_files)} 个聚类结果文件。")

    for file_path in json_files:
        try:
            # 从文件名中提取层索引
            # 例如: '.../cluster_info_layer_0.json' -> 0
            filename = os.path.basename(file_path)
            layer_idx_str = filename.split('_')[-1].split('.')[0]
            layer_idx = int(layer_idx_str)

            # 读取JSON文件并提取n_clusters
            with open(file_path, 'r') as f:
                data = json.load(f)
                n_clusters = data['n_clusters']
                layer_expert_counts[layer_idx] = n_clusters
                print(f"   - 第 {layer_idx:2d} 层: {n_clusters:2d} 个专家")

        except (ValueError, KeyError, IndexError) as e:
            print(f"⚠️  处理文件 {file_path} 时出错: {e}。已跳过。")
            continue
            
    if not layer_expert_counts:
        raise ValueError("未能从任何文件中成功提取专家数量。")

    return layer_expert_counts

def update_config_file(
    config_path: str, 
    layer_expert_counts: Dict[int, int], 
    num_layers: int,
    backup: bool = True
) -> None:
    """
    使用每层的专家数量更新模型的config.json文件。

    Args:
        config_path: `config.json` 文件的路径。
        layer_expert_counts: 层索引到专家数量的映射。
        num_layers: 模型预期的总层数。
        backup: 是否创建原始配置文件的备份。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    # 1. 将字典转换为有序列表，并检查是否有缺失的层
    expert_counts_list: List[int] = []
    for i in range(num_layers):
        if i not in layer_expert_counts:
            raise ValueError(f"第 {i} 层的聚类结果缺失，无法继续。")
        expert_counts_list.append(layer_expert_counts[i])

    print(f"\n✅ 已为所有 {num_layers} 层准备好专家数量列表。")

    backup_path = ""

    # 2. 创建备份
    if backup:
        backup_path = config_path + ".bak"
        print(f"💾 正在创建备份: {backup_path}")
        os.rename(config_path, backup_path)

    # 3. 读取配置文件（从备份中读取）
    try:
        
        with open(backup_path if backup else config_path, 'r') as f:
            config_data = json.load(f)

        # 4. 添加或更新 `layer_expert_counts` 字段
        config_data['layer_expert_counts'] = expert_counts_list
        
        # (可选) 添加一些元数据
        total_experts = sum(expert_counts_list)
        avg_experts = total_experts / num_layers
        config_data['_expert_statistics'] = {
            "total_experts": total_experts,
            "average_experts_per_layer": round(avg_experts, 2),
            "min_experts": min(expert_counts_list),
            "max_experts": max(expert_counts_list),
        }


        # 5. 写回新的配置文件
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"🚀 配置文件 '{config_path}' 更新成功！")
        print(f"   - 新增字段 'layer_expert_counts': {expert_counts_list}")
        print(f"   - 总专家数: {total_experts}, 平均每层专家数: {avg_experts:.2f}")

    except Exception as e:
        print(f"❌ 更新配置文件时发生错误: {e}")
        # 如果出错，恢复备份
        if backup:
            print(f"🔄 正在从备份恢复...")
            os.rename(backup_path, config_path)
        raise

def main():
    parser = argparse.ArgumentParser(
        description="根据聚类结果更新模型的config.json，以支持每层不同的专家数量。"
    )
    parser.add_argument(
        "--cluster_dir",
        type=str,
        required=True,
        help="包含 `cluster_info_layer_*.json` 文件的目录路径。"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="要更新模型路径"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=24,
        help="模型预期的总层数 (例如 Qwen1.5-MoE-A2.7B 为 24)。"
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不创建原始配置文件的备份。"
    )
    parser.add_argument(
        "--CLUSTER_N",
        type=int, default=30,
        help="聚类时的专家数量 (仅用于日志显示)。"
    )
    
    args = parser.parse_args()

    try:
        # path 里添加聚类数量
        cluster_dir = os.path.join(args.cluster_dir, f"kmeans_clusters_{args.CLUSTER_N}")

        config_path = os.path.join(args.config_path, f"qwen1.5_moe_merged_svd_cluster_{args.CLUSTER_N}/config.json")

        # 步骤 1: 扫描并获取每层的专家数
        layer_counts_dict = scan_cluster_results(cluster_dir)

        # 步骤 2: 更新配置文件
        update_config_file(
            config_path=config_path,
            layer_expert_counts=layer_counts_dict,
            num_layers=args.num_layers,
            backup=not args.no_backup
        )
        print("\n🎉 操作完成！")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n💥 错误: {e}")
        exit(1)

if __name__ == "__main__":
    main()