import os
import torch
import json
from tqdm import tqdm
import argparse
import numpy as np
from typing import cast, Dict

# 配置路径
SIMILARITY_RESULTS_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/analysis_results/similarity_results"
REDUNDANCY_OUTPUT_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/analysis_results/redundancy_results"
NUM_LAYERS = 24   # Qwen1.5-MoE-A2.7B 的层数


def calculate_layer_redundancy(similarity_matrix: torch.Tensor) -> float:
    """
    计算单层的冗余度：相似度矩阵中每对专家的相似度平均值
    
    Args:
        similarity_matrix: 专家相似度矩阵 (num_experts, num_experts)
        
    Returns:
        冗余度（平均相似度，排除对角线）
    """
    num_experts = similarity_matrix.shape[0]
    
    # 排除对角线元素，计算所有专家对的平均相似度
    mask = ~torch.eye(num_experts, dtype=torch.bool)
    off_diagonal_similarities = similarity_matrix[mask]
    redundancy = off_diagonal_similarities.mean().item()
    
    return redundancy


def allocate_experts_by_redundancy(layer_redundancy: dict, group_size: int, target_avg_experts: int, 
                                 min_experts: int = 5, max_experts: int = 55) -> dict:
    """
    基于冗余度为相邻n层分配专家数量
    
    Args:
        layer_redundancy: 每层的冗余度字典 {layer_idx: redundancy_value}
        group_size: 相邻层的组大小 n
        target_avg_experts: 目标平均专家数
        min_experts: 最少专家数
        max_experts: 最多专家数
        
    Returns:
        每层的专家分配 {layer_idx: num_experts}
    """
    print(f"\n🎯 开始基于冗余度分配专家数量...")
    print(f"   - 组大小: {group_size} 层")
    print(f"   - 目标平均专家数: {target_avg_experts}")
    print(f"   - 专家数范围: [{min_experts}, {max_experts}]")
    
    expert_allocation: Dict[int, int] = {}
    
    # 按组处理层
    num_groups = (NUM_LAYERS + group_size - 1) // group_size  # 向上取整
    
    for group_idx in range(num_groups):
        start_layer = group_idx * group_size
        end_layer = min(start_layer + group_size, NUM_LAYERS)
        
        print(f"\n📊 处理第 {group_idx + 1} 组: 第 {start_layer} - {end_layer - 1} 层")
        
        # 获取该组的冗余度值
        group_layers = list(range(start_layer, end_layer))
        group_redundancies = []
        
        for layer_idx in group_layers:
            if layer_idx in layer_redundancy:
                group_redundancies.append(layer_redundancy[layer_idx])
            else:
                print(f"   ⚠️  第 {layer_idx} 层没有冗余度数据，使用平均值")
                # 如果某层缺失，使用已有层的平均值
                if group_redundancies:
                    group_redundancies.append(sum(group_redundancies) / len(group_redundancies))
                else:
                    group_redundancies.append(0.8)  # 默认值
        
        # 计算该组的总专家预算
        group_target_total = target_avg_experts * len(group_layers)
        
        # 基于冗余度分配：冗余度高的层分配较少专家，冗余度低的层分配较多专家
        # 使用反比例分配：专家数 ∝ 1 / 冗余度
        inverse_redundancies = [1.0 / max(r, 0.00000001) for r in group_redundancies]  # 避免除零
        total_inverse: float = cast(float, sum(inverse_redundancies))
        
        # 按比例分配专家数
        raw_allocations = [(inv_r / total_inverse) * group_target_total for inv_r in inverse_redundancies]
        
        # 约束到范围并取整
        constrained_allocations = [max(min_experts, min(max_experts, round(alloc))) for alloc in raw_allocations]
        
        # 调整总数以匹配目标
        current_total = sum(constrained_allocations)
        target_total = round(group_target_total)
        diff = target_total - current_total
        
        # 如果有差异，优先调整冗余度最高/最低的层
        if diff != 0:
            # 按冗余度排序，准备调整
            sorted_indices = sorted(range(len(group_redundancies)), 
                                  key=lambda i: group_redundancies[i], 
                                  reverse=(diff < 0))  # diff<0时从高冗余度开始减少
            
            # 修复后的逻辑：确保能够完全消除差异
            remaining_diff = diff
            attempts = 0
            max_attempts = abs(diff) * len(sorted_indices)  # 避免无限循环
            
            while remaining_diff != 0 and attempts < max_attempts:
                adjusted = False  # 标记本轮是否有调整
                
                for idx in sorted_indices:
                    if remaining_diff == 0:
                        break
                        
                    if remaining_diff > 0:  # 需要增加专家
                        if constrained_allocations[idx] < max_experts:
                            constrained_allocations[idx] += 1
                            remaining_diff -= 1
                            adjusted = True
                    else:  # 需要减少专家 (remaining_diff < 0)
                        if constrained_allocations[idx] > min_experts:
                            constrained_allocations[idx] -= 1
                            remaining_diff += 1
                            adjusted = True
                
                # 如果本轮没有任何调整，说明无法进一步调整，跳出循环
                if not adjusted:
                    break
                    
                attempts += 1
            
            # 如果还有剩余差异，给出警告
            if remaining_diff != 0:
                print(f"   ⚠️  警告: 无法完全消除差异，剩余差异={remaining_diff}")
        
        # 记录分配结果
        for i, layer_idx in enumerate(group_layers):
            expert_allocation[layer_idx] = constrained_allocations[i]
            print(f"   第 {layer_idx:2d} 层: 冗余度={group_redundancies[i]:.4f} -> 专家数={constrained_allocations[i]}")
        
        group_actual_total = sum(constrained_allocations)
        group_actual_avg = group_actual_total / len(group_layers)
        print(f"   组汇总: 总专家数={group_actual_total}, 实际平均={group_actual_avg:.2f}")
    
    return expert_allocation


def main():
    """
    主函数：计算所有层的冗余度并基于冗余度分配专家数量
    """
    parser = argparse.ArgumentParser(description="计算专家冗余度并分配专家数量")
    parser.add_argument("--group_size", type=int, default=4, help="相邻层组的大小")
    parser.add_argument("--target_avg_CLUSTER_N", type=float, default=30.0, help="目标平均专家数")
    parser.add_argument("--min_experts", type=int, default=15, help="最少专家数")
    parser.add_argument("--max_experts", type=int, default=45, help="最多专家数")
    parser.add_argument("--allocate_experts", action="store_true", help="是否进行专家分配")
    
    args = parser.parse_args()
    
    print("🚀 开始计算专家冗余度...")
    
    # 创建输出目录
    os.makedirs(REDUNDANCY_OUTPUT_DIR, exist_ok=True)
    
    # 存储结果
    layer_redundancy = {}
    
    # 遍历所有层
    for layer_idx in tqdm(range(NUM_LAYERS), desc="计算各层冗余度"):
        similarity_file = os.path.join(SIMILARITY_RESULTS_DIR, f"avg_similarity_matrix_layer_{layer_idx}.pt")
        
        if not os.path.exists(similarity_file):
            print(f"⚠️  警告: 未找到第 {layer_idx} 层的相似度矩阵文件")
            continue
        
        # 加载相似度矩阵并计算冗余度
        try:
            similarity_matrix = torch.load(similarity_file)
            redundancy = calculate_layer_redundancy(similarity_matrix)
            layer_redundancy[layer_idx] = redundancy
            
            print(f"第 {layer_idx:2d} 层冗余度: {redundancy:.4f}")
            
        except Exception as e:
            print(f"❌ 处理第 {layer_idx} 层时出错: {e}")
            continue
    
    # 保存冗余度结果
    results_file = os.path.join(REDUNDANCY_OUTPUT_DIR, "layer_redundancy.json")
    with open(results_file, 'w') as f:
        json.dump(layer_redundancy, f, indent=2)
    
    print(f"\n💾 冗余度结果已保存到: {results_file}")
    
    # 输出汇总信息
    if layer_redundancy:
        avg_redundancy = sum(layer_redundancy.values()) / len(layer_redundancy)
        print(f"📊 平均冗余度: {avg_redundancy:.4f}")
        print(f"📊 处理层数: {len(layer_redundancy)}/{NUM_LAYERS}")
    
    # 如果启用了专家分配功能
    if args.allocate_experts and layer_redundancy:
        expert_allocation = allocate_experts_by_redundancy(
            layer_redundancy, 
            args.group_size, 
            args.target_avg_CLUSTER_N,
            args.min_experts,
            args.max_experts
        )
        
        # 保存专家分配结果
        allocation_file = os.path.join(REDUNDANCY_OUTPUT_DIR, 
                                     f"expert_allocation_group{args.group_size}_avg{args.target_avg_CLUSTER_N}.json")
        with open(allocation_file, 'w') as f:
            json.dump({
                "config": {
                    "group_size": args.group_size,
                    "target_avg_cluster_n": args.target_avg_CLUSTER_N,
                    "min_experts": args.min_experts,
                    "max_experts": args.max_experts
                },
                "allocation": expert_allocation,
                "statistics": {
                    "total_experts": sum(expert_allocation.values()),
                    "actual_avg_experts": sum(expert_allocation.values()) / len(expert_allocation),
                    "min_allocated": min(expert_allocation.values()),
                    "max_allocated": max(expert_allocation.values())
                }
            }, f, indent=2)
        
        print(f"\n💾 专家分配结果已保存到: {allocation_file}")
        
        # 输出分配统计
        total_experts = sum(expert_allocation.values())
        actual_avg = total_experts / len(expert_allocation)
        print(f"🎯 分配统计:")
        print(f"   - 总专家数: {total_experts}")
        print(f"   - 实际平均专家数: {actual_avg:.2f}")
        print(f"   - 最少分配: {min(expert_allocation.values())}")
        print(f"   - 最多分配: {max(expert_allocation.values())}")
    
    print("🎉 分析完成!")


if __name__ == "__main__":
    main()