# ...existing code...

from transformers import Qwen2MoeForCausalLM

def merge_model_experts(
    model: Qwen2MoeForCausalLM,
    cluster_dir: str,
    result_dir: str,
    target_layers: List[int],
    merging_method: str = "svd"
) -> Qwen2MoeForCausalLM:
    """
    合并整个模型中指定层的专家
    
    Args:
        model: 原始模型
        cluster_dir: 聚类结果目录
        result_dir: 分析结果目录（用于加载激活频率）
        target_layers: 要合并的层列表
        merging_method: 合并方法 ("svd" 或 "frequency")
        
    Returns:
        merged_model: 合并后的模型
    """
    print(f"Starting expert merging for {len(target_layers)} layers...")
    print(f"Target layers: {target_layers}")
    print(f"Merging method: {merging_method}")
    
    # 创建模型副本
    merged_model = copy.deepcopy(model)
    
    # 统计信息
    merge_stats = {
        'total_layers_processed': 0,
        'total_experts_before': 0,
        'total_experts_after': 0,
        'layer_details': {}
    }
    
    for layer_idx in target_layers:
        print(f"\n{'='*50}")
        print(f"Processing layer {layer_idx}...")
        
        # 检查是否为MoE层
        layer = merged_model.model.layers[layer_idx]
        if not isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
            print(f"  ⚠️  Layer {layer_idx} is not a MoE layer, skipping...")
            continue
        
        try:
            # 记录合并前的专家数量
            experts_before = len(layer.mlp.experts)
            merge_stats['total_experts_before'] += experts_before
            
            # 加载聚类结果和激活频率
            print(f"  📂 Loading clustering results for layer {layer_idx}...")
            cluster_labels, cluster_info = load_clustering_results(cluster_dir, layer_idx)
            
            print(f"  📂 Loading activation frequencies for layer {layer_idx}...")
            expert_frequencies = load_activation_frequency(result_dir, layer_idx)
            
            print(f"  🔍 Layer {layer_idx} info:")
            print(f"     - Experts before: {experts_before}")
            print(f"     - Target clusters: {cluster_info['n_clusters']}")
            print(f"     - Cluster sizes: {cluster_info['cluster_sizes']}")
            
            # 合并专家
            print(f"  🔄 Merging experts using {merging_method} method...")
            merged_moe = merge_experts_in_moe_layer(
                layer.mlp, 
                cluster_labels, 
                expert_frequencies,
                merging_method
            )
            
            # 替换层
            layer.mlp = merged_moe
            
            # 记录合并后的专家数量
            experts_after = len(merged_moe.experts)
            merge_stats['total_experts_after'] += experts_after
            merge_stats['total_layers_processed'] += 1
            
            # 记录层级详细信息
            merge_stats['layer_details'][layer_idx] = {
                'experts_before': experts_before,
                'experts_after': experts_after,
                'compression_ratio': experts_before / experts_after if experts_after > 0 else float('inf'),
                'cluster_sizes': cluster_info['cluster_sizes']
            }
            
            print(f"  ✅ Layer {layer_idx} merged successfully: {experts_before} -> {experts_after} experts")
            print(f"     Compression ratio: {experts_before/experts_after:.2f}x")
            
        except FileNotFoundError as e:
            print(f"  ❌ Error: Missing required files for layer {layer_idx}")
            print(f"     {str(e)}")
            continue
            
        except Exception as e:
            print(f"  ❌ Error processing layer {layer_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印总结统计
    print(f"\n{'='*60}")
    print("🎉 EXPERT MERGING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed layers: {merge_stats['total_layers_processed']}/{len(target_layers)}")
    print(f"📊 Total experts before merging: {merge_stats['total_experts_before']}")
    print(f"📊 Total experts after merging: {merge_stats['total_experts_after']}")
    
    if merge_stats['total_experts_before'] > 0:
        overall_compression = merge_stats['total_experts_before'] / merge_stats['total_experts_after']
        print(f"🎯 Overall compression ratio: {overall_compression:.2f}x")
        print(f"💾 Model size reduction: {((merge_stats['total_experts_before'] - merge_stats['total_experts_after']) / merge_stats['total_experts_before']) * 100:.1f}%")
    
    print(f"\nPer-layer details:")
    for layer_idx, details in merge_stats['layer_details'].items():
        print(f"  Layer {layer_idx}: {details['experts_before']} -> {details['experts_after']} experts ({details['compression_ratio']:.2f}x)")
    
    return merged_model

def save_merged_model(
    merged_model: Qwen2MoeForCausalLM,
    tokenizer,  # 添加分词器参数
    output_dir: str,
    model_name: str = "merged_model",
    save_config: bool = True
) -> str:
    """
    保存合并后的模型
    
    Args:
        merged_model: 合并后的模型
        tokenizer: 分词器
        output_dir: 输出目录
        model_name: 模型名称
        save_config: 是否保存配置信息
        
    Returns:
        model_path: 保存的模型路径
    """
    model_path = os.path.join(output_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("💾 SAVING MERGED MODEL")
    print(f"{'='*50}")
    print(f"📁 Output directory: {model_path}")
    
    try:
        # 保存模型
        print("🔄 Saving model weights and configuration...")
        merged_model.save_pretrained(
            model_path,
            safe_serialization=True,  # 使用SafeTensors格式
            max_shard_size="2GB"      # 分片大小
        )
        
        # 保存分词器
        if tokenizer is not None:
            print("🔄 Saving tokenizer...")
            tokenizer.save_pretrained(model_path)
        
        # 保存额外的配置信息
        if save_config:
            print("🔄 Saving merge configuration...")
            merge_info = {
                "merge_timestamp": str(torch.datetime.now()),
                "original_model_type": "Qwen2MoeForCausalLM",
                "merged_layers": [],  # 这个可以在调用时填充
                "merging_method": "svd",
                "total_parameters": sum(p.numel() for p in merged_model.parameters()),
                "trainable_parameters": sum(p.numel() for p in merged_model.parameters() if p.requires_grad),
            }
            
            # 统计每层的专家数量
            moe_layer_info = {}
            for i, layer in enumerate(merged_model.model.layers):
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock):
                    moe_layer_info[f"layer_{i}"] = {
                        "num_experts": len(layer.mlp.experts),
                        "is_moe_layer": True
                    }
                else:
                    moe_layer_info[f"layer_{i}"] = {
                        "is_moe_layer": False
                    }
            
            merge_info["moe_layers_info"] = moe_layer_info
            
            config_path = os.path.join(model_path, "merge_info.json")
            with open(config_path, 'w') as f:
                json.dump(merge_info, f, indent=2, default=str)
            
            print(f"📝 Merge configuration saved to: {config_path}")
        
        # 验证保存是否成功
        print("🔍 Validating saved model...")
        saved_files = os.listdir(model_path)
        required_files = ['config.json']
        
        missing_files = [f for f in required_files if f not in saved_files]
        if missing_files:
            print(f"⚠️  Warning: Missing files: {missing_files}")
        else:
            print("✅ All required files saved successfully!")
        
        # 计算模型大小
        total_size = sum(os.path.getsize(os.path.join(model_path, f)) 
                        for f in saved_files if os.path.isfile(os.path.join(model_path, f)))
        size_gb = total_size / (1024**3)
        
        print(f"📊 Model statistics:")
        print(f"   - Total parameters: {sum(p.numel() for p in merged_model.parameters()):,}")
        print(f"   - Model size on disk: {size_gb:.2f} GB")
        print(f"   - Number of files: {len(saved_files)}")
        
        print(f"✅ Model successfully saved to: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """主函数：执行专家合并"""
    # 配置参数
    MODEL_PATH = "/root/fsas/models/Qwen/Qwen1.5-MoE-A2.7B"
    CLUSTER_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/analysis_results"  # 聚类结果存放位置
    RESULT_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/analysis_results"   # 激活频率存放位置
    OUTPUT_DIR = "/root/fsas/zhanghongyu/SMoE/qwen/merged_models"
    
    # 要合并的MoE层 (Qwen1.5-MoE的MoE层通常是奇数层)
    TARGET_LAYERS = [1, 3, 5, 7, 9]  
    MERGING_METHOD = "svd"  # 可选: "svd" 或 "frequency"
    
    try:
        print("🚀 Starting Expert Merging Pipeline")
        print(f"{'='*60}")
        
        # 加载原始模型和分词器
        print("📂 Loading original model and tokenizer...")
        from transformers import AutoTokenizer
        
        model = Qwen2MoeForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # 在CPU上进行合并以节省显存
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # 执行专家合并
        merged_model = merge_model_experts(
            model=model,
            cluster_dir=CLUSTER_DIR,
            result_dir=RESULT_DIR,
            target_layers=TARGET_LAYERS,
            merging_method=MERGING_METHOD
        )
        
        # 保存合并后的模型
        model_name = f"qwen1.5_moe_merged_{MERGING_METHOD}_layers_{'_'.join(map(str, TARGET_LAYERS))}"
        saved_path = save_merged_model(
            merged_model=merged_model,
            tokenizer=tokenizer,
            output_dir=OUTPUT_DIR,
            model_name=model_name
        )
        
        print(f"\n🎉 Expert merging pipeline completed successfully!")
        print(f"🎯 Merged model saved to: {saved_path}")
        
    except Exception as e:
        print(f"💥 Fatal error during merging: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()