#!/usr/bin/env python3
"""
Galvatron CostModel 快速入门示例

展示如何使用新的 Galvatron CostModel 进行：
1. 创建配置
2. 预测单个配置
3. 比较多个配置
4. 使用实测数据校准
"""

import sys
sys.path.insert(0, '/root/paddlejob/workspace/env_run/zhangdongqi')

from galvatron_costmodel import (
    GalvatronCostModel, CostModelConfig,
    HardwareConfig, GPUSpecs, NetworkTopology, ClusterConfig,
    ModelConfig, TrainingConfig, ParallelConfig,
    ZeROConfig, ActivationCheckpointConfig, ZeROStage, CheckpointGranularity,
    create_qwen3_30b_costmodel,
)


def example_1_basic_usage():
    """示例 1: 基本使用"""
    print("=" * 80)
    print("示例 1: 基本使用")
    print("=" * 80)
    
    # 使用预设配置创建 CostModel
    cm = create_qwen3_30b_costmodel(
        gpu_memory_gb=80.0,
        num_nodes=1,
        gpus_per_node=8
    )
    
    # 创建并行配置
    parallel = ParallelConfig(
        dp_degree=1,
        tp_degree=4,
        pp_degree=2,
        ep_degree=8
    )
    
    # 预测
    result = cm.predict_full(
        parallel,
        micro_batch_size=1,
        sequence_length=2048
    )
    
    print(f"\n并行配置: DP1-TP4-PP2-EP8")
    print(f"  预测时延: {result.total_step_time_ms:.2f} ms")
    print(f"  显存占用: {result.memory_breakdown.total_memory_gb:.2f} GB")
    print(f"  是否满足约束: {'✅' if result.fits_memory else '❌'}")
    print(f"  MFU: {result.mfu:.1%}")
    print()


def example_2_custom_config():
    """示例 2: 自定义配置"""
    print("=" * 80)
    print("示例 2: 自定义配置")
    print("=" * 80)
    
    # 自定义硬件配置
    hardware = HardwareConfig(
        gpu=GPUSpecs(
            memory_gb=80.0,
            bf16_tflops=989.0,
            memory_bandwidth_gbps=3350.0,
        ),
        network=NetworkTopology(
            intra_node_bandwidth_gbps=900.0,  # NVLink 4
            inter_node_bandwidth_gbps=200.0,  # 200 Gbps IB
            intra_node_latency_us=1.0,
            inter_node_latency_us=5.0,
        ),
        cluster=ClusterConfig(
            num_nodes=1,
            gpus_per_node=8,
        )
    )
    
    # 自定义模型配置 (Qwen3-30B-A3B)
    model = ModelConfig(
        num_layers=48,
        hidden_size=6144,
        intermediate_size=16384,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts=128,
        moe_top_k=8,
        num_moe_layers=24,
    )
    
    # 自定义训练配置
    training = TrainingConfig(
        micro_batch_size=1,
        sequence_length=4096,
        gradient_accumulation_steps=64,
    )
    
    # 创建配置
    config = CostModelConfig(
        hardware=hardware,
        model=model,
        training=training,
    )
    
    # 创建 CostModel
    cm = GalvatronCostModel(config)
    
    # 预测多个配置
    configs = [
        {"dp": 1, "tp": 8, "pp": 1, "ep": 8},
        {"dp": 1, "tp": 4, "pp": 2, "ep": 8},
        {"dp": 2, "tp": 4, "pp": 1, "ep": 4},
    ]
    
    print("\n不同并行配置的预测结果:")
    print("-" * 80)
    print(f"{'配置':<20} {'时延(ms)':<15} {'显存(GB)':<15} {'状态':<10}")
    print("-" * 80)
    
    for cfg in configs:
        parallel = ParallelConfig(
            dp_degree=cfg["dp"],
            tp_degree=cfg["tp"],
            pp_degree=cfg["pp"],
            ep_degree=cfg["ep"]
        )
        
        result = cm.predict_full(parallel)
        config_str = f"DP{cfg['dp']}-TP{cfg['tp']}-PP{cfg['pp']}-EP{cfg['ep']}"
        status = "✅" if result.fits_memory else "❌"
        
        print(f"{config_str:<20} {result.total_step_time_ms:<15.2f} "
              f"{result.memory_breakdown.total_memory_gb:<15.2f} {status:<10}")
    
    print()


def example_3_memory_breakdown():
    """示例 3: 显存详细分解"""
    print("=" * 80)
    print("示例 3: 显存详细分解")
    print("=" * 80)
    
    cm = create_qwen3_30b_costmodel()
    
    parallel = ParallelConfig(
        dp_degree=1,
        tp_degree=4,
        pp_degree=2,
        ep_degree=8
    )
    
    # 使用不同的优化策略
    optimizations = [
        ("无优化", ZeROConfig(stage=ZeROStage.NONE), 
         ActivationCheckpointConfig(granularity=CheckpointGranularity.NONE)),
        ("ZeRO-1", ZeROConfig(stage=ZeROStage.STAGE_1),
         ActivationCheckpointConfig(granularity=CheckpointGranularity.NONE)),
        ("ZeRO-2 + Selective Recompute", ZeROConfig(stage=ZeROStage.STAGE_2),
         ActivationCheckpointConfig(granularity=CheckpointGranularity.SELECTIVE)),
        ("ZeRO-2 + Full Recompute", ZeROConfig(stage=ZeROStage.STAGE_2),
         ActivationCheckpointConfig(granularity=CheckpointGranularity.FULL)),
    ]
    
    print("\n不同优化策略的显存占用:")
    print("-" * 100)
    print(f"{'策略':<30} {'参数':<10} {'梯度':<10} {'优化器':<10} {'激活':<10} {'总计':<10} {'状态':<8}")
    print("-" * 100)
    
    for name, zero_cfg, ckpt_cfg in optimizations:
        breakdown = cm.predict_memory(parallel, zero_cfg, ckpt_cfg)
        
        status = "✅" if breakdown.total_memory_gb <= 80.0 else "❌"
        
        print(f"{name:<30} "
              f"{breakdown.parameter_memory_gb:<10.2f} "
              f"{breakdown.gradient_memory_gb:<10.2f} "
              f"{breakdown.optimizer_memory_gb:<10.2f} "
              f"{breakdown.activation_memory_gb:<10.2f} "
              f"{breakdown.total_memory_gb:<10.2f} "
              f"{status:<8}")
    
    print()


def example_4_rank_configurations():
    """示例 4: 配置排序"""
    print("=" * 80)
    print("示例 4: 配置排序（自动找最优）")
    print("=" * 80)
    
    cm = create_qwen3_30b_costmodel()
    
    # 生成候选配置
    candidate_configs = [
        {"dp_degree": 1, "tp_degree": 8, "pp_degree": 1, "ep_degree": 8},
        {"dp_degree": 1, "tp_degree": 4, "pp_degree": 2, "ep_degree": 8},
        {"dp_degree": 1, "tp_degree": 2, "pp_degree": 4, "ep_degree": 8},
        {"dp_degree": 2, "tp_degree": 4, "pp_degree": 1, "ep_degree": 8},
        {"dp_degree": 2, "tp_degree": 4, "pp_degree": 1, "ep_degree": 4},
        {"dp_degree": 1, "tp_degree": 8, "pp_degree": 1, "ep_degree": 4},
        {"dp_degree": 2, "tp_degree": 2, "pp_degree": 2, "ep_degree": 4},
    ]
    
    # 排序（会自动打印报告）
    ranked = cm.rank_configurations(candidate_configs, top_k=5)
    
    print()


def example_5_time_breakdown():
    """示例 5: 时延分解"""
    print("=" * 80)
    print("示例 5: 时延详细分解")
    print("=" * 80)
    
    cm = create_qwen3_30b_costmodel()
    
    parallel = ParallelConfig(
        dp_degree=1,
        tp_degree=4,
        pp_degree=2,
        ep_degree=8
    )
    
    # 获取时延分解
    time_pred = cm.predict_step_time(parallel, micro_batch_size=1, sequence_length=2048)
    
    print("\n时延分解 (DP1-TP4-PP2-EP8, seq=2048):")
    print("-" * 60)
    print(f"  前向时间:       {time_pred['forward_time_ms']:>10.2f} ms")
    print(f"  反向时间:       {time_pred['backward_time_ms']:>10.2f} ms")
    print(f"  计算总时间:     {time_pred['compute_time_ms']:>10.2f} ms")
    print("-" * 60)
    print(f"  TP 通信时间:    {time_pred['tp_comm_time_ms']:>10.2f} ms")
    print(f"  DP 通信时间:    {time_pred['dp_comm_time_ms']:>10.2f} ms")
    print(f"  EP 通信时间:    {time_pred['ep_comm_time_ms']:>10.2f} ms")
    print(f"  PP 通信时间:    {time_pred['pp_comm_time_ms']:>10.2f} ms")
    print("-" * 60)
    print(f"  气泡时间:       {time_pred['bubble_time_ms']:>10.2f} ms")
    print(f"  气泡比例:       {time_pred['bubble_ratio']:>10.1%}")
    print("-" * 60)
    print(f"  总时延:         {time_pred['total_step_time_ms']:>10.2f} ms")
    print()


if __name__ == "__main__":
    example_1_basic_usage()
    example_2_custom_config()
    example_3_memory_breakdown()
    example_4_rank_configurations()
    example_5_time_breakdown()
    
    print("=" * 80)
    print("所有示例运行完成！")
    print("=" * 80)