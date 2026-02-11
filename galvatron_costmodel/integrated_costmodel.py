#!/usr/bin/env python3
"""
集成 CostModel - 参考 Galvatron 的整合设计

核心功能：
1. 整合硬件配置、通信模型、计算模型、显存模型
2. 预测完整 step 时延
3. 预测显存占用
4. 自动选择优化策略
5. 配置排序和验证
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .hardware_config import HardwareConfig, GPUSpecs, NetworkTopology, ClusterConfig
from .communication_model import CommunicationModel, CommResult
from .computation_model import ComputationModel, ModelConfig, ComputeMode
from .memory_model import (
    MemoryModel, MemoryBreakdown, ParallelConfig, TrainingConfig,
    ZeROConfig, ActivationCheckpointConfig, OffloadConfig,
    ZeROStage, CheckpointGranularity, AutoMemoryOptimizer
)


@dataclass
class CostModelConfig:
    """CostModel 配置"""
    # 硬件配置
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 训练配置
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CostModelConfig":
        """从字典创建配置"""
        config = cls()
        
        if "hardware" in data:
            hw = data["hardware"]
            config.hardware = HardwareConfig(
                gpu=GPUSpecs(
                    memory_gb=hw.get("gpu_memory_gb", 80.0),
                    bf16_tflops=hw.get("bf16_tflops", 989.0),
                ),
                network=NetworkTopology(
                    intra_node_bandwidth_gbps=hw.get("intra_node_bandwidth_gbps", 900.0),
                    inter_node_bandwidth_gbps=hw.get("inter_node_bandwidth_gbps", 200.0),
                ),
                cluster=ClusterConfig(
                    num_nodes=hw.get("num_nodes", 1),
                    gpus_per_node=hw.get("gpus_per_node", 8),
                ),
            )
        
        if "model" in data:
            m = data["model"]
            config.model = ModelConfig(
                num_layers=m.get("num_layers", 48),
                hidden_size=m.get("hidden_size", 6144),
                intermediate_size=m.get("intermediate_size", 16384),
                num_attention_heads=m.get("num_attention_heads", 32),
                num_key_value_heads=m.get("num_key_value_heads", 4),
                num_experts=m.get("num_experts", 128),
                moe_top_k=m.get("moe_top_k", 8),
                num_moe_layers=m.get("num_moe_layers", 24),
                vocab_size=m.get("vocab_size", 152064),
            )
        
        if "training" in data:
            t = data["training"]
            config.training = TrainingConfig(
                micro_batch_size=t.get("micro_batch_size", 1),
                sequence_length=t.get("sequence_length", 8192),
                global_batch_size=t.get("global_batch_size", 512),
                gradient_accumulation_steps=t.get("gradient_accumulation_steps", 64),
            )
        
        return config
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "hardware": {
                "gpu_memory_gb": self.hardware.gpu.memory_gb,
                "bf16_tflops": self.hardware.gpu.bf16_tflops,
                "intra_node_bandwidth_gbps": self.hardware.network.intra_node_bandwidth_gbps,
                "inter_node_bandwidth_gbps": self.hardware.network.inter_node_bandwidth_gbps,
                "num_nodes": self.hardware.cluster.num_nodes,
                "gpus_per_node": self.hardware.cluster.gpus_per_node,
            },
            "model": {
                "num_layers": self.model.num_layers,
                "hidden_size": self.model.hidden_size,
                "intermediate_size": self.model.intermediate_size,
                "num_attention_heads": self.model.num_attention_heads,
                "num_key_value_heads": self.model.num_key_value_heads,
                "num_experts": self.model.num_experts,
                "moe_top_k": self.model.moe_top_k,
                "num_moe_layers": self.model.num_moe_layers,
                "vocab_size": self.model.vocab_size,
            },
            "training": {
                "micro_batch_size": self.training.micro_batch_size,
                "sequence_length": self.training.sequence_length,
                "global_batch_size": self.training.global_batch_size,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            },
        }


@dataclass
class PredictionResult:
    """预测结果"""
    # 时延预测 (ms)
    total_step_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    
    # 通信时延 (ms)
    tp_comm_time_ms: float = 0.0
    dp_comm_time_ms: float = 0.0
    ep_comm_time_ms: float = 0.0
    pp_comm_time_ms: float = 0.0
    
    # 流水线
    bubble_time_ms: float = 0.0
    bubble_ratio: float = 0.0
    
    # 显存 (GB)
    memory_breakdown: MemoryBreakdown = field(default_factory=MemoryBreakdown)
    fits_memory: bool = True
    
    # 优化配置
    zero_config: ZeROConfig = field(default_factory=ZeROConfig)
    checkpoint_config: ActivationCheckpointConfig = field(default_factory=ActivationCheckpointConfig)
    offload_config: OffloadConfig = field(default_factory=OffloadConfig)
    
    # 效率指标
    compute_efficiency: float = 0.0
    mfu: float = 0.0  # Model FLOPs Utilization
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "time": {
                "total_step_time_ms": self.total_step_time_ms,
                "compute_time_ms": self.compute_time_ms,
                "forward_time_ms": self.forward_time_ms,
                "backward_time_ms": self.backward_time_ms,
                "tp_comm_time_ms": self.tp_comm_time_ms,
                "dp_comm_time_ms": self.dp_comm_time_ms,
                "ep_comm_time_ms": self.ep_comm_time_ms,
                "pp_comm_time_ms": self.pp_comm_time_ms,
                "bubble_time_ms": self.bubble_time_ms,
                "bubble_ratio": self.bubble_ratio,
            },
            "memory": self.memory_breakdown.to_dict(),
            "fits_memory": self.fits_memory,
            "optimization": {
                "zero_stage": self.zero_config.stage.value,
                "checkpoint_granularity": self.checkpoint_config.granularity.value,
                "optimizer_offload": self.offload_config.optimizer_offload.value,
            },
            "efficiency": {
                "compute_efficiency": self.compute_efficiency,
                "mfu": self.mfu,
            },
        }


class GalvatronCostModel:
    """
    Galvatron 风格的集成 CostModel
    
    整合：
    - 硬件配置（GPU、网络、集群）
    - 通信模型（TP/DP/EP/PP 通信）
    - 计算模型（α-β 模型）
    - 显存模型（精确分解）
    
    功能：
    - 预测 step 时延
    - 预测显存占用
    - 自动选择优化策略
    - 配置排序和验证
    """
    
    def __init__(self, config: CostModelConfig):
        self.config = config
        
        # 初始化子模型
        self.comm_model = CommunicationModel(config.hardware)
        self.compute_model = ComputationModel(
            config.hardware, config.model, ComputeMode.LINEAR
        )
        self.memory_model = MemoryModel(config.model, config.training)
        
        # 自动显存优化器
        self.memory_optimizer = AutoMemoryOptimizer(
            self.memory_model, config.hardware.gpu.memory_gb
        )
        
        # 校准数据
        self.calibration_data: Dict = {}
    
    def predict_step_time(self, parallel: ParallelConfig,
                          micro_batch_size: int = None,
                          sequence_length: int = None,
                          num_micro_batches: int = None) -> Dict[str, float]:
        """
        预测单步训练时延
        
        Args:
            parallel: 并行配置
            micro_batch_size: micro batch size（默认使用训练配置）
            sequence_length: 序列长度（默认使用训练配置）
            num_micro_batches: micro batch 数量（默认使用 gradient_accumulation_steps）
        
        Returns:
            时延详情字典
        """
        if micro_batch_size is None:
            micro_batch_size = self.config.training.micro_batch_size
        if sequence_length is None:
            sequence_length = self.config.training.sequence_length
        if num_micro_batches is None:
            num_micro_batches = self.config.training.gradient_accumulation_steps
        
        # ========== 计算时间 ==========
        forward_time = self.compute_model.estimate_forward_time(
            micro_batch_size, sequence_length,
            parallel.tp_degree, parallel.pp_degree, parallel.ep_degree
        )
        backward_time = self.compute_model.estimate_backward_time(
            micro_batch_size, sequence_length,
            parallel.tp_degree, parallel.pp_degree, parallel.ep_degree
        )
        
        # 流水线气泡
        bubble_time = self.compute_model.estimate_pipeline_bubble_time(
            forward_time, backward_time,
            parallel.pp_degree, num_micro_batches
        )
        
        compute_time = (forward_time + backward_time) * num_micro_batches + bubble_time
        
        # ========== 通信时间 ==========
        h = self.config.model.hidden_size
        activation_size = micro_batch_size * sequence_length * h * self.config.training.dtype_bytes
        
        # TP AllReduce (每层 2 次)
        layers_per_stage = self.config.model.num_layers // parallel.pp_degree
        tp_comm_result = self.comm_model.predict_tp_comm(activation_size, parallel.tp_degree)
        tp_comm_time = tp_comm_result.time_ms * 2 * layers_per_stage * num_micro_batches
        
        # EP AllToAll (MoE 层)
        moe_layers = self.config.model.num_moe_layers // parallel.pp_degree
        token_data_size = micro_batch_size * sequence_length * h * self.config.model.moe_top_k * self.config.training.dtype_bytes
        ep_comm_result = self.comm_model.predict_ep_comm(
            token_data_size, parallel.ep_degree,
            topk=self.config.model.moe_top_k,
            num_experts=self.config.model.num_experts
        )
        ep_comm_time = ep_comm_result.time_ms * moe_layers * num_micro_batches
        
        # PP P2P
        pp_comm_result = self.comm_model.predict_pp_comm(
            activation_size, parallel.pp_degree, num_micro_batches
        )
        pp_comm_time = pp_comm_result.time_ms
        
        # DP AllReduce/ReduceScatter (梯度同步)
        param_count = self.memory_model.estimate_parameter_count(parallel)
        grad_size = param_count["total_params"] * self.config.training.dtype_bytes
        dp_comm_result = self.comm_model.predict_dp_comm(
            grad_size, parallel.dp_degree, use_sharding=False
        )
        dp_comm_time = dp_comm_result.time_ms
        
        # ========== 总时延 ==========
        # 通信与计算的重叠
        # TP 通信在关键路径上，无法完全 overlap
        # DP 通信可以与后向计算部分 overlap
        # EP 通信在关键路径上
        
        overlap_factor = 0.3  # 假设 30% 的通信可以 overlap
        effective_comm_time = (
            tp_comm_time +
            ep_comm_time +
            pp_comm_time +
            dp_comm_time * (1 - overlap_factor)
        )
        
        total_time = compute_time + effective_comm_time
        
        # 气泡比例
        bubble_ratio = bubble_time / total_time if total_time > 0 else 0
        
        return {
            "total_step_time_ms": total_time,
            "compute_time_ms": compute_time,
            "forward_time_ms": forward_time * num_micro_batches,
            "backward_time_ms": backward_time * num_micro_batches,
            "tp_comm_time_ms": tp_comm_time,
            "dp_comm_time_ms": dp_comm_time,
            "ep_comm_time_ms": ep_comm_time,
            "pp_comm_time_ms": pp_comm_time,
            "bubble_time_ms": bubble_time,
            "bubble_ratio": bubble_ratio,
        }
    
    def predict_memory(self, parallel: ParallelConfig,
                       zero_config: ZeROConfig = None,
                       checkpoint_config: ActivationCheckpointConfig = None,
                       offload_config: OffloadConfig = None) -> MemoryBreakdown:
        """预测显存占用"""
        return self.memory_model.estimate_memory(
            parallel, zero_config, checkpoint_config, offload_config
        )
    
    def predict_full(self, parallel: ParallelConfig,
                     micro_batch_size: int = None,
                     sequence_length: int = None,
                     auto_optimize: bool = True) -> PredictionResult:
        """
        完整预测：时延 + 显存 + 自动优化
        
        Args:
            parallel: 并行配置
            micro_batch_size: micro batch size
            sequence_length: 序列长度
            auto_optimize: 是否自动选择优化策略
        """
        if micro_batch_size is None:
            micro_batch_size = self.config.training.micro_batch_size
        if sequence_length is None:
            sequence_length = self.config.training.sequence_length
        
        result = PredictionResult()
        
        # 自动选择优化策略
        if auto_optimize:
            zero_cfg, ckpt_cfg, offload_cfg, breakdown = self.memory_optimizer.find_optimal_config(parallel)
            result.zero_config = zero_cfg
            result.checkpoint_config = ckpt_cfg
            result.offload_config = offload_cfg
            result.memory_breakdown = breakdown
        else:
            zero_cfg = ZeROConfig()
            ckpt_cfg = ActivationCheckpointConfig()
            offload_cfg = OffloadConfig()
            result.memory_breakdown = self.predict_memory(parallel, zero_cfg, ckpt_cfg, offload_cfg)
            result.zero_config = zero_cfg
            result.checkpoint_config = ckpt_cfg
            result.offload_config = offload_cfg
        
        # 检查显存
        result.fits_memory = result.memory_breakdown.total_memory_gb <= self.config.hardware.gpu.memory_gb
        
        # 预测时延
        time_pred = self.predict_step_time(parallel, micro_batch_size, sequence_length)
        
        # 考虑 recompute 开销
        recompute_overhead = result.checkpoint_config.get_recompute_overhead()
        
        result.total_step_time_ms = time_pred["total_step_time_ms"] * recompute_overhead
        result.compute_time_ms = time_pred["compute_time_ms"] * recompute_overhead
        result.forward_time_ms = time_pred["forward_time_ms"] * recompute_overhead
        result.backward_time_ms = time_pred["backward_time_ms"] * recompute_overhead
        result.tp_comm_time_ms = time_pred["tp_comm_time_ms"]
        result.dp_comm_time_ms = time_pred["dp_comm_time_ms"]
        result.ep_comm_time_ms = time_pred["ep_comm_time_ms"]
        result.pp_comm_time_ms = time_pred["pp_comm_time_ms"]
        result.bubble_time_ms = time_pred["bubble_time_ms"]
        result.bubble_ratio = time_pred["bubble_ratio"]
        
        # 计算效率指标
        result.compute_efficiency = self._calculate_compute_efficiency(result, parallel)
        result.mfu = self._calculate_mfu(result, parallel, micro_batch_size, sequence_length)
        
        return result
    
    def _calculate_compute_efficiency(self, result: PredictionResult,
                                      parallel: ParallelConfig) -> float:
        """计算硬件利用效率"""
        if result.total_step_time_ms <= 0:
            return 0.0
        
        # 计算时间占总时间的比例
        compute_ratio = result.compute_time_ms / result.total_step_time_ms
        
        # 考虑并行效率损失
        tp_efficiency = 0.9 if parallel.tp_degree > 1 else 1.0
        pp_efficiency = 1.0 - result.bubble_ratio
        ep_efficiency = 0.85 if parallel.ep_degree > 1 else 1.0
        
        return compute_ratio * tp_efficiency * pp_efficiency * ep_efficiency
    
    def _calculate_mfu(self, result: PredictionResult,
                       parallel: ParallelConfig,
                       micro_batch_size: int,
                       sequence_length: int) -> float:
        """
        计算 Model FLOPs Utilization (MFU)
        
        MFU = 实际吞吐量 / 理论峰值吞吐量
        
        参考 Megatron-LM 的 MFU 计算方式
        """
        if result.total_step_time_ms <= 0:
            return 0.0
        
        # 估算模型 FLOPs (per token)
        h = self.config.model.hidden_size
        ffn = self.config.model.intermediate_size
        num_layers = self.config.model.num_layers
        vocab = self.config.model.vocab_size
        
        # 每个 token 的 FLOPs（前向传播）
        # Attention: 4 * h^2 (Q,K,V,O) + 2 * seq * h (attention scores, 近似)
        # MLP: 8 * h * ffn (SwiGLU: gate, up, down)
        # 对于 MoE 层，只有 topk 个专家被激活
        
        # 简化公式：每 token 约 6 * num_params FLOPs（前向）
        # 参考: https://arxiv.org/abs/2104.04473
        params_per_layer = 12 * h * h + 8 * h * ffn  # 粗略估算
        flops_per_token_forward = 2 * params_per_layer * num_layers
        
        # 总 tokens
        tokens = micro_batch_size * sequence_length
        num_micro_batches = self.config.training.gradient_accumulation_steps
        total_tokens = tokens * num_micro_batches
        
        # 总 FLOPs (前向 + 反向 ≈ 3x 前向)
        total_flops = flops_per_token_forward * total_tokens * 3
        
        # 实际 TFLOPS（单卡）
        actual_tflops = total_flops / (result.total_step_time_ms / 1000) / 1e12
        
        # 峰值 TFLOPS（单卡理论峰值）
        peak_tflops = self.config.hardware.gpu.get_effective_tflops("bf16")
        
        # 计算 MFU（考虑所有 GPU 一起工作）
        world_size = parallel.dp_degree * parallel.tp_degree * parallel.pp_degree
        
        # MFU = 实际单卡 TFLOPS / 峰值单卡 TFLOPS
        # 注意：total_flops 是总工作量，分摊到每卡后除以峰值
        mfu = (actual_tflops / world_size) / peak_tflops if peak_tflops > 0 else 0.0
        
        # 限制在合理范围内
        return min(mfu, 1.0)
    
    def rank_configurations(self, configs: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        对并行配置列表进行排序
        
        排序依据：
        1. 是否满足显存约束
        2. step 时延
        3. 硬件利用率
        """
        results = []
        
        for i, cfg in enumerate(configs):
            try:
                parallel = ParallelConfig(
                    dp_degree=cfg.get("dp_degree", 1),
                    tp_degree=cfg.get("tp_degree", 1),
                    pp_degree=cfg.get("pp_degree", 1),
                    ep_degree=cfg.get("ep_degree", 1),
                )
                
                micro_bsz = cfg.get("micro_batch_size", self.config.training.micro_batch_size)
                seq_len = cfg.get("sequence_length", self.config.training.sequence_length)
                
                prediction = self.predict_full(parallel, micro_bsz, seq_len)
                
                results.append({
                    "rank": 0,
                    "config": cfg,
                    "prediction": prediction.to_dict(),
                    "total_step_time_ms": prediction.total_step_time_ms,
                    "total_memory_gb": prediction.memory_breakdown.total_memory_gb,
                    "fits_memory": prediction.fits_memory,
                    "compute_efficiency": prediction.compute_efficiency,
                    "mfu": prediction.mfu,
                })
            except Exception as e:
                print(f"Warning: Failed to predict config {cfg}: {e}")
                continue
        
        # 排序：先按 fits_memory，再按时延
        results.sort(key=lambda x: (not x["fits_memory"], x["total_step_time_ms"]))
        
        # 更新排名
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        # 打印报告
        self._print_ranking_report(results[:top_k])
        
        return results[:top_k]
    
    def _print_ranking_report(self, results: List[Dict]):
        """打印排序报告"""
        if not results:
            return
        
        print("\n" + "=" * 130)
        print("🚀 Galvatron CostModel - 并行配置排序报告")
        print("=" * 130)
        print(f"{'排名':<4} {'并行配置':<25} {'时延(ms)':<12} {'显存(GB)':<12} "
              f"{'满足约束':<10} {'效率':<8} {'MFU':<8}")
        print("-" * 130)
        
        for r in results:
            cfg = r["config"]
            config_str = f"DP{cfg.get('dp_degree',1)}-TP{cfg.get('tp_degree',1)}-PP{cfg.get('pp_degree',1)}-EP{cfg.get('ep_degree',1)}"
            fits = "✅" if r["fits_memory"] else "❌"
            
            print(f"{r['rank']:<4} {config_str:<25} {r['total_step_time_ms']:<12.2f} "
                  f"{r['total_memory_gb']:<12.2f} {fits:<10} "
                  f"{r['compute_efficiency']:<8.1%} {r['mfu']:<8.1%}")
        
        print("-" * 130)
        
        if results:
            best = results[0]
            print(f"\n📊 最优配置: {best['config']}")
            print(f"   • 预计时延: {best['total_step_time_ms']:.2f} ms")
            print(f"   • 显存占用: {best['total_memory_gb']:.2f} GB")
            print(f"   • MFU: {best['mfu']:.1%}")
    
    def calibrate_with_data(self, calibration_data: List[Dict]):
        """
        使用实测数据校准模型
        
        calibration_data 格式：
        [
            {
                "parallel_config": {"dp": 1, "tp": 8, "pp": 1, "ep": 8},
                "micro_batch_size": 1,
                "sequence_length": 2048,
                "actual_step_time_ms": 150.0,
                "actual_a2a_time_ms": 20.0,
            },
            ...
        ]
        """
        if not calibration_data:
            return
        
        # 提取 A2A 校准数据
        a2a_calibration = []
        for data in calibration_data:
            if "actual_a2a_time_ms" in data:
                cfg = data.get("parallel_config", {})
                micro_bsz = data.get("micro_batch_size", 1)
                seq_len = data.get("sequence_length", 2048)
                ep = cfg.get("ep", cfg.get("ep_degree", 1))
                
                h = self.config.model.hidden_size
                topk = self.config.model.moe_top_k
                data_size = micro_bsz * seq_len * h * topk * self.config.training.dtype_bytes
                
                a2a_calibration.append({
                    "data_size_bytes": data_size,
                    "num_gpus": ep,
                    "actual_time_ms": data["actual_a2a_time_ms"],
                })
        
        # 校准 A2A 模型
        if a2a_calibration:
            self.comm_model.calibrate_alltoall(a2a_calibration)
        
        self.calibration_data = {"calibration_data": calibration_data}
        print(f"Calibrated with {len(calibration_data)} data points")
    
    def load_calibration(self, calibration_path: str):
        """从文件加载校准数据"""
        if not os.path.exists(calibration_path):
            print(f"Warning: Calibration file not found: {calibration_path}")
            return
        
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.calibrate_with_data(data)
        elif isinstance(data, dict) and "results" in data:
            # 支持从验证文件加载
            self.calibrate_with_data(data["results"])
    
    def save_config(self, path: str):
        """保存配置到文件"""
        data = self.config.to_dict()
        data["calibration"] = self.calibration_data
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_config_file(cls, path: str) -> "GalvatronCostModel":
        """从配置文件创建 CostModel"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = CostModelConfig.from_dict(data)
        model = cls(config)
        
        if "calibration" in data:
            model.calibration_data = data["calibration"]
        
        return model


# ==================== 便捷函数 ====================

def create_qwen3_30b_costmodel(gpu_memory_gb: float = 80.0,
                               num_nodes: int = 1,
                               gpus_per_node: int = 8) -> GalvatronCostModel:
    """
    创建 Qwen3-30B-A3B 的 CostModel
    
    预设配置：
    - 48 层 (24 Dense + 24 MoE)
    - hidden_size = 6144
    - intermediate_size = 16384
    - num_experts = 128
    - topk = 8
    """
    config = CostModelConfig(
        hardware=HardwareConfig(
            gpu=GPUSpecs(memory_gb=gpu_memory_gb),
            cluster=ClusterConfig(num_nodes=num_nodes, gpus_per_node=gpus_per_node),
        ),
        model=ModelConfig(
            num_layers=48,
            hidden_size=6144,
            intermediate_size=16384,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_experts=128,
            moe_top_k=8,
            num_moe_layers=24,
            vocab_size=152064,
        ),
        training=TrainingConfig(
            micro_batch_size=1,
            sequence_length=8192,
            gradient_accumulation_steps=64,
        ),
    )
    
    return GalvatronCostModel(config)


# ==================== 测试函数 ====================

def test_galvatron_costmodel():
    """测试 Galvatron CostModel"""
    print("=" * 80)
    print("Galvatron CostModel 测试")
    print("=" * 80)
    
    # 创建 CostModel
    cm = create_qwen3_30b_costmodel(gpu_memory_gb=80.0, num_nodes=1, gpus_per_node=8)
    
    # 测试配置
    test_configs = [
        {"dp_degree": 1, "tp_degree": 8, "pp_degree": 1, "ep_degree": 8},
        {"dp_degree": 1, "tp_degree": 4, "pp_degree": 2, "ep_degree": 8},
        {"dp_degree": 2, "tp_degree": 4, "pp_degree": 1, "ep_degree": 4},
        {"dp_degree": 4, "tp_degree": 2, "pp_degree": 1, "ep_degree": 2},
        {"dp_degree": 8, "tp_degree": 1, "pp_degree": 1, "ep_degree": 1},
    ]
    
    print("\n单配置预测测试:")
    print("-" * 80)
    
    for cfg in test_configs:
        parallel = ParallelConfig(
            dp_degree=cfg["dp_degree"],
            tp_degree=cfg["tp_degree"],
            pp_degree=cfg["pp_degree"],
            ep_degree=cfg["ep_degree"],
        )
        
        result = cm.predict_full(parallel, micro_batch_size=1, sequence_length=2048)
        
        config_str = f"DP{cfg['dp_degree']}-TP{cfg['tp_degree']}-PP{cfg['pp_degree']}-EP{cfg['ep_degree']}"
        fits = "✅" if result.fits_memory else "❌"
        
        print(f"{config_str:<20} 时延: {result.total_step_time_ms:>8.2f}ms  "
              f"显存: {result.memory_breakdown.total_memory_gb:>6.1f}GB {fits}  "
              f"MFU: {result.mfu:>5.1%}")
    
    print("-" * 80)
    
    # 配置排序测试
    print("\n配置排序测试:")
    cm.rank_configurations(test_configs, top_k=5)


if __name__ == "__main__":
    test_galvatron_costmodel()