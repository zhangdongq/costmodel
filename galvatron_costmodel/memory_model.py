#!/usr/bin/env python3
"""
显存模型模块 - 参考 Galvatron 的精确显存建模

核心功能：
1. 参数显存估算（考虑 TP/PP/EP 切分）
2. 梯度显存估算（考虑 ZeRO 分片）
3. 优化器状态显存估算（AdamW = 2 × fp32）
4. 激活值显存估算（考虑 Activation Checkpointing）
5. 通信缓冲区估算
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .computation_model import ModelConfig


class ZeROStage(Enum):
    """ZeRO 优化阶段"""
    NONE = 0        # 无 ZeRO
    STAGE_1 = 1     # 优化器状态分片
    STAGE_2 = 2     # 优化器状态 + 梯度分片
    STAGE_3 = 3     # 优化器状态 + 梯度 + 参数分片


class CheckpointGranularity(Enum):
    """Activation Checkpointing 粒度"""
    NONE = "none"           # 不使用
    SELECTIVE = "selective" # 选择性（只 checkpoint attention）
    FULL = "full"           # 全量（每层都 checkpoint）
    BLOCK = "block"         # 块状（每 N 层 checkpoint）


class OffloadTarget(Enum):
    """Offload 目标"""
    NONE = "none"
    CPU = "cpu"
    NVME = "nvme"


@dataclass
class ZeROConfig:
    """ZeRO 配置"""
    stage: ZeROStage = ZeROStage.NONE
    
    # ZeRO-3 特殊配置
    partition_activations: bool = False
    cpu_offload: bool = False
    
    def get_param_sharding_factor(self, dp_degree: int) -> float:
        """获取参数分片因子"""
        if self.stage == ZeROStage.STAGE_3:
            return dp_degree
        return 1
    
    def get_grad_sharding_factor(self, dp_degree: int) -> float:
        """获取梯度分片因子"""
        if self.stage in [ZeROStage.STAGE_2, ZeROStage.STAGE_3]:
            return dp_degree
        return 1
    
    def get_optimizer_sharding_factor(self, dp_degree: int) -> float:
        """获取优化器状态分片因子"""
        if self.stage in [ZeROStage.STAGE_1, ZeROStage.STAGE_2, ZeROStage.STAGE_3]:
            return dp_degree
        return 1


@dataclass
class ActivationCheckpointConfig:
    """Activation Checkpointing 配置"""
    granularity: CheckpointGranularity = CheckpointGranularity.NONE
    checkpoint_num_layers: int = 1  # 块状 checkpoint 的层数间隔
    
    def get_memory_reduction_factor(self) -> float:
        """获取显存降低因子"""
        if self.granularity == CheckpointGranularity.NONE:
            return 1.0
        elif self.granularity == CheckpointGranularity.SELECTIVE:
            return 0.7  # 大约降低 30%
        elif self.granularity == CheckpointGranularity.FULL:
            return 0.35  # 大约降低 65%
        elif self.granularity == CheckpointGranularity.BLOCK:
            # 根据 checkpoint_num_layers 计算
            return 1.0 / self.checkpoint_num_layers
        return 1.0
    
    def get_recompute_overhead(self) -> float:
        """获取重计算开销（时间倍数）"""
        if self.granularity == CheckpointGranularity.NONE:
            return 1.0
        elif self.granularity == CheckpointGranularity.SELECTIVE:
            return 1.15  # 约 15% 额外计算
        elif self.granularity == CheckpointGranularity.FULL:
            return 1.33  # 约 33% 额外计算（前向重算一次）
        elif self.granularity == CheckpointGranularity.BLOCK:
            return 1.0 + 0.33 / self.checkpoint_num_layers
        return 1.0


@dataclass
class OffloadConfig:
    """Offload 配置"""
    optimizer_offload: OffloadTarget = OffloadTarget.NONE
    activation_offload: OffloadTarget = OffloadTarget.NONE
    activation_offload_ratio: float = 0.0  # 0-1，offload 的激活比例
    
    def get_optimizer_memory_factor(self) -> float:
        """获取优化器显存因子"""
        if self.optimizer_offload != OffloadTarget.NONE:
            return 0.0  # 完全 offload 到 CPU/NVMe
        return 1.0
    
    def get_activation_memory_factor(self) -> float:
        """获取激活显存因子"""
        if self.activation_offload != OffloadTarget.NONE:
            return 1.0 - self.activation_offload_ratio
        return 1.0


@dataclass
class ParallelConfig:
    """并行配置"""
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    
    # Sequence Parallel
    sequence_parallel: bool = False
    
    @property
    def world_size(self) -> int:
        """总并行度"""
        return self.dp_degree * self.tp_degree * self.pp_degree


@dataclass
class TrainingConfig:
    """训练配置"""
    micro_batch_size: int = 1
    sequence_length: int = 8192
    global_batch_size: int = 512
    gradient_accumulation_steps: int = 64
    
    # 数据类型
    dtype: str = "bfloat16"
    dtype_bytes: int = 2
    
    # 优化器
    optimizer_type: str = "adamw"


@dataclass
class MemoryBreakdown:
    """显存分解结果"""
    # 主要组成
    parameter_memory_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    optimizer_memory_gb: float = 0.0
    activation_memory_gb: float = 0.0
    
    # 其他
    communication_buffer_gb: float = 0.0
    temporary_buffer_gb: float = 0.0
    reserved_memory_gb: float = 2.0  # CUDA 保留显存
    
    @property
    def total_memory_gb(self) -> float:
        """总显存占用"""
        return (
            self.parameter_memory_gb +
            self.gradient_memory_gb +
            self.optimizer_memory_gb +
            self.activation_memory_gb +
            self.communication_buffer_gb +
            self.temporary_buffer_gb +
            self.reserved_memory_gb
        )
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "parameter_memory_gb": self.parameter_memory_gb,
            "gradient_memory_gb": self.gradient_memory_gb,
            "optimizer_memory_gb": self.optimizer_memory_gb,
            "activation_memory_gb": self.activation_memory_gb,
            "communication_buffer_gb": self.communication_buffer_gb,
            "temporary_buffer_gb": self.temporary_buffer_gb,
            "reserved_memory_gb": self.reserved_memory_gb,
            "total_memory_gb": self.total_memory_gb,
        }


class MemoryModel:
    """
    显存模型 - 参考 Galvatron 的精确显存估算
    
    显存组成：
    1. 参数 (Parameters)
    2. 梯度 (Gradients)
    3. 优化器状态 (Optimizer States)
    4. 激活值 (Activations)
    5. 通信缓冲区 (Communication Buffers)
    6. 临时缓冲区 (Temporary Buffers)
    """
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model = model_config
        self.training = training_config
    
    def estimate_parameter_count(self, parallel: ParallelConfig) -> Dict[str, int]:
        """
        估算参数数量
        
        返回每个 GPU 上的参数数量（考虑并行切分）
        """
        h = self.model.hidden_size
        ffn = self.model.intermediate_size
        v = self.model.vocab_size
        num_layers = self.model.num_layers
        pp = parallel.pp_degree
        tp = parallel.tp_degree
        ep = parallel.ep_degree
        
        # 每个 PP stage 的层数
        layers_per_stage = num_layers // pp
        dense_layers = self.model.num_dense_layers // pp
        moe_layers = self.model.num_moe_layers // pp
        
        # ========== Attention 参数 ==========
        # Q, K, V, O projections: 4 * h * h
        # 被 TP 切分
        attention_params = 4 * h * h // tp
        
        # ========== LayerNorm 参数 ==========
        # 每层 2 个 LayerNorm: 2 * 2 * h = 4h
        # 不切分
        layernorm_params = 4 * h
        
        # ========== Dense MLP 参数 ==========
        # Gate + Up + Down: 3 * h * ffn (SwiGLU)
        # 被 TP 切分
        dense_mlp_params = 3 * h * ffn // tp
        
        # ========== MoE 参数 ==========
        # Router: h * num_experts
        router_params = h * self.model.num_experts
        
        # Expert MLP: 3 * h * ffn * (num_experts / ep)
        # 被 EP 和 TP 切分
        experts_per_gpu = self.model.num_experts // ep
        expert_params = 3 * h * ffn * experts_per_gpu // tp
        
        # ========== Embedding 参数 ==========
        # 词嵌入：v * h，被 TP 切分，只在第一个 PP stage
        # 输出投影：v * h，被 TP 切分，只在最后一个 PP stage
        embedding_params = v * h // tp if pp == 1 else v * h // tp // 2
        
        # ========== 汇总 ==========
        # Dense 层参数
        dense_layer_params = attention_params + layernorm_params + dense_mlp_params
        total_dense_params = dense_layer_params * dense_layers
        
        # MoE 层参数
        moe_layer_params = attention_params + layernorm_params + router_params + expert_params
        total_moe_params = moe_layer_params * moe_layers
        
        total_params = total_dense_params + total_moe_params + embedding_params
        
        return {
            "attention_params": attention_params * layers_per_stage,
            "layernorm_params": layernorm_params * layers_per_stage,
            "dense_mlp_params": dense_mlp_params * dense_layers,
            "router_params": router_params * moe_layers,
            "expert_params": expert_params * moe_layers,
            "embedding_params": embedding_params,
            "total_params": total_params,
        }
    
    def estimate_parameter_memory(self, parallel: ParallelConfig,
                                  zero_config: ZeROConfig) -> float:
        """估算参数显存 (GB)"""
        param_count = self.estimate_parameter_count(parallel)
        total_params = param_count["total_params"]
        
        # ZeRO-3 参数分片
        sharding_factor = zero_config.get_param_sharding_factor(parallel.dp_degree)
        
        # 参数显存 = 参数数量 × 数据类型字节数 / 分片因子
        param_bytes = total_params * self.training.dtype_bytes / sharding_factor
        
        return param_bytes / (1024 ** 3)
    
    def estimate_gradient_memory(self, parallel: ParallelConfig,
                                 zero_config: ZeROConfig) -> float:
        """估算梯度显存 (GB)"""
        param_count = self.estimate_parameter_count(parallel)
        total_params = param_count["total_params"]
        
        # ZeRO-2/3 梯度分片
        sharding_factor = zero_config.get_grad_sharding_factor(parallel.dp_degree)
        
        # 梯度显存 = 参数数量 × 数据类型字节数 / 分片因子
        grad_bytes = total_params * self.training.dtype_bytes / sharding_factor
        
        return grad_bytes / (1024 ** 3)
    
    def estimate_optimizer_memory(self, parallel: ParallelConfig,
                                  zero_config: ZeROConfig,
                                  offload_config: OffloadConfig) -> float:
        """
        估算优化器状态显存 (GB)
        
        AdamW: 2 × fp32 states (momentum + variance)
        """
        param_count = self.estimate_parameter_count(parallel)
        total_params = param_count["total_params"]
        
        # ZeRO-1/2/3 优化器状态分片
        sharding_factor = zero_config.get_optimizer_sharding_factor(parallel.dp_degree)
        
        # Offload 因子
        offload_factor = offload_config.get_optimizer_memory_factor()
        
        # AdamW: 2 个 fp32 状态
        optimizer_bytes = total_params * 4 * 2 / sharding_factor * offload_factor
        
        return optimizer_bytes / (1024 ** 3)
    
    def estimate_activation_memory(self, parallel: ParallelConfig,
                                   checkpoint_config: ActivationCheckpointConfig,
                                   offload_config: OffloadConfig) -> float:
        """
        估算激活值显存 (GB)
        
        激活值 = f(batch_size, seq_len, hidden_size, num_layers)
        """
        micro_bsz = self.training.micro_batch_size
        seq_len = self.training.sequence_length
        h = self.model.hidden_size
        num_layers = self.model.num_layers
        
        pp = parallel.pp_degree
        tp = parallel.tp_degree
        
        # 每个 PP stage 的层数
        layers_per_stage = num_layers // pp
        
        # ========== 每层激活大小估算 ==========
        # Attention 激活：
        # - Q, K, V: 3 * bsz * seq * h
        # - Attention scores: bsz * num_heads * seq * seq
        # - Output: bsz * seq * h
        num_heads = self.model.num_attention_heads
        attention_activation = (
            3 * micro_bsz * seq_len * h +  # Q, K, V
            micro_bsz * num_heads * seq_len * seq_len +  # attention scores
            micro_bsz * seq_len * h  # output
        )
        
        # MLP 激活：
        # - Gate: bsz * seq * ffn
        # - Up: bsz * seq * ffn
        # - Down input: bsz * seq * ffn
        ffn = self.model.intermediate_size
        mlp_activation = 3 * micro_bsz * seq_len * ffn
        
        # MoE 层额外激活：
        # - Router logits: bsz * seq * num_experts
        # - Expert outputs: bsz * seq * h * topk
        moe_extra_activation = (
            micro_bsz * seq_len * self.model.num_experts +
            micro_bsz * seq_len * h * self.model.moe_top_k
        )
        
        # TP 切分激活
        if tp > 1:
            attention_activation = attention_activation // tp
            mlp_activation = mlp_activation // tp
        
        # Sequence Parallel 进一步切分
        if parallel.sequence_parallel and tp > 1:
            # 某些激活在序列维度切分
            pass  # 简化处理
        
        # ========== 总激活显存 ==========
        dense_layers_per_stage = self.model.num_dense_layers // pp
        moe_layers_per_stage = self.model.num_moe_layers // pp
        
        dense_activation = (attention_activation + mlp_activation) * dense_layers_per_stage
        moe_activation = (attention_activation + mlp_activation + moe_extra_activation) * moe_layers_per_stage
        
        total_activation = dense_activation + moe_activation
        
        # 数据类型转换
        activation_bytes = total_activation * self.training.dtype_bytes
        
        # Checkpoint 降低因子
        checkpoint_factor = checkpoint_config.get_memory_reduction_factor()
        
        # Offload 因子
        offload_factor = offload_config.get_activation_memory_factor()
        
        activation_bytes = activation_bytes * checkpoint_factor * offload_factor
        
        return activation_bytes / (1024 ** 3)
    
    def estimate_communication_buffer(self, parallel: ParallelConfig) -> float:
        """
        估算通信缓冲区显存 (GB)
        
        包括：
        - TP AllReduce buffer
        - PP Send/Recv buffer
        - DP AllReduce/ReduceScatter buffer
        - SP AllGather buffer
        """
        micro_bsz = self.training.micro_batch_size
        seq_len = self.training.sequence_length
        h = self.model.hidden_size
        
        buffer_bytes = 0
        
        # TP 通信缓冲区
        if parallel.tp_degree > 1:
            tp_buffer = micro_bsz * seq_len * h * self.training.dtype_bytes
            buffer_bytes += tp_buffer * 2  # 前向 + 后向
        
        # PP 通信缓冲区
        if parallel.pp_degree > 1:
            pp_buffer = micro_bsz * seq_len * h * self.training.dtype_bytes
            buffer_bytes += pp_buffer * 2  # send + recv
        
        # SP AllGather 缓冲区（如果启用）
        if parallel.sequence_parallel and parallel.tp_degree > 1:
            sp_buffer = micro_bsz * seq_len * h * self.training.dtype_bytes
            buffer_bytes += sp_buffer
        
        return buffer_bytes / (1024 ** 3)
    
    def estimate_memory(self, parallel: ParallelConfig,
                        zero_config: ZeROConfig = None,
                        checkpoint_config: ActivationCheckpointConfig = None,
                        offload_config: OffloadConfig = None) -> MemoryBreakdown:
        """
        完整显存估算
        
        Returns:
            MemoryBreakdown 详细分解
        """
        if zero_config is None:
            zero_config = ZeROConfig()
        if checkpoint_config is None:
            checkpoint_config = ActivationCheckpointConfig()
        if offload_config is None:
            offload_config = OffloadConfig()
        
        breakdown = MemoryBreakdown()
        
        breakdown.parameter_memory_gb = self.estimate_parameter_memory(parallel, zero_config)
        breakdown.gradient_memory_gb = self.estimate_gradient_memory(parallel, zero_config)
        breakdown.optimizer_memory_gb = self.estimate_optimizer_memory(parallel, zero_config, offload_config)
        breakdown.activation_memory_gb = self.estimate_activation_memory(parallel, checkpoint_config, offload_config)
        breakdown.communication_buffer_gb = self.estimate_communication_buffer(parallel)
        
        # 临时缓冲区（粗略估计为激活的 10%）
        breakdown.temporary_buffer_gb = breakdown.activation_memory_gb * 0.1
        
        return breakdown
    
    def fits_memory(self, parallel: ParallelConfig,
                    gpu_memory_gb: float,
                    zero_config: ZeROConfig = None,
                    checkpoint_config: ActivationCheckpointConfig = None,
                    offload_config: OffloadConfig = None) -> Tuple[bool, MemoryBreakdown]:
        """
        检查是否能放入 GPU 显存
        
        Returns:
            (fits, breakdown)
        """
        breakdown = self.estimate_memory(parallel, zero_config, checkpoint_config, offload_config)
        fits = breakdown.total_memory_gb <= gpu_memory_gb
        
        return fits, breakdown


class AutoMemoryOptimizer:
    """
    自动显存优化器
    
    自动选择最优的 ZeRO stage、Checkpoint 策略和 Offload 策略
    """
    
    def __init__(self, memory_model: MemoryModel, gpu_memory_gb: float):
        self.memory_model = memory_model
        self.gpu_memory_gb = gpu_memory_gb
    
    def find_optimal_config(self, parallel: ParallelConfig) -> Tuple[
        ZeROConfig, ActivationCheckpointConfig, OffloadConfig, MemoryBreakdown
    ]:
        """
        搜索最优的显存优化配置
        
        优先级：最小化计算开销
        1. 先尝试不使用任何优化
        2. 再尝试 ZeRO stages
        3. 再尝试 Activation Checkpointing
        4. 最后尝试 Offload
        """
        # 优化策略组合（按计算开销从小到大排序）
        strategies = [
            # (ZeRO stage, Checkpoint granularity, Offload)
            (ZeROStage.NONE, CheckpointGranularity.NONE, False),
            (ZeROStage.STAGE_1, CheckpointGranularity.NONE, False),
            (ZeROStage.STAGE_2, CheckpointGranularity.NONE, False),
            (ZeROStage.STAGE_1, CheckpointGranularity.SELECTIVE, False),
            (ZeROStage.STAGE_2, CheckpointGranularity.SELECTIVE, False),
            (ZeROStage.STAGE_1, CheckpointGranularity.FULL, False),
            (ZeROStage.STAGE_2, CheckpointGranularity.FULL, False),
            (ZeROStage.STAGE_3, CheckpointGranularity.FULL, False),
            (ZeROStage.STAGE_3, CheckpointGranularity.FULL, True),  # with offload
        ]
        
        for zero_stage, ckpt_granularity, use_offload in strategies:
            zero_config = ZeROConfig(stage=zero_stage)
            checkpoint_config = ActivationCheckpointConfig(granularity=ckpt_granularity)
            offload_config = OffloadConfig(
                optimizer_offload=OffloadTarget.CPU if use_offload else OffloadTarget.NONE
            )
            
            fits, breakdown = self.memory_model.fits_memory(
                parallel, self.gpu_memory_gb,
                zero_config, checkpoint_config, offload_config
            )
            
            if fits:
                return zero_config, checkpoint_config, offload_config, breakdown
        
        # 如果都不满足，返回最激进的配置
        return (
            ZeROConfig(stage=ZeROStage.STAGE_3, cpu_offload=True),
            ActivationCheckpointConfig(granularity=CheckpointGranularity.FULL),
            OffloadConfig(
                optimizer_offload=OffloadTarget.CPU,
                activation_offload=OffloadTarget.CPU,
                activation_offload_ratio=0.5
            ),
            breakdown
        )