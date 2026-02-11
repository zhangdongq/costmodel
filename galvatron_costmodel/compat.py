#!/usr/bin/env python3
"""
兼容性适配器 - 保持与旧 API 的兼容性

将新的 Galvatron CostModel 接口适配到旧的 costmodel_integrated.py 接口
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .integrated_costmodel import (
    GalvatronCostModel, CostModelConfig, PredictionResult,
    create_qwen3_30b_costmodel
)
from .memory_model import (
    ParallelConfig, TrainingConfig, MemoryBreakdown,
    ZeROConfig, ActivationCheckpointConfig, OffloadConfig,
    ZeROStage, CheckpointGranularity, OffloadTarget
)
from .hardware_config import HardwareConfig, GPUSpecs, ClusterConfig
from .computation_model import ModelConfig


# ==================== 旧接口兼容类 ====================

@dataclass
class LegacyHardwareConfig:
    """旧版硬件配置（兼容）"""
    num_gpus: int = 8
    gpu_memory_gb: float = 80.0
    num_machines: int = 1
    gpus_per_machine: int = 8
    network_bandwidth_gbps: float = 100.0


@dataclass
class LegacyModelConfig:
    """旧版模型配置（兼容）"""
    num_layers: int = 48
    hidden_size: int = 6144
    intermediate_size: int = 16384
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    num_experts: int = 128
    max_seq_len: int = 8192
    vocab_size: int = 152064


@dataclass
class LegacyParallelConfig:
    """旧版并行配置（兼容）"""
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = 1
    sharding_degree: int = 1
    sequence_parallel: bool = False
    gradient_accumulation_steps: int = 64
    micro_batch_size: int = 1
    data_type: str = "bfloat16"


@dataclass
class LegacyCostModelConfig:
    """旧版 CostModel 配置（兼容）"""
    model_config: LegacyModelConfig
    hardware_config: LegacyHardwareConfig
    training_config: Dict


class IntegratedCostModel:
    """
    兼容旧版 IntegratedCostModel 的适配器
    
    内部使用新的 GalvatronCostModel
    """
    
    def __init__(self, config: LegacyCostModelConfig):
        self.legacy_config = config
        
        # 转换为新配置
        new_config = self._convert_config(config)
        self.costmodel = GalvatronCostModel(new_config)
    
    def _convert_config(self, legacy: LegacyCostModelConfig) -> CostModelConfig:
        """将旧配置转换为新配置"""
        model = legacy.model_config
        hw = legacy.hardware_config
        training = legacy.training_config
        
        return CostModelConfig(
            hardware=HardwareConfig(
                gpu=GPUSpecs(memory_gb=hw.gpu_memory_gb),
                cluster=ClusterConfig(
                    num_nodes=hw.num_machines,
                    gpus_per_node=hw.gpus_per_machine,
                ),
            ),
            model=ModelConfig(
                num_layers=model.num_layers,
                hidden_size=model.hidden_size,
                intermediate_size=model.intermediate_size,
                num_attention_heads=model.num_attention_heads,
                num_key_value_heads=model.num_key_value_heads,
                num_experts=model.num_experts,
                vocab_size=model.vocab_size,
                num_moe_layers=model.num_layers // 2,  # 假设一半是 MoE
            ),
            training=TrainingConfig(
                micro_batch_size=training.get("micro_batch_size", 1),
                sequence_length=training.get("sequence_length", 8192),
                gradient_accumulation_steps=training.get("gradient_accumulation_steps", 64),
            ),
        )
    
    def _convert_parallel_config(self, legacy: LegacyParallelConfig) -> ParallelConfig:
        """转换并行配置"""
        return ParallelConfig(
            dp_degree=legacy.dp_degree,
            tp_degree=legacy.tp_degree,
            pp_degree=legacy.pp_degree,
            ep_degree=legacy.ep_degree,
            sequence_parallel=legacy.sequence_parallel,
        )
    
    def predict_step_time(self, parallel_config: LegacyParallelConfig,
                          micro_batch_size: int,
                          sequence_length: int,
                          num_micro_batches: int = 64) -> Dict[str, float]:
        """预测 step 时延（兼容旧接口）"""
        parallel = self._convert_parallel_config(parallel_config)
        
        result = self.costmodel.predict_step_time(
            parallel, micro_batch_size, sequence_length, num_micro_batches
        )
        
        # 转换为旧格式
        return {
            "compute_time_ms": result["compute_time_ms"],
            "a2a_time_ms": result["ep_comm_time_ms"],
            "other_comm_time_ms": result["tp_comm_time_ms"] + result["dp_comm_time_ms"] + result["pp_comm_time_ms"],
            "total_step_time_ms": result["total_step_time_ms"],
            "bubble_ratio": result["bubble_ratio"],
            "utilization_estimate": 1.0 - result["bubble_ratio"],
        }
    
    def predict_memory(self, parallel_config: LegacyParallelConfig,
                       optimization=None) -> Dict:
        """预测显存（兼容旧接口）"""
        parallel = self._convert_parallel_config(parallel_config)
        
        breakdown = self.costmodel.predict_memory(parallel)
        
        return {
            "total_memory_gb": breakdown.total_memory_gb,
            "parameter_memory_gb": breakdown.parameter_memory_gb,
            "optimizer_memory_gb": breakdown.optimizer_memory_gb,
            "activation_memory_gb": breakdown.activation_memory_gb,
            "gradient_memory_gb": breakdown.gradient_memory_gb,
            "communication_buffer_gb": breakdown.communication_buffer_gb,
            "reserved_memory_gb": breakdown.reserved_memory_gb,
            "compute_overhead": 1.0,
            "fits_memory": breakdown.total_memory_gb <= self.legacy_config.hardware_config.gpu_memory_gb
        }
    
    def auto_select_optimization(self, parallel_config: LegacyParallelConfig):
        """自动选择优化策略（兼容旧接口）"""
        parallel = self._convert_parallel_config(parallel_config)
        
        result = self.costmodel.predict_full(parallel, auto_optimize=True)
        
        # 转换优化配置
        opt_config = {
            "recompute_granularity": result.checkpoint_config.granularity.value,
            "recompute_method": "uniform",
            "recompute_num_layers": result.checkpoint_config.checkpoint_num_layers,
            "offload_strategy": result.offload_config.optimizer_offload.value,
            "sharding": f"stage{result.zero_config.stage.value}",
            "split_param": True,
        }
        
        memory_info = {
            "total_memory_gb": result.memory_breakdown.total_memory_gb,
            "parameter_memory_gb": result.memory_breakdown.parameter_memory_gb,
            "optimizer_memory_gb": result.memory_breakdown.optimizer_memory_gb,
            "activation_memory_gb": result.memory_breakdown.activation_memory_gb,
            "gradient_memory_gb": result.memory_breakdown.gradient_memory_gb,
            "compute_overhead": result.checkpoint_config.get_recompute_overhead(),
            "fits_memory": result.fits_memory,
        }
        
        return opt_config, memory_info
    
    def predict_full(self, parallel_config: LegacyParallelConfig,
                     micro_batch_size: int,
                     sequence_length: int,
                     auto_optimize: bool = True) -> Dict:
        """完整预测（兼容旧接口）"""
        parallel = self._convert_parallel_config(parallel_config)
        
        result = self.costmodel.predict_full(
            parallel, micro_batch_size, sequence_length, auto_optimize
        )
        
        return {
            # 时延信息
            "total_step_time_ms": result.total_step_time_ms,
            "base_step_time_ms": result.compute_time_ms,
            "compute_time_ms": result.compute_time_ms,
            "a2a_time_ms": result.ep_comm_time_ms,
            "bubble_ratio": result.bubble_ratio,
            "utilization": result.compute_efficiency,
            
            # 显存信息
            "total_memory_gb": result.memory_breakdown.total_memory_gb,
            "parameter_memory_gb": result.memory_breakdown.parameter_memory_gb,
            "optimizer_memory_gb": result.memory_breakdown.optimizer_memory_gb,
            "activation_memory_gb": result.memory_breakdown.activation_memory_gb,
            "gradient_memory_gb": result.memory_breakdown.gradient_memory_gb,
            "fits_memory": result.fits_memory,
            
            # 优化策略
            "optimization_config": {
                "recompute_granularity": result.checkpoint_config.granularity.value,
                "offload_strategy": result.offload_config.optimizer_offload.value,
                "sharding": f"stage{result.zero_config.stage.value}",
            },
            "compute_overhead": result.checkpoint_config.get_recompute_overhead(),
        }
    
    def rank_configurations(self, config_list: List[Dict], top_k: int = 10) -> List[Dict]:
        """配置排序（兼容旧接口）"""
        return self.costmodel.rank_configurations(config_list, top_k)
    
    def validate_with_real_data(self, real_runs: List[Dict]) -> Dict:
        """验证准确性（兼容旧接口）"""
        if not real_runs:
            return {"sample_count": 0, "error": "No validation data"}
        
        import numpy as np
        errors = []
        
        for run in real_runs:
            try:
                parallel = ParallelConfig(
                    dp_degree=run.get("dp_degree", 1),
                    tp_degree=run.get("tp_degree", 1),
                    pp_degree=run.get("pp_degree", 1),
                    ep_degree=run.get("ep_degree", 1),
                )
                
                prediction = self.costmodel.predict_step_time(
                    parallel,
                    run.get("micro_batch_size", 1),
                    run.get("sequence_length", 2048)
                )
                
                actual_time = run.get("actual_step_time_ms", 0)
                predicted_time = prediction["total_step_time_ms"]
                
                if actual_time > 0:
                    error_pct = abs(predicted_time - actual_time) / actual_time * 100
                    errors.append(error_pct)
            except Exception as e:
                print(f"Warning: Failed to validate run {run}: {e}")
                continue
        
        if errors:
            return {
                "sample_count": len(errors),
                "avg_error_pct": np.mean(errors),
                "max_error_pct": np.max(errors),
                "std_error_pct": np.std(errors),
                "accuracy_good": np.mean(errors) < 15.0,
                "within_10_pct": len([e for e in errors if e <= 10]) / len(errors)
            }
        else:
            return {"sample_count": 0, "error": "No validation data"}


# ==================== 导出旧接口名称 ====================

# 为了完全兼容，导出旧的类名
CostModelConfig_Legacy = LegacyCostModelConfig
HardwareConfig_Legacy = LegacyHardwareConfig
ModelConfig_Legacy = LegacyModelConfig
ParallelConfig_Legacy = LegacyParallelConfig