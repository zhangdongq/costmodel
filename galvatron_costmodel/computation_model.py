#!/usr/bin/env python3
"""
计算模型模块 - 参考 Galvatron 的 α-β 模型

核心功能：
1. 分层计算时间建模（Attention、MLP、MoE）
2. α-β 模型拟合（线性/曲线模式）
3. 流水线气泡时间预测
4. 支持通过 Profiling 数据校准
"""

import math
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .hardware_config import HardwareConfig, GPUSpecs


class ComputeMode(Enum):
    """计算建模模式"""
    STATIC = "static"    # 固定效率
    LINEAR = "linear"    # 线性拟合 (T = α × batch_size)
    CURVE = "curve"      # α-β 曲线拟合 (T = α × batch_size + β)


class LayerType(Enum):
    """层类型"""
    ATTENTION = "attention"
    MLP = "mlp"
    MOE_ROUTER = "moe_router"
    MOE_EXPERT = "moe_expert"
    LAYERNORM = "layernorm"
    EMBEDDING = "embedding"


@dataclass
class ModelConfig:
    """模型架构配置"""
    # 基本参数
    num_layers: int = 48
    hidden_size: int = 6144
    intermediate_size: int = 16384
    num_attention_heads: int = 32
    num_key_value_heads: int = 4  # GQA
    head_dim: int = 192
    
    # MoE 参数
    num_experts: int = 128
    moe_top_k: int = 8
    num_moe_layers: int = 24  # MoE 层数（Qwen3-30B 一半是 MoE）
    
    # 其他参数
    vocab_size: int = 152064
    max_position_embeddings: int = 32768
    
    @property
    def num_dense_layers(self) -> int:
        """Dense 层数"""
        return self.num_layers - self.num_moe_layers


@dataclass
class AlphaBetaParams:
    """α-β 模型参数"""
    alpha: float = 0.0  # 斜率（时间/batch_size）
    beta: float = 0.0   # 截距（固定开销）
    
    def predict(self, batch_size: int) -> float:
        """预测计算时间"""
        return self.alpha * batch_size + self.beta


@dataclass
class LayerComputeProfile:
    """层计算 Profile"""
    layer_type: LayerType
    
    # FLOPs 估算参数
    flops_per_token: int = 0
    
    # α-β 模型参数（不同序列长度）
    alpha_beta_params: Dict[int, AlphaBetaParams] = field(default_factory=dict)
    
    # 静态效率（备用）
    static_efficiency: float = 0.5
    
    def get_alpha_beta(self, seq_len: int) -> AlphaBetaParams:
        """获取对应序列长度的 α-β 参数"""
        if seq_len in self.alpha_beta_params:
            return self.alpha_beta_params[seq_len]
        
        # 找最接近的
        if self.alpha_beta_params:
            closest_seq = min(self.alpha_beta_params.keys(), 
                            key=lambda x: abs(x - seq_len))
            return self.alpha_beta_params[closest_seq]
        
        # 默认参数
        return AlphaBetaParams(alpha=0.001, beta=0.01)


class AlphaBetaModel:
    """
    α-β 模型 - 参考 Galvatron 的计算时间建模
    
    核心思想：
    - 计算时间 = α × batch_size + β
    - α: 与 batch_size 成比例的部分（主要是计算）
    - β: 固定开销（kernel launch、内存分配等）
    
    通过多点实测数据拟合参数
    """
    
    def __init__(self):
        self.params: Dict[str, AlphaBetaParams] = {}
    
    def fit(self, batch_sizes: List[int], times_ms: List[float], 
            key: str = "default") -> AlphaBetaParams:
        """
        拟合 α-β 参数
        
        Args:
            batch_sizes: batch size 列表
            times_ms: 对应的计算时间列表 (ms)
            key: 参数标识键
        """
        if len(batch_sizes) < 2:
            # 数据点不足，使用默认参数
            params = AlphaBetaParams(
                alpha=times_ms[0] / batch_sizes[0] if batch_sizes else 0.001,
                beta=0.01
            )
            self.params[key] = params
            return params
        
        # 线性回归拟合
        n = len(batch_sizes)
        sum_x = sum(batch_sizes)
        sum_y = sum(times_ms)
        sum_xy = sum(x * y for x, y in zip(batch_sizes, times_ms))
        sum_xx = sum(x * x for x in batch_sizes)
        
        # α = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        # β = (Σy - α*Σx) / n
        denominator = n * sum_xx - sum_x * sum_x
        
        if abs(denominator) < 1e-10:
            # 数值问题，使用简单平均
            alpha = sum_y / sum_x if sum_x > 0 else 0.001
            beta = 0.01
        else:
            alpha = (n * sum_xy - sum_x * sum_y) / denominator
            beta = (sum_y - alpha * sum_x) / n
        
        # 确保参数非负
        alpha = max(0.0, alpha)
        beta = max(0.0, beta)
        
        params = AlphaBetaParams(alpha=alpha, beta=beta)
        self.params[key] = params
        
        return params
    
    def predict(self, batch_size: int, key: str = "default") -> float:
        """预测计算时间"""
        if key not in self.params:
            return batch_size * 0.001 + 0.01  # 默认值
        
        return self.params[key].predict(batch_size)


class ComputationModel:
    """
    计算模型 - 整合 Galvatron 的计算时间预测
    
    支持三种模式：
    1. static: 使用固定效率估算
    2. linear: 时间与 batch_size 成正比
    3. curve: 使用 α-β 模型拟合
    """
    
    def __init__(self, hardware: HardwareConfig, model_config: ModelConfig,
                 mode: ComputeMode = ComputeMode.LINEAR):
        self.hardware = hardware
        self.model_config = model_config
        self.mode = mode
        
        # α-β 模型
        self.alpha_beta_model = AlphaBetaModel()
        
        # 层 Profile
        self.layer_profiles: Dict[LayerType, LayerComputeProfile] = {}
        self._init_layer_profiles()
        
        # Profiling 数据
        self.profiled_data: Dict = {}
    
    def _init_layer_profiles(self):
        """初始化层 Profile"""
        h = self.model_config.hidden_size
        ffn = self.model_config.intermediate_size
        num_heads = self.model_config.num_attention_heads
        head_dim = self.model_config.head_dim
        num_experts = self.model_config.num_experts
        
        # Attention FLOPs: 4 * h^2 (Q,K,V,O projections) + 2 * seq^2 * h (attention scores)
        self.layer_profiles[LayerType.ATTENTION] = LayerComputeProfile(
            layer_type=LayerType.ATTENTION,
            flops_per_token=4 * h * h,  # 简化，不含 attention score
            static_efficiency=0.45,
        )
        
        # MLP FLOPs: 8 * h * ffn (up + gate + down projections with SwiGLU)
        self.layer_profiles[LayerType.MLP] = LayerComputeProfile(
            layer_type=LayerType.MLP,
            flops_per_token=8 * h * ffn,
            static_efficiency=0.55,
        )
        
        # MoE Router FLOPs: 2 * h * num_experts
        self.layer_profiles[LayerType.MOE_ROUTER] = LayerComputeProfile(
            layer_type=LayerType.MOE_ROUTER,
            flops_per_token=2 * h * num_experts,
            static_efficiency=0.30,
        )
        
        # MoE Expert FLOPs: 8 * h * ffn (与 MLP 相同)
        self.layer_profiles[LayerType.MOE_EXPERT] = LayerComputeProfile(
            layer_type=LayerType.MOE_EXPERT,
            flops_per_token=8 * h * ffn,
            static_efficiency=0.45,
        )
        
        # LayerNorm FLOPs: 4 * h
        self.layer_profiles[LayerType.LAYERNORM] = LayerComputeProfile(
            layer_type=LayerType.LAYERNORM,
            flops_per_token=4 * h,
            static_efficiency=0.20,
        )
    
    def _estimate_layer_time_static(self, layer_type: LayerType,
                                    tokens: int, tp_degree: int) -> float:
        """使用静态效率估算层计算时间"""
        profile = self.layer_profiles.get(layer_type)
        if profile is None:
            return 0.0
        
        # 总 FLOPs
        total_flops = profile.flops_per_token * tokens * 2  # FMA = 2 ops
        
        # TP 并行切分
        parallel_flops = total_flops / tp_degree
        
        # 效率
        efficiency = profile.static_efficiency
        
        # 峰值算力 (TFLOPS)
        peak_tflops = self.hardware.gpu.get_effective_tflops("bf16")
        
        # 计算时间 (ms)
        time_sec = parallel_flops / (peak_tflops * 1e12 * efficiency)
        
        return time_sec * 1000
    
    def _estimate_layer_time_linear(self, layer_type: LayerType,
                                    batch_size: int, seq_len: int,
                                    tp_degree: int) -> float:
        """使用线性模式估算层计算时间"""
        # 基于 tokens 数量线性增长
        tokens = batch_size * seq_len
        
        # 使用静态估算作为基准，然后线性缩放
        base_tokens = 1024  # 基准 tokens
        base_time = self._estimate_layer_time_static(layer_type, base_tokens, tp_degree)
        
        return base_time * (tokens / base_tokens)
    
    def _estimate_layer_time_curve(self, layer_type: LayerType,
                                   batch_size: int, seq_len: int,
                                   tp_degree: int) -> float:
        """使用 α-β 曲线模式估算层计算时间"""
        key = f"{layer_type.value}_seq{seq_len}_tp{tp_degree}"
        
        if key in self.alpha_beta_model.params:
            return self.alpha_beta_model.predict(batch_size, key)
        
        # 回退到线性模式
        return self._estimate_layer_time_linear(layer_type, batch_size, seq_len, tp_degree)
    
    def estimate_layer_time(self, layer_type: LayerType,
                           batch_size: int, seq_len: int,
                           tp_degree: int = 1) -> float:
        """
        估算单层计算时间
        
        Args:
            layer_type: 层类型
            batch_size: micro batch size
            seq_len: 序列长度
            tp_degree: TP 并行度
        
        Returns:
            计算时间 (ms)
        """
        if self.mode == ComputeMode.STATIC:
            tokens = batch_size * seq_len
            return self._estimate_layer_time_static(layer_type, tokens, tp_degree)
        elif self.mode == ComputeMode.LINEAR:
            return self._estimate_layer_time_linear(layer_type, batch_size, seq_len, tp_degree)
        else:  # CURVE
            return self._estimate_layer_time_curve(layer_type, batch_size, seq_len, tp_degree)
    
    def estimate_attention_time(self, batch_size: int, seq_len: int,
                                tp_degree: int = 1) -> float:
        """估算 Attention 计算时间"""
        return self.estimate_layer_time(LayerType.ATTENTION, batch_size, seq_len, tp_degree)
    
    def estimate_mlp_time(self, batch_size: int, seq_len: int,
                          tp_degree: int = 1) -> float:
        """估算 MLP 计算时间"""
        return self.estimate_layer_time(LayerType.MLP, batch_size, seq_len, tp_degree)
    
    def estimate_moe_time(self, batch_size: int, seq_len: int,
                          tp_degree: int = 1, ep_degree: int = 1) -> float:
        """估算 MoE 层计算时间（Router + Expert）"""
        router_time = self.estimate_layer_time(
            LayerType.MOE_ROUTER, batch_size, seq_len, tp_degree
        )
        
        # Expert 计算被 EP 切分，且只处理 TopK 个专家
        expert_time = self.estimate_layer_time(
            LayerType.MOE_EXPERT, batch_size, seq_len, tp_degree
        )
        # 考虑 EP 并行和 TopK 选择
        topk = self.model_config.moe_top_k
        experts_per_gpu = self.model_config.num_experts / max(ep_degree, 1)
        workload_factor = min(topk / ep_degree, 1.0)
        expert_time = expert_time * workload_factor / max(ep_degree, 1)
        
        return router_time + expert_time
    
    def estimate_dense_layer_time(self, batch_size: int, seq_len: int,
                                  tp_degree: int = 1) -> float:
        """估算单个 Dense 层（Attention + MLP）的计算时间"""
        attn_time = self.estimate_attention_time(batch_size, seq_len, tp_degree)
        mlp_time = self.estimate_mlp_time(batch_size, seq_len, tp_degree)
        
        return attn_time + mlp_time
    
    def estimate_moe_layer_time(self, batch_size: int, seq_len: int,
                                tp_degree: int = 1, ep_degree: int = 1) -> float:
        """估算单个 MoE 层（Attention + MoE）的计算时间"""
        attn_time = self.estimate_attention_time(batch_size, seq_len, tp_degree)
        moe_time = self.estimate_moe_time(batch_size, seq_len, tp_degree, ep_degree)
        
        return attn_time + moe_time
    
    def estimate_forward_time(self, batch_size: int, seq_len: int,
                              tp_degree: int = 1, pp_degree: int = 1,
                              ep_degree: int = 1) -> float:
        """
        估算完整前向传播时间
        
        考虑 PP 切分，每个 stage 只计算部分层
        """
        layers_per_stage = self.model_config.num_layers // pp_degree
        dense_layers_per_stage = self.model_config.num_dense_layers // pp_degree
        moe_layers_per_stage = self.model_config.num_moe_layers // pp_degree
        
        # Dense 层时间
        dense_time = self.estimate_dense_layer_time(batch_size, seq_len, tp_degree)
        total_dense_time = dense_time * dense_layers_per_stage
        
        # MoE 层时间
        moe_time = self.estimate_moe_layer_time(batch_size, seq_len, tp_degree, ep_degree)
        total_moe_time = moe_time * moe_layers_per_stage
        
        return total_dense_time + total_moe_time
    
    def estimate_backward_time(self, batch_size: int, seq_len: int,
                               tp_degree: int = 1, pp_degree: int = 1,
                               ep_degree: int = 1) -> float:
        """
        估算完整反向传播时间
        
        通常约为前向的 2 倍
        """
        forward_time = self.estimate_forward_time(
            batch_size, seq_len, tp_degree, pp_degree, ep_degree
        )
        
        # 反向传播约为前向的 2 倍
        return forward_time * 2.0
    
    def estimate_pipeline_bubble_time(self, forward_time: float, backward_time: float,
                                      pp_degree: int, num_micro_batches: int) -> float:
        """
        估算流水线气泡时间
        
        1F1B 调度：bubble_ratio = (pp_degree - 1) / num_micro_batches
        """
        if pp_degree <= 1:
            return 0.0
        
        bubble_ratio = (pp_degree - 1) / num_micro_batches
        single_stage_time = forward_time + backward_time
        
        return single_stage_time * bubble_ratio
    
    def load_profile(self, profile_path: str):
        """
        加载 Profiling 数据并拟合 α-β 模型
        
        Profile 数据格式：
        {
            "attention": {
                "seq_1024": [
                    {"batch_size": 1, "time_ms": 0.5},
                    {"batch_size": 2, "time_ms": 0.9},
                    ...
                ]
            },
            ...
        }
        """
        if not os.path.exists(profile_path):
            print(f"Warning: Profile file not found: {profile_path}")
            return
        
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        self.profiled_data = data
        
        # 为每种层类型拟合 α-β 参数
        for layer_name, seq_data in data.items():
            try:
                layer_type = LayerType(layer_name)
            except ValueError:
                continue
            
            for seq_key, measurements in seq_data.items():
                if not measurements:
                    continue
                
                batch_sizes = [m["batch_size"] for m in measurements]
                times = [m["time_ms"] for m in measurements]
                
                # 提取序列长度
                seq_len = int(seq_key.replace("seq_", ""))
                
                # 拟合参数
                key = f"{layer_name}_seq{seq_len}_tp1"
                self.alpha_beta_model.fit(batch_sizes, times, key)
        
        # 切换到 curve 模式
        self.mode = ComputeMode.CURVE
        print(f"Loaded profile from {profile_path}, switched to CURVE mode")
    
    def save_profile(self, profile_path: str):
        """保存 Profiling 数据"""
        os.makedirs(os.path.dirname(profile_path) if os.path.dirname(profile_path) else ".", exist_ok=True)
        with open(profile_path, 'w') as f:
            json.dump(self.profiled_data, f, indent=2)