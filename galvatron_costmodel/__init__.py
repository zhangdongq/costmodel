#!/usr/bin/env python3
"""
Galvatron CostModel - 参考 Hetu-Galvatron 的分布式训练成本模型

版本: 2.0.0

架构设计 (参考 Galvatron):
┌─────────────────────────────────────────────────────────────────┐
│                    GalvatronCostModel                           │
│    ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│    │ Computation│  │Communication│ │  Memory    │              │
│    │   Model    │  │    Model    │ │   Model    │              │
│    └─────┬──────┘  └──────┬─────┘ └─────┬──────┘              │
│          │               │              │                      │
│          └───────┬───────┴──────────────┘                      │
│                  │                                              │
│          ┌───────▼───────┐                                      │
│          │ HardwareConfig│                                      │
│          │ (GPU, Network,│                                      │
│          │   Cluster)    │                                      │
│          └───────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘

模块结构：
- hardware_config: 硬件配置（GPU、网络、集群）
- communication_model: 通信模型（AllReduce、AllToAll、P2P）
- computation_model: 计算模型（α-β 拟合）
- memory_model: 显存模型（精确分解）
- integrated_costmodel: 集成 CostModel
- compat: 兼容旧 API 的适配器

主要改进 (相比 v1):
1. 分层硬件感知: 区分节点内（NVLink）和节点间（IB）通信
2. α-β 模型拟合: 支持通过 profiling 数据校准
3. 精确显存分解: 参数、梯度、优化器、激活值独立建模
4. 自动优化策略: 自动选择 ZeRO、Checkpoint、Offload
5. MoE 专项支持: AllToAll 通信的分段拟合
"""

__version__ = "2.0.0"

from .hardware_config import (
    HardwareConfig,
    NetworkTopology,
    GPUSpecs,
    ClusterConfig,
    GPUType,
    NetworkType,
    profile_gpu_compute,
    profile_network_bandwidth,
)

from .communication_model import (
    CommunicationModel,
    AllReduceModel,
    AllGatherModel,
    ReduceScatterModel,
    AllToAllModel,
    P2PModel,
    CommType,
    CommResult,
    CommConfig,
)

from .computation_model import (
    ComputationModel,
    AlphaBetaModel,
    AlphaBetaParams,
    LayerComputeProfile,
    ModelConfig,
    ComputeMode,
    LayerType,
)

from .memory_model import (
    MemoryModel,
    MemoryBreakdown,
    ParallelConfig,
    TrainingConfig,
    ZeROConfig,
    ActivationCheckpointConfig,
    OffloadConfig,
    ZeROStage,
    CheckpointGranularity,
    OffloadTarget,
    AutoMemoryOptimizer,
    RecomputeMethod,
)

from .integrated_costmodel import (
    GalvatronCostModel,
    CostModelConfig,
    PredictionResult,
    create_qwen3_30b_costmodel,
)

from .compute_profiler import (
    ComputeProfiler,
    ComputeProfile,
    GEMMTestPoint,
    profile_and_save,
    load_latest_profile,
)

# 兼容旧 API
from .compat import (
    IntegratedCostModel,
    LegacyHardwareConfig,
    LegacyModelConfig,
    LegacyParallelConfig,
    LegacyCostModelConfig,
)

__all__ = [
    # Hardware
    "HardwareConfig",
    "NetworkTopology", 
    "GPUSpecs",
    "ClusterConfig",
    "GPUType",
    "NetworkType",
    "profile_gpu_compute",
    "profile_network_bandwidth",
    
    # Communication
    "CommunicationModel",
    "AllReduceModel",
    "AllGatherModel",
    "ReduceScatterModel",
    "AllToAllModel",
    "P2PModel",
    "CommType",
    "CommResult",
    "CommConfig",
    
    # Computation
    "ComputationModel",
    "AlphaBetaModel",
    "AlphaBetaParams",
    "LayerComputeProfile",
    "ModelConfig",
    "ComputeMode",
    "LayerType",
    
    # Memory
    "MemoryModel",
    "MemoryBreakdown",
    "ParallelConfig",
    "TrainingConfig",
    "ZeROConfig",
    "ActivationCheckpointConfig",
    "OffloadConfig",
    "ZeROStage",
    "CheckpointGranularity",
    "OffloadTarget",
    "AutoMemoryOptimizer",
    "RecomputeMethod",
    
    # Integrated
    "GalvatronCostModel",
    "CostModelConfig",
    "PredictionResult",
    "create_qwen3_30b_costmodel",
    
    # Compute Profiler
    "ComputeProfiler",
    "ComputeProfile",
    "GEMMTestPoint",
    "profile_and_save",
    "load_latest_profile",
    
    # Legacy Compatibility
    "IntegratedCostModel",
    "LegacyHardwareConfig",
    "LegacyModelConfig",
    "LegacyParallelConfig",
    "LegacyCostModelConfig",
]