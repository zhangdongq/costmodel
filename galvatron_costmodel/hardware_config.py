#!/usr/bin/env python3
"""
硬件配置模块 - 参考 Galvatron 的硬件感知设计

核心功能：
1. GPU 规格定义（算力、显存、带宽）
2. 网络拓扑建模（节点内 NVLink / 节点间 IB）
3. 集群配置（多机多卡拓扑）
4. 硬件 Profiling 数据管理
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class GPUType(Enum):
    """GPU 类型枚举"""
    H800 = "h800"
    H100 = "h100"
    A100_80G = "a100_80g"
    A100_40G = "a100_40g"
    A800 = "a800"
    V100 = "v100"
    CUSTOM = "custom"


class NetworkType(Enum):
    """网络类型枚举"""
    NVLINK = "nvlink"           # 节点内高速互联
    NVSWITCH = "nvswitch"       # NVSwitch 全互联
    PCIE = "pcie"               # PCIe 总线
    INFINIBAND = "infiniband"   # InfiniBand
    ROCE = "roce"               # RoCE
    ETHERNET = "ethernet"       # 以太网


@dataclass
class GPUSpecs:
    """GPU 硬件规格"""
    
    # 基本信息
    gpu_type: GPUType = GPUType.H800
    gpu_name: str = "NVIDIA H800"
    
    # 计算能力 (TFLOPS)
    fp16_tflops: float = 989.0      # FP16 峰值算力
    bf16_tflops: float = 989.0      # BF16 峰值算力
    fp32_tflops: float = 67.0       # FP32 峰值算力
    tf32_tflops: float = 495.0      # TF32 峰值算力
    fp8_tflops: float = 1979.0      # FP8 峰值算力 (Hopper+)
    
    # 显存
    memory_gb: float = 80.0         # 显存容量 (GB)
    memory_bandwidth_gbps: float = 3350.0  # 显存带宽 (GB/s)
    
    # Tensor Core
    tensor_core_enabled: bool = True
    tensor_core_speedup: float = 2.0  # Tensor Core 相对加速比
    
    # Kernel 开销
    kernel_launch_overhead_us: float = 5.0  # Kernel 启动延迟 (μs)
    
    @classmethod
    def h800(cls) -> "GPUSpecs":
        """H800 预设配置"""
        return cls(
            gpu_type=GPUType.H800,
            gpu_name="NVIDIA H800",
            fp16_tflops=989.0,
            bf16_tflops=989.0,
            fp32_tflops=67.0,
            tf32_tflops=495.0,
            fp8_tflops=1979.0,
            memory_gb=80.0,
            memory_bandwidth_gbps=3350.0,
        )
    
    @classmethod
    def a100_80g(cls) -> "GPUSpecs":
        """A100 80GB 预设配置"""
        return cls(
            gpu_type=GPUType.A100_80G,
            gpu_name="NVIDIA A100 80GB",
            fp16_tflops=312.0,
            bf16_tflops=312.0,
            fp32_tflops=19.5,
            tf32_tflops=156.0,
            fp8_tflops=0.0,  # A100 不支持 FP8
            memory_gb=80.0,
            memory_bandwidth_gbps=2039.0,
        )
    
    def get_effective_tflops(self, dtype: str = "bf16") -> float:
        """获取有效算力（考虑 Tensor Core）"""
        dtype_map = {
            "fp16": self.fp16_tflops,
            "bf16": self.bf16_tflops,
            "fp32": self.fp32_tflops,
            "tf32": self.tf32_tflops,
            "fp8": self.fp8_tflops,
        }
        base_tflops = dtype_map.get(dtype, self.bf16_tflops)
        
        if self.tensor_core_enabled and dtype in ["fp16", "bf16", "tf32", "fp8"]:
            return base_tflops
        return base_tflops


@dataclass
class NetworkTopology:
    """网络拓扑配置 - 参考 Galvatron 的分层通信建模"""
    
    # 节点内通信（NVLink/NVSwitch）
    intra_node_type: NetworkType = NetworkType.NVLINK
    intra_node_bandwidth_gbps: float = 900.0  # NVLink 4: 900 GB/s 双向
    intra_node_latency_us: float = 1.0        # 节点内延迟 (μs)
    
    # 节点间通信（InfiniBand/RoCE）
    inter_node_type: NetworkType = NetworkType.INFINIBAND
    inter_node_bandwidth_gbps: float = 200.0  # 200 Gbps IB
    inter_node_latency_us: float = 5.0        # 节点间延迟 (μs)
    
    # 通信效率因子
    allreduce_efficiency: float = 0.85        # Ring AllReduce 效率
    allgather_efficiency: float = 0.90        # AllGather 效率
    alltoall_efficiency: float = 0.80         # AllToAll 效率
    p2p_efficiency: float = 0.95              # P2P 效率
    
    # 消息开销
    message_header_bytes: int = 64            # 消息头开销
    
    @classmethod
    def h800_nvlink(cls) -> "NetworkTopology":
        """H800 NVLink 配置"""
        return cls(
            intra_node_type=NetworkType.NVLINK,
            intra_node_bandwidth_gbps=900.0,
            intra_node_latency_us=1.0,
            inter_node_type=NetworkType.INFINIBAND,
            inter_node_bandwidth_gbps=200.0,
            inter_node_latency_us=5.0,
        )
    
    @classmethod
    def a100_nvlink(cls) -> "NetworkTopology":
        """A100 NVLink 配置"""
        return cls(
            intra_node_type=NetworkType.NVLINK,
            intra_node_bandwidth_gbps=600.0,  # NVLink 3
            intra_node_latency_us=1.5,
            inter_node_type=NetworkType.INFINIBAND,
            inter_node_bandwidth_gbps=200.0,
            inter_node_latency_us=5.0,
        )
    
    def get_bandwidth(self, is_intra_node: bool) -> float:
        """获取带宽 (GB/s)"""
        if is_intra_node:
            return self.intra_node_bandwidth_gbps
        else:
            # IB 带宽单位通常是 Gbps，需要转换为 GB/s
            return self.inter_node_bandwidth_gbps / 8.0  # Gbps -> GB/s
    
    def get_latency(self, is_intra_node: bool) -> float:
        """获取延迟 (μs)"""
        if is_intra_node:
            return self.intra_node_latency_us
        else:
            return self.inter_node_latency_us
    
    def is_intra_node(self, src_rank: int, dst_rank: int, gpus_per_node: int) -> bool:
        """判断两个 rank 是否在同一节点内"""
        return src_rank // gpus_per_node == dst_rank // gpus_per_node


@dataclass
class ClusterConfig:
    """集群配置"""
    
    # 集群规模
    num_nodes: int = 1                    # 节点数
    gpus_per_node: int = 8                # 每节点 GPU 数
    
    # 拓扑信息
    topology_type: str = "fat_tree"       # fat_tree, torus, ring
    
    @property
    def total_gpus(self) -> int:
        """总 GPU 数"""
        return self.num_nodes * self.gpus_per_node
    
    def get_node_id(self, rank: int) -> int:
        """获取 rank 所在节点 ID"""
        return rank // self.gpus_per_node
    
    def get_local_rank(self, rank: int) -> int:
        """获取 rank 在节点内的本地 ID"""
        return rank % self.gpus_per_node
    
    def is_same_node(self, rank1: int, rank2: int) -> bool:
        """判断两个 rank 是否在同一节点"""
        return self.get_node_id(rank1) == self.get_node_id(rank2)


@dataclass
class HardwareConfig:
    """完整硬件配置 - 整合 GPU、网络、集群信息"""
    
    gpu: GPUSpecs = field(default_factory=GPUSpecs.h800)
    network: NetworkTopology = field(default_factory=NetworkTopology.h800_nvlink)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    
    # Profiling 数据（可选，用于校准）
    profiled_data: Dict = field(default_factory=dict)
    
    @classmethod
    def from_profile(cls, profile_path: str) -> "HardwareConfig":
        """从 Profiling 数据加载配置"""
        if not os.path.exists(profile_path):
            print(f"Warning: Profile file not found: {profile_path}")
            return cls()
        
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # 解析 GPU 配置
        if "gpu" in data:
            gpu_data = data["gpu"]
            config.gpu = GPUSpecs(
                gpu_type=GPUType.CUSTOM,
                gpu_name=gpu_data.get("name", "Custom GPU"),
                fp16_tflops=gpu_data.get("fp16_tflops", 989.0),
                bf16_tflops=gpu_data.get("bf16_tflops", 989.0),
                memory_gb=gpu_data.get("memory_gb", 80.0),
                memory_bandwidth_gbps=gpu_data.get("memory_bandwidth_gbps", 3350.0),
            )
        
        # 解析网络配置
        if "network" in data:
            net_data = data["network"]
            config.network = NetworkTopology(
                intra_node_bandwidth_gbps=net_data.get("intra_node_bandwidth_gbps", 900.0),
                intra_node_latency_us=net_data.get("intra_node_latency_us", 1.0),
                inter_node_bandwidth_gbps=net_data.get("inter_node_bandwidth_gbps", 200.0),
                inter_node_latency_us=net_data.get("inter_node_latency_us", 5.0),
            )
        
        # 解析集群配置
        if "cluster" in data:
            cluster_data = data["cluster"]
            config.cluster = ClusterConfig(
                num_nodes=cluster_data.get("num_nodes", 1),
                gpus_per_node=cluster_data.get("gpus_per_node", 8),
            )
        
        config.profiled_data = data.get("profiled_data", {})
        
        return config
    
    def save_profile(self, profile_path: str):
        """保存配置到文件"""
        data = {
            "gpu": {
                "name": self.gpu.gpu_name,
                "type": self.gpu.gpu_type.value,
                "fp16_tflops": self.gpu.fp16_tflops,
                "bf16_tflops": self.gpu.bf16_tflops,
                "memory_gb": self.gpu.memory_gb,
                "memory_bandwidth_gbps": self.gpu.memory_bandwidth_gbps,
            },
            "network": {
                "intra_node_type": self.network.intra_node_type.value,
                "intra_node_bandwidth_gbps": self.network.intra_node_bandwidth_gbps,
                "intra_node_latency_us": self.network.intra_node_latency_us,
                "inter_node_type": self.network.inter_node_type.value,
                "inter_node_bandwidth_gbps": self.network.inter_node_bandwidth_gbps,
                "inter_node_latency_us": self.network.inter_node_latency_us,
            },
            "cluster": {
                "num_nodes": self.cluster.num_nodes,
                "gpus_per_node": self.cluster.gpus_per_node,
            },
            "profiled_data": self.profiled_data,
        }
        
        os.makedirs(os.path.dirname(profile_path) if os.path.dirname(profile_path) else ".", exist_ok=True)
        with open(profile_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_intra_node_comm(self, src_rank: int, dst_rank: int) -> bool:
        """判断通信是否在节点内"""
        return self.cluster.is_same_node(src_rank, dst_rank)
    
    def get_comm_bandwidth(self, is_intra_node: bool) -> float:
        """获取通信带宽 (GB/s)"""
        return self.network.get_bandwidth(is_intra_node)
    
    def get_comm_latency(self, is_intra_node: bool) -> float:
        """获取通信延迟 (μs)"""
        return self.network.get_latency(is_intra_node)


# ==================== Profiling 工具函数 ====================

def profile_gpu_compute(warmup_iters: int = 10, bench_iters: int = 100) -> Dict:
    """
    Profile GPU 计算性能
    
    返回实测的 TFLOPS 数据
    """
    try:
        import paddle
        import time
        
        # 测试矩阵乘法性能
        sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]
        results = {}
        
        for m, n in sizes:
            k = n
            a = paddle.randn([m, k], dtype='bfloat16')
            b = paddle.randn([k, n], dtype='bfloat16')
            
            # Warmup
            for _ in range(warmup_iters):
                c = paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(bench_iters):
                c = paddle.matmul(a, b)
            paddle.device.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            # 计算 TFLOPS
            flops = 2 * m * n * k * bench_iters
            tflops = flops / elapsed / 1e12
            
            results[f"gemm_{m}x{n}"] = {
                "time_ms": elapsed / bench_iters * 1000,
                "tflops": tflops,
            }
        
        return results
        
    except Exception as e:
        print(f"Warning: GPU profiling failed: {e}")
        return {}


def profile_network_bandwidth(warmup_iters: int = 5, bench_iters: int = 20) -> Dict:
    """
    Profile 网络带宽
    
    返回实测的带宽数据 (GB/s)
    """
    try:
        import paddle
        import paddle.distributed as dist
        import time
        
        if not dist.is_initialized():
            print("Warning: Distributed not initialized, skipping network profiling")
            return {}
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        results = {}
        
        # 测试不同数据大小
        sizes_mb = [1, 4, 16, 64, 256]
        
        for size_mb in sizes_mb:
            num_elements = size_mb * 1024 * 1024 // 2  # bfloat16
            tensor = paddle.randn([num_elements], dtype='bfloat16')
            
            # Warmup
            for _ in range(warmup_iters):
                dist.all_reduce(tensor)
            paddle.device.cuda.synchronize()
            
            # Benchmark AllReduce
            start = time.perf_counter()
            for _ in range(bench_iters):
                dist.all_reduce(tensor)
            paddle.device.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            # Ring AllReduce 实际传输量: 2 * (N-1) / N * data_size
            data_size_bytes = num_elements * 2
            ring_factor = 2 * (world_size - 1) / world_size
            total_bytes = data_size_bytes * ring_factor * bench_iters
            
            bandwidth_gbps = total_bytes / elapsed / 1e9
            
            results[f"allreduce_{size_mb}mb"] = {
                "time_ms": elapsed / bench_iters * 1000,
                "bandwidth_gbps": bandwidth_gbps,
            }
        
        return results
        
    except Exception as e:
        print(f"Warning: Network profiling failed: {e}")
        return {}