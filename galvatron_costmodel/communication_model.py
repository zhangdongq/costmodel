#!/usr/bin/env python3
"""
通信模型模块 - 参考 Galvatron 的通信时间建模

核心功能：
1. AllReduce 通信建模（TP 层内同步）
2. AllGather/ReduceScatter 建模（ZeRO/SDP）
3. AllToAll 通信建模（MoE EP）
4. P2P 通信建模（PP 流水线）
5. 分层通信感知（节点内/节点间）
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .hardware_config import HardwareConfig, NetworkTopology


class CommType(Enum):
    """通信类型枚举"""
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    REDUCE_SCATTER = "reduce_scatter"
    ALLTOALL = "alltoall"
    P2P_SEND = "p2p_send"
    P2P_RECV = "p2p_recv"
    BROADCAST = "broadcast"


@dataclass
class CommConfig:
    """通信配置"""
    comm_type: CommType
    data_size_bytes: int
    num_gpus: int
    is_intra_node: bool = True
    dtype_bytes: int = 2  # bfloat16 = 2 bytes


@dataclass
class CommResult:
    """通信预测结果"""
    time_ms: float                  # 预测时间 (ms)
    bandwidth_utilized_gbps: float  # 实际利用带宽 (GB/s)
    comm_volume_bytes: int          # 实际通信量 (bytes)
    latency_ms: float               # 延迟部分 (ms)
    transfer_ms: float              # 传输部分 (ms)


class BaseCommModel:
    """通信模型基类"""
    
    def __init__(self, hardware: HardwareConfig):
        self.hardware = hardware
    
    def _get_bandwidth(self, is_intra_node: bool) -> float:
        """获取带宽 (GB/s)"""
        return self.hardware.get_comm_bandwidth(is_intra_node)
    
    def _get_latency(self, is_intra_node: bool) -> float:
        """获取延迟 (μs)"""
        return self.hardware.get_comm_latency(is_intra_node)
    
    def _bytes_to_gb(self, bytes_count: int) -> float:
        """字节转 GB"""
        return bytes_count / (1024 ** 3)


class AllReduceModel(BaseCommModel):
    """
    AllReduce 通信模型
    
    用于：TP 张量并行的梯度同步
    
    Ring AllReduce 算法：
    - 通信量 = 2 * (N-1)/N * data_size
    - 延迟 = 2 * (N-1) * α
    
    其中 N 是参与的 GPU 数量，α 是单次通信延迟
    """
    
    def predict(self, data_size_bytes: int, num_gpus: int, 
                is_intra_node: bool = True) -> CommResult:
        """
        预测 AllReduce 通信时间
        
        Args:
            data_size_bytes: 需要 AllReduce 的数据量 (bytes)
            num_gpus: 参与的 GPU 数量
            is_intra_node: 是否为节点内通信
        """
        if num_gpus <= 1:
            return CommResult(
                time_ms=0.0,
                bandwidth_utilized_gbps=0.0,
                comm_volume_bytes=0,
                latency_ms=0.0,
                transfer_ms=0.0,
            )
        
        # Ring AllReduce 通信量
        ring_factor = 2 * (num_gpus - 1) / num_gpus
        comm_volume_bytes = int(data_size_bytes * ring_factor)
        
        # 获取带宽和延迟
        bandwidth_gbps = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        
        # 效率因子
        efficiency = self.hardware.network.allreduce_efficiency
        
        # 延迟部分：2 * (N-1) 次通信
        num_steps = 2 * (num_gpus - 1)
        latency_ms = num_steps * latency_us / 1000.0
        
        # 传输部分
        data_size_gb = self._bytes_to_gb(comm_volume_bytes)
        transfer_ms = data_size_gb / (bandwidth_gbps * efficiency) * 1000.0
        
        # 总时间
        total_time_ms = latency_ms + transfer_ms
        
        # 实际利用带宽
        bandwidth_utilized = data_size_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_utilized_gbps=bandwidth_utilized,
            comm_volume_bytes=comm_volume_bytes,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )


class AllGatherModel(BaseCommModel):
    """
    AllGather 通信模型
    
    用于：
    - ZeRO-3 参数收集
    - Sequence Parallel 激活收集
    
    通信量 = (N-1)/N * data_size
    """
    
    def predict(self, data_size_bytes: int, num_gpus: int,
                is_intra_node: bool = True) -> CommResult:
        """预测 AllGather 通信时间"""
        if num_gpus <= 1:
            return CommResult(
                time_ms=0.0,
                bandwidth_utilized_gbps=0.0,
                comm_volume_bytes=0,
                latency_ms=0.0,
                transfer_ms=0.0,
            )
        
        # AllGather 通信量
        gather_factor = (num_gpus - 1) / num_gpus
        comm_volume_bytes = int(data_size_bytes * gather_factor)
        
        # 获取带宽和延迟
        bandwidth_gbps = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        
        efficiency = self.hardware.network.allgather_efficiency
        
        # 延迟部分
        num_steps = num_gpus - 1
        latency_ms = num_steps * latency_us / 1000.0
        
        # 传输部分
        data_size_gb = self._bytes_to_gb(comm_volume_bytes)
        transfer_ms = data_size_gb / (bandwidth_gbps * efficiency) * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        bandwidth_utilized = data_size_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_utilized_gbps=bandwidth_utilized,
            comm_volume_bytes=comm_volume_bytes,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )


class ReduceScatterModel(BaseCommModel):
    """
    ReduceScatter 通信模型
    
    用于：ZeRO-2 梯度分片
    
    通信量 = (N-1)/N * data_size（与 AllGather 相同）
    """
    
    def predict(self, data_size_bytes: int, num_gpus: int,
                is_intra_node: bool = True) -> CommResult:
        """预测 ReduceScatter 通信时间"""
        # 与 AllGather 类似
        if num_gpus <= 1:
            return CommResult(
                time_ms=0.0,
                bandwidth_utilized_gbps=0.0,
                comm_volume_bytes=0,
                latency_ms=0.0,
                transfer_ms=0.0,
            )
        
        scatter_factor = (num_gpus - 1) / num_gpus
        comm_volume_bytes = int(data_size_bytes * scatter_factor)
        
        bandwidth_gbps = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.allreduce_efficiency  # 使用 AllReduce 效率
        
        num_steps = num_gpus - 1
        latency_ms = num_steps * latency_us / 1000.0
        
        data_size_gb = self._bytes_to_gb(comm_volume_bytes)
        transfer_ms = data_size_gb / (bandwidth_gbps * efficiency) * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        bandwidth_utilized = data_size_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_utilized_gbps=bandwidth_utilized,
            comm_volume_bytes=comm_volume_bytes,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )


class AllToAllModel(BaseCommModel):
    """
    AllToAll 通信模型
    
    用于：MoE Expert Parallel 的 dispatch 和 combine
    
    特点：
    - 每个 GPU 向其他所有 GPU 发送不同数据
    - 通信量 = (N-1)/N * data_size（每个 GPU 发送和接收）
    - 可能存在负载不均衡
    
    参考 Galvatron 的 A2A 建模，采用分段拟合：
    - 小消息：受延迟主导，使用幂律模型
    - 大消息：受带宽主导，使用线性模型
    """
    
    def __init__(self, hardware: HardwareConfig):
        super().__init__(hardware)
        
        # 分段阈值 (MB)
        self.segment_threshold_mb = 4.0
        
        # 小消息幂律参数 (可通过校准调整)
        self.small_msg_alpha = 0.05
        self.small_msg_beta = 0.8
        
        # 负载不均衡因子
        self.imbalance_factor = 1.2
        
        # 校准因子（从实测数据获取）
        self.calibration_factor = 1.0
    
    def predict(self, data_size_bytes: int, num_gpus: int,
                is_intra_node: bool = True,
                topk: int = 8,
                num_experts: int = 128) -> CommResult:
        """
        预测 AllToAll 通信时间
        
        Args:
            data_size_bytes: 单次 A2A 的数据量 (bytes)
            num_gpus: EP 并行度
            is_intra_node: 是否节点内通信
            topk: MoE TopK 值
            num_experts: 专家总数
        """
        if num_gpus <= 1:
            return CommResult(
                time_ms=0.0,
                bandwidth_utilized_gbps=0.0,
                comm_volume_bytes=0,
                latency_ms=0.0,
                transfer_ms=0.0,
            )
        
        # A2A 通信量估算
        # 每个 GPU 需要向 (N-1) 个其他 GPU 发送数据
        a2a_factor = (num_gpus - 1) / num_gpus
        comm_volume_bytes = int(data_size_bytes * a2a_factor * 2)  # dispatch + combine
        
        # 数据大小 (MB)
        data_size_mb = comm_volume_bytes / (1024 * 1024)
        
        # 分段拟合
        bandwidth_gbps = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.alltoall_efficiency
        
        if data_size_mb < self.segment_threshold_mb:
            # 小消息：幂律模型 (受延迟主导)
            latency_ms = latency_us / 1000.0
            transfer_ms = self.small_msg_alpha * (data_size_mb ** self.small_msg_beta)
        else:
            # 大消息：线性模型 (受带宽主导)
            latency_ms = latency_us / 1000.0
            data_size_gb = self._bytes_to_gb(comm_volume_bytes)
            transfer_ms = data_size_gb / (bandwidth_gbps * efficiency) * 1000.0
        
        # 考虑负载不均衡
        imbalance_penalty = self.imbalance_factor
        
        # 消息数：每个 rank 与 (N-1) 个其他 rank 通信
        num_messages = num_gpus - 1
        
        # 总时间 = (延迟 + 传输) × 消息数 × 不均衡因子 × 校准因子
        total_time_ms = (latency_ms + transfer_ms) * num_messages * imbalance_penalty * self.calibration_factor
        
        bandwidth_utilized = self._bytes_to_gb(comm_volume_bytes) / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_utilized_gbps=bandwidth_utilized,
            comm_volume_bytes=comm_volume_bytes,
            latency_ms=latency_ms * num_messages,
            transfer_ms=transfer_ms * num_messages,
        )
    
    def calibrate(self, calibration_data: List[Dict]):
        """
        使用实测数据校准模型
        
        calibration_data 格式：
        [
            {
                "data_size_bytes": 1000000,
                "num_gpus": 8,
                "actual_time_ms": 1.5,
            },
            ...
        ]
        """
        if not calibration_data:
            return
        
        errors = []
        for data in calibration_data:
            predicted = self.predict(
                data["data_size_bytes"],
                data["num_gpus"],
                data.get("is_intra_node", True),
            )
            
            actual_time = data["actual_time_ms"]
            if actual_time > 0:
                ratio = actual_time / predicted.time_ms
                errors.append(ratio)
        
        if errors:
            # 使用中位数作为校准因子（更鲁棒）
            import statistics
            self.calibration_factor = statistics.median(errors)
            print(f"A2A calibration factor updated: {self.calibration_factor:.3f}")


class P2PModel(BaseCommModel):
    """
    P2P (Point-to-Point) 通信模型
    
    用于：Pipeline Parallel 的 Send/Recv
    
    时间 = latency + data_size / bandwidth
    """
    
    def predict(self, data_size_bytes: int, is_intra_node: bool = False) -> CommResult:
        """预测 P2P 通信时间"""
        bandwidth_gbps = self._get_bandwidth(is_intra_node)
        latency_us = self._get_latency(is_intra_node)
        efficiency = self.hardware.network.p2p_efficiency
        
        latency_ms = latency_us / 1000.0
        
        data_size_gb = self._bytes_to_gb(data_size_bytes)
        transfer_ms = data_size_gb / (bandwidth_gbps * efficiency) * 1000.0
        
        total_time_ms = latency_ms + transfer_ms
        bandwidth_utilized = data_size_gb / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
        
        return CommResult(
            time_ms=total_time_ms,
            bandwidth_utilized_gbps=bandwidth_utilized,
            comm_volume_bytes=data_size_bytes,
            latency_ms=latency_ms,
            transfer_ms=transfer_ms,
        )


class CommunicationModel:
    """
    统一通信模型 - 整合所有通信类型
    
    参考 Galvatron 的设计：
    1. 感知网络拓扑（节点内/节点间）
    2. 支持多种通信原语
    3. 支持通过实测数据校准
    """
    
    def __init__(self, hardware: HardwareConfig):
        self.hardware = hardware
        
        # 初始化各通信模型
        self.allreduce = AllReduceModel(hardware)
        self.allgather = AllGatherModel(hardware)
        self.reduce_scatter = ReduceScatterModel(hardware)
        self.alltoall = AllToAllModel(hardware)
        self.p2p = P2PModel(hardware)
    
    def predict_tp_comm(self, activation_size_bytes: int, tp_degree: int) -> CommResult:
        """
        预测 TP (Tensor Parallel) 通信时间
        
        TP 在每层的 Attention 和 MLP 后各需要一次 AllReduce
        通常在节点内进行
        """
        if tp_degree <= 1:
            return CommResult(0.0, 0.0, 0, 0.0, 0.0)
        
        # TP 通常在节点内
        is_intra_node = tp_degree <= self.hardware.cluster.gpus_per_node
        
        return self.allreduce.predict(activation_size_bytes, tp_degree, is_intra_node)
    
    def predict_dp_comm(self, gradient_size_bytes: int, dp_degree: int,
                        use_sharding: bool = False) -> CommResult:
        """
        预测 DP (Data Parallel) 通信时间
        
        - 普通 DP: AllReduce 梯度
        - SDP/ZeRO-2: ReduceScatter 梯度
        """
        if dp_degree <= 1:
            return CommResult(0.0, 0.0, 0, 0.0, 0.0)
        
        # DP 通常跨节点
        is_intra_node = dp_degree <= self.hardware.cluster.gpus_per_node
        
        if use_sharding:
            return self.reduce_scatter.predict(gradient_size_bytes, dp_degree, is_intra_node)
        else:
            return self.allreduce.predict(gradient_size_bytes, dp_degree, is_intra_node)
    
    def predict_ep_comm(self, token_data_bytes: int, ep_degree: int,
                        topk: int = 8, num_experts: int = 128) -> CommResult:
        """
        预测 EP (Expert Parallel) 通信时间
        
        MoE 层的 dispatch + combine 两次 AllToAll
        """
        if ep_degree <= 1:
            return CommResult(0.0, 0.0, 0, 0.0, 0.0)
        
        # EP 可能跨节点
        is_intra_node = ep_degree <= self.hardware.cluster.gpus_per_node
        
        return self.alltoall.predict(
            token_data_bytes, ep_degree, is_intra_node,
            topk=topk, num_experts=num_experts
        )
    
    def predict_pp_comm(self, activation_size_bytes: int, pp_degree: int,
                        num_micro_batches: int) -> CommResult:
        """
        预测 PP (Pipeline Parallel) 通信时间
        
        每个 micro-batch 需要在 stage 之间传递激活值
        """
        if pp_degree <= 1:
            return CommResult(0.0, 0.0, 0, 0.0, 0.0)
        
        # PP 通常跨节点
        is_intra_node = False
        
        # 每个 micro-batch 需要 (pp_degree - 1) 次 P2P 通信
        # 1F1B 调度：总共 2 * (pp_degree - 1) * num_micro_batches 次
        single_p2p = self.p2p.predict(activation_size_bytes, is_intra_node)
        
        num_comms = 2 * (pp_degree - 1)  # 每个 micro-batch
        
        return CommResult(
            time_ms=single_p2p.time_ms * num_comms,
            bandwidth_utilized_gbps=single_p2p.bandwidth_utilized_gbps,
            comm_volume_bytes=single_p2p.comm_volume_bytes * num_comms,
            latency_ms=single_p2p.latency_ms * num_comms,
            transfer_ms=single_p2p.transfer_ms * num_comms,
        )
    
    def predict_sp_comm(self, activation_size_bytes: int, tp_degree: int) -> CommResult:
        """
        预测 SP (Sequence Parallel) 通信时间
        
        Megatron-SP: 使用 AllGather 收集序列维度
        """
        if tp_degree <= 1:
            return CommResult(0.0, 0.0, 0, 0.0, 0.0)
        
        is_intra_node = tp_degree <= self.hardware.cluster.gpus_per_node
        
        return self.allgather.predict(activation_size_bytes, tp_degree, is_intra_node)
    
    def calibrate_alltoall(self, calibration_data: List[Dict]):
        """校准 AllToAll 模型"""
        self.alltoall.calibrate(calibration_data)