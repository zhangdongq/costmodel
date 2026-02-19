#!/usr/bin/env python3
"""
算力 Profiler - 测试真实 GPU 算力并拟合算力曲线

功能：
1. 多点测试不同规模下的有效算力 (TFLOPS)
2. 拟合算力曲线（考虑不同 GEMM 规模的效率差异）
3. 根据训练配置估算平均有效算力
4. 保存/加载测试结果
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np


@dataclass
class GEMMTestPoint:
    """单个 GEMM 测试点"""
    m: int
    n: int
    k: int
    time_ms: float
    tflops: float
    efficiency: float  # 相对于峰值的效率


@dataclass
class ComputeProfile:
    """算力测试 Profile"""
    # 硬件信息
    gpu_name: str = ""
    gpu_count: int = 1
    peak_tflops: float = 989.0  # 理论峰值
    
    # 测试信息
    test_date: str = ""
    dtype: str = "bfloat16"
    
    # 测试结果
    gemm_results: List[Dict] = field(default_factory=list)
    
    # 拟合参数（效率 vs GEMM 规模）
    # efficiency = a * log(flops) + b （对数拟合）
    fit_a: float = 0.0
    fit_b: float = 0.0
    
    # 分段效率（按 GEMM 规模分段）
    efficiency_small: float = 0.3   # M*N*K < 1e9
    efficiency_medium: float = 0.5  # 1e9 <= M*N*K < 1e11
    efficiency_large: float = 0.6   # M*N*K >= 1e11
    
    def get_efficiency(self, m: int, n: int, k: int) -> float:
        """根据 GEMM 规模获取效率"""
        flops = 2 * m * n * k
        
        # 使用拟合曲线
        if self.fit_a != 0 or self.fit_b != 0:
            log_flops = np.log10(max(flops, 1))
            efficiency = self.fit_a * log_flops + self.fit_b
            return max(0.1, min(0.9, efficiency))
        
        # 使用分段效率
        if flops < 1e9:
            return self.efficiency_small
        elif flops < 1e11:
            return self.efficiency_medium
        else:
            return self.efficiency_large
    
    def get_effective_tflops(self, m: int, n: int, k: int) -> float:
        """获取有效算力"""
        return self.peak_tflops * self.get_efficiency(m, n, k)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ComputeProfile":
        """从字典创建"""
        profile = cls()
        for key, value in data.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        return profile


class ComputeProfiler:
    """
    算力 Profiler
    
    测试多个 GEMM 规模点，拟合算力曲线
    """
    
    def __init__(self, dtype: str = "bfloat16"):
        self.dtype = dtype
        self.profile: Optional[ComputeProfile] = None
        
        # 默认测试点：覆盖 Transformer 常见的 GEMM 规模
        # (M, N, K) 对应不同的计算场景
        self.default_test_points = [
            # 小规模 GEMM (Attention 的小 batch)
            (256, 6144, 6144),      # 小 batch attention
            (512, 6144, 6144),
            (1024, 6144, 6144),
            
            # 中等规模 GEMM (典型的 MLP)
            (2048, 6144, 16384),    # MLP up/gate
            (4096, 6144, 16384),
            (8192, 6144, 16384),
            
            # 大规模 GEMM (长序列)
            (8192, 16384, 6144),    # MLP down
            (16384, 6144, 6144),    # 长序列 attention
            (32768, 6144, 6144),
            
            # 特大规模 GEMM
            (65536, 6144, 6144),
        ]
    
    def run_gemm_test(self, m: int, n: int, k: int,
                      warmup_iters: int = 10,
                      bench_iters: int = 50) -> GEMMTestPoint:
        """
        运行单个 GEMM 测试点
        """
        import paddle
        
        # 创建测试矩阵
        dtype_map = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}
        paddle_dtype = dtype_map.get(self.dtype, "bfloat16")
        
        a = paddle.randn([m, k], dtype=paddle_dtype)
        b = paddle.randn([k, n], dtype=paddle_dtype)
        
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
        
        # 计算指标
        time_ms = elapsed / bench_iters * 1000
        flops = 2 * m * n * k * bench_iters
        tflops = flops / elapsed / 1e12
        
        # 效率（需要知道峰值才能计算，先用默认值）
        peak_tflops = 989.0  # H800 BF16
        efficiency = tflops / peak_tflops
        
        return GEMMTestPoint(
            m=m, n=n, k=k,
            time_ms=time_ms,
            tflops=tflops,
            efficiency=efficiency
        )
    
    def run_full_profile(self, 
                         test_points: List[Tuple[int, int, int]] = None,
                         warmup_iters: int = 10,
                         bench_iters: int = 50,
                         peak_tflops: float = 989.0) -> ComputeProfile:
        """
        运行完整的算力测试
        
        Args:
            test_points: 测试点列表 [(M, N, K), ...]
            warmup_iters: 预热迭代次数
            bench_iters: 基准测试迭代次数
            peak_tflops: GPU 理论峰值算力
        
        Returns:
            ComputeProfile 测试结果
        """
        import paddle
        
        if test_points is None:
            test_points = self.default_test_points
        
        # 获取 GPU 信息
        try:
            gpu_name = paddle.device.cuda.get_device_name(0)
            gpu_count = paddle.device.cuda.device_count()
        except:
            gpu_name = "Unknown GPU"
            gpu_count = 1
        
        print(f"\n{'='*70}")
        print(f"🚀 GPU 算力测试 - {gpu_name}")
        print(f"{'='*70}")
        print(f"  理论峰值: {peak_tflops} TFLOPS ({self.dtype})")
        print(f"  测试点数: {len(test_points)}")
        print(f"  预热次数: {warmup_iters}, 基准次数: {bench_iters}")
        print(f"{'-'*70}")
        print(f"{'M':>8} {'N':>8} {'K':>8} {'Time(ms)':>12} {'TFLOPS':>10} {'Efficiency':>12}")
        print(f"{'-'*70}")
        
        results = []
        
        for m, n, k in test_points:
            try:
                point = self.run_gemm_test(m, n, k, warmup_iters, bench_iters)
                point.efficiency = point.tflops / peak_tflops
                
                results.append(asdict(point))
                
                print(f"{m:>8} {n:>8} {k:>8} {point.time_ms:>12.3f} "
                      f"{point.tflops:>10.2f} {point.efficiency:>11.1%}")
                
            except Exception as e:
                print(f"{m:>8} {n:>8} {k:>8} {'FAILED':>12} - {e}")
        
        print(f"{'-'*70}")
        
        # 创建 Profile
        profile = ComputeProfile(
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            peak_tflops=peak_tflops,
            test_date=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            dtype=self.dtype,
            gemm_results=results,
        )
        
        # 拟合效率曲线
        self._fit_efficiency_curve(profile)
        
        self.profile = profile
        
        # 打印拟合结果
        print(f"\n📊 效率拟合结果:")
        print(f"  小规模 (< 1e9 FLOPs):    {profile.efficiency_small:.1%}")
        print(f"  中等规模 (1e9 - 1e11):   {profile.efficiency_medium:.1%}")
        print(f"  大规模 (>= 1e11):        {profile.efficiency_large:.1%}")
        print(f"  对数拟合: efficiency = {profile.fit_a:.4f} * log10(FLOPs) + {profile.fit_b:.4f}")
        print(f"{'='*70}\n")
        
        return profile
    
    def _fit_efficiency_curve(self, profile: ComputeProfile):
        """
        拟合效率曲线
        
        使用两种方式：
        1. 分段平均效率
        2. 对数线性拟合: efficiency = a * log10(FLOPs) + b
        """
        if not profile.gemm_results:
            return
        
        # 分类数据点
        small_effs = []
        medium_effs = []
        large_effs = []
        
        log_flops_list = []
        efficiency_list = []
        
        for result in profile.gemm_results:
            flops = 2 * result["m"] * result["n"] * result["k"]
            eff = result["efficiency"]
            
            log_flops_list.append(np.log10(flops))
            efficiency_list.append(eff)
            
            if flops < 1e9:
                small_effs.append(eff)
            elif flops < 1e11:
                medium_effs.append(eff)
            else:
                large_effs.append(eff)
        
        # 分段平均效率
        profile.efficiency_small = np.mean(small_effs) if small_effs else 0.3
        profile.efficiency_medium = np.mean(medium_effs) if medium_effs else 0.5
        profile.efficiency_large = np.mean(large_effs) if large_effs else 0.6
        
        # 对数线性拟合
        if len(log_flops_list) >= 2:
            log_flops = np.array(log_flops_list)
            efficiencies = np.array(efficiency_list)
            
            # 简单线性回归: y = ax + b
            n = len(log_flops)
            sum_x = np.sum(log_flops)
            sum_y = np.sum(efficiencies)
            sum_xy = np.sum(log_flops * efficiencies)
            sum_xx = np.sum(log_flops * log_flops)
            
            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) > 1e-10:
                profile.fit_a = (n * sum_xy - sum_x * sum_y) / denom
                profile.fit_b = (sum_y - profile.fit_a * sum_x) / n
    
    def estimate_training_efficiency(self, 
                                     hidden_size: int = 6144,
                                     intermediate_size: int = 16384,
                                     seq_len: int = 8192,
                                     batch_size: int = 1,
                                     num_samples: int = 10) -> float:
        """
        估算真实训练场景的平均效率
        
        随机采样训练中可能出现的 GEMM 规模，计算平均效率
        """
        if self.profile is None:
            return 0.5  # 默认值
        
        # 典型的 Transformer GEMM 规模
        gemm_types = [
            # Attention: QKV projection
            (batch_size * seq_len, hidden_size * 3, hidden_size),
            # Attention: Output projection
            (batch_size * seq_len, hidden_size, hidden_size),
            # MLP: Gate and Up
            (batch_size * seq_len, intermediate_size * 2, hidden_size),
            # MLP: Down
            (batch_size * seq_len, hidden_size, intermediate_size),
        ]
        
        efficiencies = []
        for m, n, k in gemm_types:
            eff = self.profile.get_efficiency(m, n, k)
            efficiencies.append(eff)
        
        return np.mean(efficiencies)
    
    def save_profile(self, save_dir: str = "./profiles") -> str:
        """
        保存测试结果到文件
        
        文件名格式: {gpu_name}_{date}.json
        
        Returns:
            保存的文件路径
        """
        if self.profile is None:
            raise ValueError("No profile to save. Run run_full_profile() first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 清理 GPU 名称用于文件名
        gpu_name_clean = self.profile.gpu_name.replace(" ", "_").replace("/", "-")
        filename = f"{gpu_name_clean}_{self.profile.test_date}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.profile.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Profile 已保存到: {filepath}")
        return filepath
    
    def load_profile(self, filepath: str) -> ComputeProfile:
        """从文件加载测试结果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.profile = ComputeProfile.from_dict(data)
        print(f"✅ 已加载 Profile: {self.profile.gpu_name} ({self.profile.test_date})")
        return self.profile
    
    @staticmethod
    def list_profiles(save_dir: str = "./profiles") -> List[str]:
        """列出所有保存的 Profile 文件"""
        if not os.path.exists(save_dir):
            return []
        
        files = [f for f in os.listdir(save_dir) if f.endswith('.json')]
        return sorted(files)


# ==================== 便捷函数 ====================

def profile_and_save(peak_tflops: float = 989.0,
                     dtype: str = "bfloat16",
                     save_dir: str = "./profiles") -> str:
    """
    运行算力测试并保存结果
    
    Args:
        peak_tflops: GPU 理论峰值算力
        dtype: 数据类型
        save_dir: 保存目录
    
    Returns:
        保存的文件路径
    """
    profiler = ComputeProfiler(dtype=dtype)
    profiler.run_full_profile(peak_tflops=peak_tflops)
    return profiler.save_profile(save_dir)


def load_latest_profile(save_dir: str = "./profiles") -> Optional[ComputeProfile]:
    """加载最新的 Profile 文件"""
    profiler = ComputeProfiler()
    files = profiler.list_profiles(save_dir)
    
    if not files:
        print(f"⚠️ 没有找到 Profile 文件在 {save_dir}")
        return None
    
    latest_file = files[-1]  # 文件名包含日期，排序后最后一个是最新的
    filepath = os.path.join(save_dir, latest_file)
    
    return profiler.load_profile(filepath)


# ==================== 测试函数 ====================

def test_compute_profiler():
    """测试算力 Profiler"""
    print("=" * 70)
    print("测试 ComputeProfiler")
    print("=" * 70)
    
    # 创建 Profiler
    profiler = ComputeProfiler(dtype="bfloat16")
    
    # 运行测试（使用较少的测试点加快速度）
    test_points = [
        (1024, 6144, 6144),
        (4096, 6144, 16384),
        (8192, 16384, 6144),
    ]
    
    profile = profiler.run_full_profile(
        test_points=test_points,
        warmup_iters=5,
        bench_iters=20,
        peak_tflops=989.0
    )
    
    # 测试效率估算
    print("\n📊 训练效率估算:")
    for seq_len in [1024, 2048, 4096, 8192]:
        eff = profiler.estimate_training_efficiency(
            hidden_size=6144,
            intermediate_size=16384,
            seq_len=seq_len,
            batch_size=1
        )
        print(f"  seq_len={seq_len}: 平均效率 {eff:.1%}")
    
    # 保存
    filepath = profiler.save_profile("./profiles")
    
    # 重新加载
    profiler2 = ComputeProfiler()
    profiler2.load_profile(filepath)
    
    print("\n测试完成!")


if __name__ == "__main__":
    test_compute_profiler()