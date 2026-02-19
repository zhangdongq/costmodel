#!/usr/bin/env python3
"""
算力测试脚本 - 运行 GPU 算力测试并保存结果

使用方法:
    python run_compute_profile.py [--peak_tflops 989.0] [--dtype bfloat16] [--save_dir ./profiles]
"""

import argparse
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from galvatron_costmodel import (
    ComputeProfiler,
    profile_and_save,
    load_latest_profile,
)


def main():
    parser = argparse.ArgumentParser(description="GPU 算力测试")
    parser.add_argument("--peak_tflops", type=float, default=989.0,
                        help="GPU 理论峰值算力 (TFLOPS), 默认 989.0 (H800 BF16)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="数据类型, 默认 bfloat16")
    parser.add_argument("--save_dir", type=str, default="./profiles",
                        help="保存目录, 默认 ./profiles")
    parser.add_argument("--warmup", type=int, default=10,
                        help="预热迭代次数, 默认 10")
    parser.add_argument("--bench", type=int, default=50,
                        help="基准测试迭代次数, 默认 50")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式（较少测试点）")
    parser.add_argument("--list", action="store_true",
                        help="列出已保存的 Profile 文件")
    parser.add_argument("--load", type=str, default=None,
                        help="加载指定的 Profile 文件")
    
    args = parser.parse_args()
    
    # 列出已有文件
    if args.list:
        profiler = ComputeProfiler()
        files = profiler.list_profiles(args.save_dir)
        if files:
            print(f"\n📁 已保存的 Profile 文件 ({args.save_dir}):")
            for f in files:
                print(f"  • {f}")
        else:
            print(f"\n⚠️ 没有找到 Profile 文件在 {args.save_dir}")
        return
    
    # 加载已有文件
    if args.load:
        profiler = ComputeProfiler()
        profile = profiler.load_profile(args.load)
        
        print(f"\n📊 Profile 信息:")
        print(f"  GPU: {profile.gpu_name}")
        print(f"  日期: {profile.test_date}")
        print(f"  峰值算力: {profile.peak_tflops} TFLOPS")
        print(f"  效率 (小): {profile.efficiency_small:.1%}")
        print(f"  效率 (中): {profile.efficiency_medium:.1%}")
        print(f"  效率 (大): {profile.efficiency_large:.1%}")
        
        if profile.gemm_results:
            print(f"\n📈 测试结果详情:")
            print(f"  {'M':>8} {'N':>8} {'K':>8} {'TFLOPS':>10} {'Efficiency':>12}")
            print(f"  {'-'*50}")
            for r in profile.gemm_results:
                print(f"  {r['m']:>8} {r['n']:>8} {r['k']:>8} "
                      f"{r['tflops']:>10.2f} {r['efficiency']:>11.1%}")
        return
    
    # 运行测试
    profiler = ComputeProfiler(dtype=args.dtype)
    
    # 测试点配置
    if args.quick:
        # 快速测试：3 个代表性点
        test_points = [
            (1024, 6144, 6144),    # 小规模
            (4096, 6144, 16384),   # 中等规模
            (16384, 6144, 6144),   # 大规模
        ]
    else:
        # 完整测试：使用默认的多点测试
        test_points = None
    
    profile = profiler.run_full_profile(
        test_points=test_points,
        warmup_iters=args.warmup,
        bench_iters=args.bench,
        peak_tflops=args.peak_tflops
    )
    
    # 估算训练效率
    print("\n📊 训练场景效率估算 (hidden=6144, ffn=16384, batch=1):")
    for seq_len in [1024, 2048, 4096, 8192]:
        eff = profiler.estimate_training_efficiency(
            hidden_size=6144,
            intermediate_size=16384,
            seq_len=seq_len,
            batch_size=1
        )
        effective_tflops = args.peak_tflops * eff
        print(f"  seq_len={seq_len:>5}: 效率 {eff:.1%}, 有效算力 {effective_tflops:.1f} TFLOPS")
    
    # 保存结果
    filepath = profiler.save_profile(args.save_dir)
    
    print(f"\n✅ 测试完成！")
    print(f"   结果已保存到: {filepath}")
    print(f"   可使用 --load {filepath} 查看详情")


if __name__ == "__main__":
    main()