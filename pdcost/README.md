# pdcost - PaddleFormers 分布式训练代价模型

`pdcost` 是一个用于预测 PaddleFormers 分布式训练性能的工具，可以在实际运行前估算不同并行配置下的：
- **Step 时间** (训练迭代耗时，已校准 seq_len 阈值效应)
- **显存占用** (支持双指标: allocated + reserved)
- **硬件利用率** (MFU)
- **训练吞吐量** (tokens/s/GPU)

## ✨ 特性亮点

- 🎯 **高精度预测**: Step time 误差 ~5%，显存误差 ~10%
- 📊 **双指标显存**: 同时预测 `allocated` (实际分配) 和 `reserved` (框架预留)
- 🔧 **seq_len 校准**: 内置阈值效应模型，准确处理不同序列长度
- 🔍 **配置搜索**: 自动搜索最优并行配置，支持 OOM 过滤
- ⚡ **MoE 专用**: 针对 Qwen3 MoE 等稀疏模型优化

## 📦 支持的并行策略

| 并行策略 | 参数 | 说明 |
|---------|------|------|
| Tensor Parallel (TP) | `tp` | 张量并行，切分 Attention 和 MLP 权重 |
| Pipeline Parallel (PP) | `pp` | 流水线并行，切分 Transformer 层 |
| Data Parallel (DP) | `dp` | 数据并行，复制模型 |
| Expert Parallel (EP) | `ep` | 专家并行，切分 MoE 专家 |
| Sharding (ZeRO) | `sharding` | 优化器状态/梯度/参数分片 |
| Sequence Parallel (SP) | `sp` | 序列并行，配合 TP 使用 |
| Context Parallel (CP) | `cp` | 上下文并行，切分序列长度 |

## 🚀 快速开始

### 基础用法

```python
from pdcost import PDCostModel, ModelConfig, ParallelConfig

# 1. 创建模型配置 (支持预设模型)
model_config = ModelConfig.from_name("qwen3-30b-a3b")

# 2. 创建代价模型
costmodel = PDCostModel(model_config)

# 3. 预测并行配置
parallel = ParallelConfig(tp=8, pp=1, dp=1, ep=8, sharding="stage1")
result = costmodel.predict(parallel, micro_batch_size=1, seq_len=8192)

# 4. 查看结果
print(f"Step Time: {result.step_time_ms:.2f} ms")
print(f"Memory: {result.memory_gb:.2f} GB")
print(f"MFU: {result.mfu:.1%}")
```

### 比较多个配置

```python
configs = [
    {"tp": 8, "pp": 1, "dp": 1, "ep": 8, "sharding": "stage1"},
    {"tp": 4, "pp": 2, "dp": 1, "ep": 4, "sharding": "stage1"},
    {"tp": 4, "pp": 1, "dp": 2, "ep": 4, "sharding": "stage2"},
]

# 自动排序并输出报告
best_configs = costmodel.rank_configurations(configs, top_k=5)
```

### 自动搜索最优配置

```python
# 生成搜索空间
configs = costmodel.generate_search_space(total_gpus=16, max_tp=8, max_pp=4)

# 搜索最优配置
best_configs = costmodel.rank_configurations(configs, top_k=10)
```

## 📊 支持的预设模型

| 模型名称 | 类型 | 参数量 | 说明 |
|---------|------|--------|------|
| `qwen3-30b-a3b` | MoE | ~30B | Qwen3 MoE, 128 experts, top-8 |
| `qwen3-235b-a22b` | MoE | ~235B | Qwen3 大模型 |
| `deepseek-v3` | MoE | ~685B | DeepSeek V3 |
| `llama3-70b` | Dense | ~70B | LLaMA 3 70B |
| `llama3-8b` | Dense | ~8B | LLaMA 3 8B |

## 🔧 硬件校准

pdcost 支持通过实际运行 benchmark 测试 GPU 算力和显存带宽，自动校准硬件参数，提高预测精度。

### 快速校准

```python
from pdcost import quick_calibrate

# 执行校准
result = quick_calibrate(device_id=0)
print(result)
# CalibrationResult:
#   GPU: NVIDIA H800 × 8
#   Memory: 79.6 GB
#   FP32: 51.6 TFLOPS
#   FP16: 763.0 TFLOPS
#   BF16: 788.0 TFLOPS
#   Memory BW: 2781.8 GB/s
```

### 自动校准创建 CostModel

```python
from pdcost import PDCostModel, ModelConfig

model_config = ModelConfig.from_name("qwen3-30b-a3b")

# 初始化时自动校准
costmodel = PDCostModel(model_config, auto_calibrate=True)

# 使用校准后的硬件参数进行预测
result = costmodel.predict(parallel)
```

### 手动校准

```python
costmodel = PDCostModel(model_config)

# 手动触发校准 (可指定 GEMM 矩阵大小加快测试)
costmodel.calibrate(gemm_size=4096)

# 查看校准结果
print(costmodel.calibration_result)
print(f"BF16 算力: {costmodel.hardware_config.gpu.bf16_tflops:.1f} TFLOPS")
```

### 校准参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `gemm_size` | GEMM 测试矩阵大小 | 8192 |
| `test_compute` | 是否测试算力 | True |
| `test_memory` | 是否测试显存带宽 | True |
| `verbose` | 是否打印进度 | True |

## 📁 模块结构

```
pdcost/
├── __init__.py          # 主入口
├── config.py            # 配置类 (ModelConfig, ParallelConfig, etc.)
├── memory_model.py      # 显存预测模型
├── compute_model.py     # 计算时间预测模型
├── comm_model.py        # 通信时间预测模型
├── calibration.py       # 硬件校准模块
├── costmodel.py         # 主 CostModel 类
├── README.md            # 文档
└── examples/
    └── basic_usage.py   # 使用示例
```

## 🔧 配置详解

### ModelConfig - 模型架构配置

```python
ModelConfig(
    num_hidden_layers=48,       # Transformer 层数
    hidden_size=6144,           # 隐藏维度
    intermediate_size=16384,    # FFN 中间维度
    num_attention_heads=32,     # 注意力头数
    num_key_value_heads=4,      # KV 头数 (GQA)
    head_dim=192,               # 每个头的维度
    num_experts=128,            # MoE 专家数
    num_experts_per_tok=8,      # Top-K
    moe_intermediate_size=1408, # 专家 FFN 维度
    vocab_size=152064,          # 词表大小
)
```

### ParallelConfig - 并行配置

```python
ParallelConfig(
    tp=8,                # 张量并行度
    pp=1,                # 流水线并行度
    dp=1,                # 数据并行度
    ep=8,                # 专家并行度
    sharding="stage1",   # ZeRO 阶段: none/stage1/stage2/stage3
    sp=False,            # 序列并行
    cp=1,                # 上下文并行度
)
```

### TrainingConfig - 训练配置

```python
TrainingConfig(
    micro_batch_size=1,              # 每卡 batch size
    sequence_length=8192,            # 序列长度
    gradient_accumulation_steps=64,  # 梯度累积
    dtype="bfloat16",                # 数据类型
    recompute_granularity="full",    # 重计算: none/selective/full
)
```

### HardwareConfig - 硬件配置

```python
HardwareConfig(
    gpu=GPUSpec.from_name("H100-80GB-HBM3"),  # GPU 规格
    num_nodes=1,                               # 节点数
    gpus_per_node=8,                           # 每节点 GPU 数
)
```

## 📈 预测函数参数

```python
result = costmodel.predict(
    parallel,                           # ParallelConfig: 并行配置
    micro_batch_size=1,                 # 每卡 batch size
    seq_len=8192,                       # 序列长度
    max_seq_len=8192,                   # 最大序列长度 (用于激活显存估算)
    gradient_accumulation_steps=64,     # 梯度累积步数
    recompute_granularity="full",       # 重计算粒度: "none", "selective", "full"
    tensorwise_offload_optimizer=False, # 是否启用 tensorwise 优化器 offload
    tensorwise_offload_ratio=0.95,      # offload 比例 (默认 95%)
)
```

### 关键参数说明

| 参数 | 说明 | 影响 |
|------|------|------|
| `max_seq_len` | 最大序列长度 | 影响激活显存估算（考虑动态 batch padding） |
| `recompute_granularity` | 重计算策略 | "none" 不重计算；"full" 全部重计算，激活显存最低 |
| `tensorwise_offload_optimizer` | Tensorwise 优化器 offload | 优化器状态按 tensor 粒度动态 offload 到 CPU |
| `tensorwise_offload_ratio` | Offload 比例 | 默认 0.95，即 95% 的优化器状态可被 offload |
| `split_param` | ShardingV2 参数分片 | Stage1 也分片参数和梯度（PaddleFormers 特有） |
| `sd_release_grads` | 释放梯度优化 | 每次迭代后释放梯度，显著降低峰值显存 |

### 显存优化示例

```python
# 不启用优化器 offload (默认)
result = costmodel.predict(parallel)
# 优化器显存: 36.58 GB

# 启用 tensorwise offload (95%)
result = costmodel.predict(parallel, tensorwise_offload_optimizer=True)
# 优化器显存: 1.83 GB

# 使用 max_seq_len 估算峰值激活显存
result = costmodel.predict(parallel, seq_len=4096, max_seq_len=8192)
# 激活显存按 max_seq_len=8192 估算
```

## 📊 预测结果 (PredictionResult)

```python
result = costmodel.predict(parallel)

# 时延指标
result.step_time_ms          # 总 step 时间 (ms)
result.compute_time_ms       # 计算时间 (ms)
result.total_comm_time_ms    # 通信时间 (ms)
result.bubble_time_ms        # 流水线气泡 (ms)

# 显存指标
result.memory_gb             # 总显存 (GB)
result.memory_breakdown      # 详细显存分解
result.fits_memory           # 是否满足显存约束

# 效率指标
result.mfu                   # Model FLOPs Utilization
result.compute_efficiency    # 计算效率

# 吞吐量
result.tokens_per_second     # 总吞吐量 (tok/s)
result.tokens_per_second_per_gpu  # 每卡吞吐量
```

## ?? 显存分解 (MemoryBreakdown)

```python
breakdown = result.memory_breakdown

# 主要组成
breakdown.parameter_memory_gb       # 参数显存
breakdown.gradient_memory_gb        # 梯度显存
breakdown.optimizer_memory_gb       # 优化器状态显存
breakdown.activation_memory_gb      # 激活值显存
breakdown.communication_buffer_gb   # 通信缓冲区
breakdown.temporary_buffer_gb       # 临时缓冲区
breakdown.framework_overhead_gb     # 框架基础开销 (CUDA/Paddle runtime)

# 双指标显存 (PaddleFormers 特有)
breakdown.allocated_memory_gb       # 实际分配显存 (nvidia-smi 中的 allocated)
breakdown.reserved_memory_gb        # 预留显存 (nvidia-smi 中的 reserved)
breakdown.activation_buffer_pool_gb # 框架激活缓冲池 (reserved - allocated)
breakdown.total_memory_gb           # 总显存 (等于 reserved)
```

### 双指标显存说明

PaddleFormers 框架有两个显存指标：
- **allocated**: 实际分配的显存，包括参数、梯度、优化器、激活等
- **reserved**: 框架预留的显存池，包括 allocated + 激活缓冲池

```python
# 预测双指标显存
result = costmodel.predict_calibrated(parallel, seq_len=8192, ...)
mb = result.memory_breakdown

print(f"预测 allocated: {mb.allocated_memory_gb:.2f} GB")  # ~43.4 GB
print(f"预测 reserved: {mb.reserved_memory_gb:.2f} GB")    # ~58.0 GB
```

## 🔍 配置搜索

### 搜索最优吞吐量配置

```python
from pdcost import ModelConfig, PDCostModel, ParallelConfig
from pdcost.config import TrainingConfig, HardwareConfig, GPUSpec

# 加载模型配置
model = ModelConfig.from_json('Qwen3-30B-A3B-Base/config.json')
hardware = HardwareConfig(
    gpu=GPUSpec(name='H800', memory_gb=79.6, bf16_tflops=788.0),
    num_nodes=1, gpus_per_node=8
)
training = TrainingConfig(micro_batch_size=1, sequence_length=8192, dtype='bfloat16')
costmodel = PDCostModel(model, hardware, training)

# 搜索最优配置
best_configs = costmodel.search_best_throughput(
    total_gpus=8,
    seq_len=8192,
    micro_batch_size=1,
    gradient_accumulation_steps=16,
    tensorwise_offload_optimizer=True,
    top_k=5
)

# 输出结果
for i, cfg in enumerate(best_configs):
    print(f"#{i+1}: {cfg['throughput']:.0f} tok/s/GPU, "
          f"tp={cfg['tp']}, pp={cfg['pp']}, dp={cfg['dp']}, ep={cfg['ep']}")
```

### 搜索空间约束

配置搜索会自动过滤无效配置：
- `tp * pp * dp == total_gpus` (GPU 数量约束)
- `ep <= num_experts` 且 `ep` 整除专家数
- 显存不超过 GPU 容量 (OOM 过滤)
- `tensorwise_offload` 需要 `dp > 1` (Sharding 约束)

## 📋 完整使用示例

### 从 YAML 配置预测

```python
from pdcost import ModelConfig, PDCostModel, ParallelConfig
from pdcost.config import TrainingConfig, HardwareConfig, GPUSpec

# 1. 加载模型配置
model = ModelConfig.from_json('Qwen3-30B-A3B-Base/config.json')

# 2. 配置硬件 (H800 8卡)
hardware = HardwareConfig(
    gpu=GPUSpec(name='H800', memory_gb=79.6, bf16_tflops=788.0),
    num_nodes=1, 
    gpus_per_node=8
)

# 3. 训练配置
training = TrainingConfig(
    micro_batch_size=1, 
    sequence_length=8192, 
    dtype='bfloat16'
)

# 4. 创建代价模型
costmodel = PDCostModel(model, hardware, training)

# 5. 配置对应 benchmark_config.yaml
# tp=1, pp=1, dp=8, ep=8, seq=8192, mbs=1, gas=16
parallel = ParallelConfig(tp=1, pp=1, dp=8, ep=8, sharding='stage1')

# 6. 预测 (使用校准后的模型)
result = costmodel.predict_calibrated(
    parallel, 
    seq_len=8192, 
    micro_batch_size=1, 
    gradient_accumulation_steps=16,
    tensorwise_offload_optimizer=True, 
    tensorwise_offload_ratio=0.95
)

# 7. 输出结果
print(f"Step Time: {result.step_time_ms/1000:.2f} 秒")
print(f"吞吐量: {result.tokens_per_second_per_gpu:.0f} tok/s/GPU")
print(f"Allocated 显存: {result.memory_breakdown.allocated_memory_gb:.2f} GB")
print(f"Reserved 显存: {result.memory_breakdown.reserved_memory_gb:.2f} GB")
print(f"可运行: {'✅' if result.fits_memory else '❌ OOM'}")
```

### 预测精度参考

在 Qwen3-30B-A3B + H800 8卡环境下的预测精度：

| 指标 | 预测误差 |
|------|----------|
| Step Time | ~5% |
| 吞吐量 (tok/s/GPU) | ~5% |
| Allocated 显存 | ~10% |
| Reserved 显存 | ~8% |

### seq_len 阈值效应

pdcost 内置了 seq_len 对 step time 的阈值效应校准：

```python
# seq_len <= 2048: 基础效率 ~15%
# seq_len > 2048: 效率随 seq_len 线性增长
# 例如 seq_len=8192 时效率可达 ~60%

# 这个效应会自动在 predict_calibrated() 中考虑
result = costmodel.predict_calibrated(parallel, seq_len=8192, ...)
```

## 🎯 使用建议

1. **MoE 模型**: 优先使用 EP 并行，通常 `ep = min(num_experts, total_gpus)`
2. **显存不足**: 尝试增加 Sharding 阶段 (`stage2` → `stage3`) 或开启重计算
3. **大序列长度**: 考虑使用 Context Parallel (CP) 或 Sequence Parallel (SP)
4. **多节点训练**: PP 适合跨节点，TP 建议节点内使用

## 📝 运行示例

```bash
cd pdcost
python examples/basic_usage.py
```

## ⚠️ 注意事项

- 预测结果为理论估算值，实际性能受多种因素影响
- 建议在少量配置上进行实际 benchmark 验证
- 通信时间预测假设理想的网络条件
- **建议使用 `auto_calibrate=True` 获取更准确的硬件参数**

## 📖 API 参考

### PDCostModel

```python
PDCostModel(
    model_config,           # ModelConfig: 模型架构配置
    hardware_config=None,   # HardwareConfig: 硬件配置 (默认 H100-80GB)
    training_config=None,   # TrainingConfig: 训练配置
    auto_calibrate=False,   # bool: 是否自动校准硬件
)
```

**主要方法:**
- `predict(parallel, ...)` - 预测并行配置性能
- `calibrate(...)` - 执行硬件校准
- `rank_configurations(configs, ...)` - 配置排序
- `generate_search_space(total_gpus, ...)` - 生成搜索空间

### HardwareCalibrator

```python
from pdcost import HardwareCalibrator

calibrator = HardwareCalibrator(device_id=0)
result = calibrator.calibrate()
hw_config = calibrator.create_hardware_config()
```

### 便捷函数

```python
from pdcost import quick_calibrate, create_calibrated_hardware_config

# 快速校准
result = quick_calibrate()

# 创建校准后的硬件配置
hw_config = create_calibrated_hardware_config(num_nodes=1, gpus_per_node=8)
```