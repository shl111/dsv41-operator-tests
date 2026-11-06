# mhc_pre_sinkhorn（MHC Pre + Sinkhorn）测试归档

DeepSeek-V4.1 推理链路中，MHC（Multi-Head Collaboration）前向用于**融合计算**：RMSNorm + 三路投影（pre/post/res）+ Sinkhorn 迭代归一化，输出 Attention/MLP 层输入与双随机残差矩阵。同一算子有多个来源，按 API 调用名分目录归档。

> 归档字段：算子 / 算子名称 / 具体算子名称（API 调用名）/ 输入 / 输入数据 / 输出 / 输出数据 / 性能 / 精度。精度 + 性能统一汇总见根目录 [`TEST_SUMMARY.md`](../TEST_SUMMARY.md)。

## 来源矩阵（Atlas A2 实测）

| 具体算子名称 | 来源 | A2 | 目录 |
| --- | --- | --- | --- |
| `cann_ops_transformer.mhc_pre_sinkhorn` | CANN 融合版（forward + backward） | **可运行** | [`cann_ops_transformer.mhc_pre_sinkhorn/`](cann_ops_transformer.mhc_pre_sinkhorn/) |
| `torch.ops._C_ascend.npu_hc_pre_v3` | vllm-ascend 自定义内核（**DSV4.1 生产实际调用**，`model.py:859`） | **可运行** | [`torch.ops._C_ascend.npu_hc_pre_v3/`](torch.ops._C_ascend.npu_hc_pre_v3/) |
| `torch_npu.npu_mhc_pre` / `npu_mhc_sinkhorn` / `npu_mhc_post` | torch_npu 原生分步版 | pre/sinkhorn **不可运行**（561103，Apt 后端专属）；post 可运行 | `test_mhc_source_matrix.py` |
| `mhc_pre_sinkhorn_backward` | CANN 反向算子（Autograd 自动调用） | **可运行** | [`cann_ops_transformer.mhc_pre_sinkhorn/`](cann_ops_transformer.mhc_pre_sinkhorn/test_backward.py) |
| HF transformers 参考 | `modeling_deepseek_v4.py:925-952`（torch 原生实现，非算子） | 参考语义（CPU/GPU torch） | - |

> **重要**：DSV4.1 生产代码实际调用 **vllm-ascend `_C_ascend.npu_hc_pre_v3`**（`model.py:859`），CANN `mhc_pre_sinkhorn` 是同语义另一实现。两者 `post`/`comb_frag`（Sinkhorn 矩阵）逐元素一致（diff ~1e-6），`y` 语义不同：**vllm `y`=Σ(pre·x)（与 HF 参考 `modeling_deepseek_v4.py:951` 一致）；CANN `hin`=x@H^pre^T**。性能 vllm 版更优（S=2048/C=1280：0.241ms vs 0.326ms，Event wall-time；kernel 同口径见下表）。

## 计算公式（参考实现推导，float32 对照）

```
x' = RMSNorm(x, eps=normEps)                          # 沿 (N,C) 两维
phi 拆分: pre=[0:N], post=[N:2N], res=[2N:N*N+2N]      # (N,N*C)/(N,N*C)/(N*N,N*C)
H^pre  = sigmoid(alpha[0]·(x'_flat·phi_pre^T) + b_pre)       # [B,S,N]
H^post = 2·sigmoid(alpha[1]·(x'_flat·phi_post^T) + b_post)   # [B,S,N]
H^res  = alpha[2]·(x'_flat·phi_res^T) + b_res          # [B,S,N*N] → [B,S,N,N]
H^res  = Sinkhorn(H^res, numIters=20)                  # 交替行/列归一化
h_in   = x @ H^pre^T                                  # [B,S,C]（x 原始，非 RMSNorm）
```

## 实测约束（Atlas A2）

| 规格 | 实测 | 说明 |
| --- | --- | --- |
| `numIters` | **必须 = 20** | 0/1/21 均报 `EZ1001`（文档写 ≤20，实测固定 20） |
| `N` | 4 | 文档固定 |
| `C` | 1280/1920/2560 推荐 | 实测 C=2048 也可运行（非硬约束） |
| 平台 | A2/A3 支持；A5(950) 不支持 | 文档产品支持表 |

## Sinkhorn 输出性质

- **列和 = 1.0**（`|col-1|` ≤ 1.2e-6，精确）
- **行和 ≈ 1**（`|row-1|` ≤ 0.056，20 次迭代收敛，随矩阵规模增大收敛性减弱）
- 这是 Sinkhorn 迭代的既定数值行为（未完全收敛到双随机），非精度错误

## 测试产物

| 具体算子名称 | 文件 | 运行 |
| --- | --- | --- |
| `cann_ops_transformer.mhc_pre_sinkhorn` | `cann_ops_transformer.mhc_pre_sinkhorn/test.py` | `python3 test.py --bench` / `--bench --msprof` |
| `cann_ops_transformer.mhc_pre_sinkhorn`（backward） | `cann_ops_transformer.mhc_pre_sinkhorn/test_backward.py` | `python3 test_backward.py --bench` / `--bench --msprof` |
| `torch.ops._C_ascend.npu_hc_pre_v3` | `torch.ops._C_ascend.npu_hc_pre_v3/test.py` | `python3 test.py --bench` / `--bench --msprof` |
| 来源矩阵（含 torch_npu 不可运行项） | `test_mhc_source_matrix.py` | `python3 test_mhc_source_matrix.py` |
| 全算子 msprof 一键采集 | `run_all_msprof.py` | `python3 run_all_msprof.py`（根目录，10 step 顺序执行） |

## msprof 精准测量（torch_npu.profiler, Level1 + PipeUtilization）

`op_statistic.csv` 官方权威 kernel 执行时间（Avg Time(us)，warmup=5 在 profiler 外 + active=10 在 profiler 内，MIX_AIC 核）：

| 算子 | shape | kernel avg | aic_mac_ratio | aiv_vec_ratio |
| --- | --- | --- | --- | --- |
| CANN `mhc_pre_sinkhorn`（forward） | S=1024/C=1280 | **184.506 us** | 0.078% | 0.657% |
| 同上 | S=2048/C=1280 | **322.052 us** | 0.118% | 0.744% |
| 同上 | S=2048/C=2560 | **373.692 us** | 0.143% | 0.710% |
| vllm `npu_hc_pre_v3` | S=1024/C=1280 | **131.654 us** | 0.221% | 0.600% |
| 同上 | S=2048/C=1280 | **237.770 us** | 0.267% | 0.646% |
| 同上 | S=2048/C=2560 | **384.394 us** | 0.333% | 0.646% |
| CANN `mhc_pre_sinkhorn_backward` | S=1024/C=1280 | **197.420 us** | 0.038% | 0.254% |
| 同上 | S=2048/C=1280 | **383.714 us** | 0.035% | 0.261% |

> **backward 统计口径（已修复）**：backward 的 msprof 采样的是 `loss.backward()`，`op_statistic` 中被测算子为 **`MhcPreSinkhornBackward`**（Autograd 自动调用 CANN 反向算子），非 forward 的 `MhcPreSinkhorn`；辅助 vector kernel（loss 计算侧 Mul/BroadcastTo/Add/TensorMove/Cast/Fill，各 1-15us）为 loss 图尾端开销，非 backward 主体。早期版本因 forward/backward 测试脚本共用 `./prof_msprof/S{S}_C{C}` 相对路径、os.walk 读到上一次运行残留产物，曾误将 forward 统计当作 backward——`msprof_bench` 现已在采集前 `rmtree` 清空 save_path，确保读到本次产物（kernel 名 `MhcPreSinkhornBackward`）。

结论（卡 7 全量复测）：
- **forward 主体 kernel 纯执行**：vllm `HcPre`（131.654-384.394us）全面优于 CANN `MhcPreSinkhorn`（184.506-373.692us），S=2048/C=1280 快 **26%**（237.770 vs 322.052）；S=2048/C=2560 接近（384.394 vs 373.692，CANN 略优 2.8%）。
- **backward 与 forward 同量级**（197.420/383.714 vs 184.506/322.052us），backward 略慢（多 7-19%）。
- **mac<0.5%**：pre+sinkhorn 为 memory/vector bound 融合算子（AIC 计算利用率极低），瓶颈在 HBM 带宽与 launch 开销。
- 统计口径：`op_statistic.csv` `Avg Time(us)` 为官方权威值；`kernel_details.csv` 仅用于 mac/vec 利用率参考。

## 判定口径

- **精度**：hPost/hRes 与 float32 参考：hres max diff < 5e-3；hpost 允许极少数点（≤1%）在 5e-3~2e-2 间（大 shape 下 kernel 量化噪声，mean ≤ 2e-4）；hin（bf16 输出）< 0.03（bf16 量化噪声）
- **Sinkhorn 性质**：列和精确 1，行和 ≈ 1（容差 0.08，收敛特性）
- **约束**：numIters 固定 20 报参数校验即符合预期；C 支持 1280~2560（含 2048）
- **性能**：单次毫秒级，随 S/C 线性增长；vllm-ascend 版优于 CANN 融合版
- **backward**：3 输出梯度路径独立（hin→α[0]、hpost→α[1]、hres→α[2]），4 梯度有限非 NaN
- 精度 + 性能统一汇总见根目录 [`TEST_SUMMARY.md`](../TEST_SUMMARY.md)
