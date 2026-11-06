# mhc_post（MHC Post 后处理）测试归档

DeepSeek-V4.1 推理链路中，`mhc_post` 用于 MHC 层**后处理**：将残差矩阵变换与输出状态投影融合为单次计算。同一算子有 **3 个来源**，按 API 调用名分目录归档。

> 归档字段：算子 / 算子名称 / 具体算子名称（API 调用名）/ 输入 / 输入数据 / 输出 / 输出数据 / 性能 / 精度。精度 + 性能统一汇总见根目录 [`TEST_SUMMARY.md`](../TEST_SUMMARY.md)。

## 来源矩阵（Atlas A2 实测，均可运行）

| 具体算子名称 | 来源 | A2 | 目录 |
| --- | --- | --- | --- |
| `cann_ops_transformer.mhc_post` | CANN 融合版 | **可运行** | [`cann_ops_transformer.mhc_post/`](cann_ops_transformer.mhc_post/) |
| `torch_npu.npu_mhc_post` | torch_npu 原生 | **可运行** | [`torch_npu.npu_mhc_post/`](torch_npu.npu_mhc_post/) |
| `torch.ops._C_ascend.npu_hc_post` | vllm-ascend 自定义内核（**DSV4.1 生产实际调用**，`model.py:872`） | **可运行** | [`torch.ops._C_ascend.npu_hc_post/`](torch.ops._C_ascend.npu_hc_post/) |

## 计算公式（CANN 文档）

```
y[i,:] = H^post[i]·h_out[:] + Σ_j H^res[j,i]·x[j,:]
```

即 `y = post·h_out + comb^T·residual`（残差流经 comb 转置加权 + collapsed 流经 post 加权）。

## 布局差异

| 来源 | 调用 | 输入布局 |
| --- | --- | --- |
| CANN / torch_npu | `mhc_post(x, h_res, h_out, h_post)` | x `(B,S,N,D)` 残差流；h_res `(B,S,N,N)`；h_out `(B,S,D)` collapsed；h_post `(B,S,N)` |
| vllm `npu_hc_post` | `npu_hc_post(x, residual, post, comb)` | x `(B,S,D)` collapsed；residual `(B,S,N,D)` hc 流；post `(B,S,N)`；comb `(B,S,N,N)` |

参数映射：CANN `x`=vllm `residual`；CANN `h_res`=vllm `comb`；CANN `h_out`=vllm `x`；CANN `h_post`=vllm `post`。

## 精度 / 性能结论

| 具体算子名称 | vs float32 参考 | S=1024 | S=2048 |
| --- | --- | --- | --- |
| `cann_ops_transformer.mhc_post` | 0.026（BSND）/ 0.030（TND） | 0.092 ms | 0.095 ms |
| `torch_npu.npu_mhc_post` | 0.026 | 0.075 ms | 0.077 ms |
| `torch.ops._C_ascend.npu_hc_post` | 0.015 | **0.045 ms** | **0.046 ms** |

三来源同输入输出**逐元素一致**（max diff = 0.0，见各子目录 README）。vllm-ascend 版最快（约 2 倍于原生），且为 DSV4.1 生产实际调用。

## msprof 精准测量（torch_npu.profiler, Level1 + PipeUtilization）

Event wall-time 含 host launch / 排队开销；msprof `op_statistic.csv` 给**官方权威 kernel 执行时间**（Avg Time(us)，CANN 9.1 逐 OP Type 聚合，warmup=5 在 profiler 外 + active=10 在 profiler 内）：

| 具体算子名称 | Event wall-time | kernel avg（S=1024） | kernel avg（S=2048） | aic_mac_ratio | aiv_vec_ratio |
| --- | --- | --- | --- | --- | --- |
| `cann_ops_transformer.mhc_post` | 0.092 / 0.095 ms | **27.456 us** | **48.536 us** | 0% | 0.389% |
| `torch_npu.npu_mhc_post` | 0.075 / 0.077 ms | **27.112 us** | **48.408 us** | 0% | 0.396% |
| `torch.ops._C_ascend.npu_hc_post` | 0.045 / 0.046 ms | **22.700 us** | **39.452 us** | 0% | 0.512% |

结论：
- **三来源 kernel 纯执行时间同量级（22-48us）**，Event wall-time 的差异（45-95us）主要来自 host launch 开销（vllm 内核 launch 更轻），非 kernel 计算差异。
- **mac=0%**：三来源均为**纯 vector/memory bound 算子**（无 cube matmul），`vec~0.4-0.5%` 表明计算利用率极低，性能瓶颈在 HBM 带宽 / launch 开销，而非 AIC 计算。
- **统计口径**：`op_statistic.csv` 的 `Avg Time(us)` 是官方权威口径（按 OP Type 聚合 min/max/avg/total/count/core）；`kernel_details.csv`（kernel 名带 hash 后缀）只用于 mac/vec 利用率参考，手算其 Duration 均值会与官方口径有偏差。
- 精准测量命令：`python3 test.py --bench --msprof`（三来源各自内联 msprof 到 test.py；`--msprof` 触发 warmup+active+op_statistic 统计，产物 `./prof_msprof/S{S}_C{C}/.../ASCEND_PROFILER_OUTPUT/`）。

## 测试产物

| 具体算子名称 | 文件 | 运行 |
| --- | --- | --- |
| `cann_ops_transformer.mhc_post` | `cann_ops_transformer.mhc_post/test.py` | `python3 test.py --bench` / `--bench --msprof` |
| `torch_npu.npu_mhc_post` | `torch_npu.npu_mhc_post/test.py` | `python3 test.py --bench` / `--bench --msprof` |
| `torch.ops._C_ascend.npu_hc_post` | `torch.ops._C_ascend.npu_hc_post/test.py` | `python3 test.py --bench` / `--bench --msprof` |
| 全算子 msprof 一键采集 | `run_all_msprof.py` | `python3 run_all_msprof.py`（根目录，10 step 顺序执行） |
