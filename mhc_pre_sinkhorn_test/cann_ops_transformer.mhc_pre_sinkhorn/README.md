# cann_ops_transformer.mhc_pre_sinkhorn

MHC Pre + Sinkhorn 融合前向算子（CANN 融合版）。

| 字段 | 内容 |
| --- | --- |
| 算子 | MHC Pre + Sinkhorn |
| 算子名称 | `cann_ops_transformer.mhc_pre_sinkhorn` |
| 具体算子名称 | `cann_ops_transformer.ops.mhc_pre_sinkhorn`（`docs/zh/mhc_pre_sinkhorn.md`，Autograd 封装） |
| 输入 | `x` `(B,S,N,C)` bf16；`phi` `(N*N+2N, N*C)` fp32；`alpha` `(3)` fp32；`bias` `(N*N+2N)` fp32；`hcMult`、`numIters=20` |
| 输入数据 | torch.randn（bf16/fp32） |
| 输出 | `hin` `(B,S,C)` bf16；`hPost` `(B,S,N)` fp32；`hRes` `(B,S,N*N)` fp32 |
| 输出数据 | `hin=x@H^pre^T`；`hPost=2·sigmoid`；`hRes=Sinkhorn(H^res)`（列和=1，行和≈1） |
| 性能 | S=1024/C=1280 0.186ms；S=2048 0.325ms；C=2560 0.379ms（A2，Event wall-time）；kernel avg 见下 |
| 精度 | hPost/hRes vs float32 参考：hres<5e-3；hpost 噪声点占比≤1%（mean≤2e-4，max≤2e-2）；hin（bf16）<0.03 |

## 运行

```bash
python3 test.py            # 精度(3 shape) + 约束(6 项)
python3 test.py --bench    # 追加性能
python3 test.py --bench --msprof   # msprof 精准测 kernel avg + 利用率
python3 test_backward.py [--bench] [--msprof]   # 反向算子梯度路径验证 + 精准测量
```

## msprof 精准测量（op_statistic.csv 官方 Avg Time(us)，warmup5+active10）

| 方向 | shape | kernel avg | mac | vec |
| --- | --- | --- | --- | --- |
| forward `MhcPreSinkhorn` | S=1024/C=1280 | **184.506 us** | 0.078% | 0.657% |
| forward `MhcPreSinkhorn` | S=2048/C=1280 | **322.052 us** | 0.118% | 0.744% |
| forward `MhcPreSinkhorn` | S=2048/C=2560 | **373.692 us** | 0.143% | 0.710% |
| backward `MhcPreSinkhornBackward` | S=1024/C=1280 | **197.420 us** | 0.038% | 0.254% |
| backward `MhcPreSinkhornBackward` | S=2048/C=1280 | **383.714 us** | 0.035% | 0.261% |

backward 侧额外包含 loss 图尾端 vector kernel（Mul/BroadcastTo/Add/TensorMove/Cast/Fill，各 1-15us，非 backward 主体）。backward 采样对象为 `loss.backward()`，`op_statistic` 中为 `MhcPreSinkhornBackward` 而非 forward 的 `MhcPreSinkhorn`（早期曾因产物目录复用读到 forward 统计，已修复：msprof_bench 采集前 rmtree save_path）。本表为卡 7 全量复测数据。

## 约束

- `numIters` 必须 = 20（0/1/21 报 `EZ1001`）
- `N`=4；`C`∈{1280,1920,2560}（实测 2048 也可）
- A2/A3 支持；A5(950) 不支持

## 判定口径

- hPost/hRes vs float32 参考：**hres max diff < 5e-3；hpost 允许极少数点（≤1%）在 5e-3~2e-2 间（大 shape 下 kernel 量化噪声，mean ≤ 2e-4）**（S=1024/C=1280 卡 7 复测 hpost max 0.0014、mean 6.72e-5、>5e-3 占比 0.00%，vllm 版同输入 2.5e-6 精确）
- hRes Sinkhorn：列和精确 1，行和 ≈ 1（容差 0.08）
- backward：3 输出梯度路径独立（hin→α[0]、hpost→α[1]、hres→α[2]），4 梯度有限非 NaN
- 精度 + 性能统一汇总见根目录 [`TEST_SUMMARY.md`](../../TEST_SUMMARY.md)
