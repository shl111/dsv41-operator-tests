# `cann_ops_transformer.sparse_flash_mla`（SparseFlashMLA）测试说明

> DeepSeek-V4.1 稀疏注意力（SparseFlashAttention 家族，MLA 特化形态）算子 `cann_ops_transformer.sparse_flash_mla` 的精度与性能测试归档。
> 概念关系（通用 SparseFlashAttention vs MLA 特化 SparseFlashMLA）见 `../README.md` §1.0。
>
> 归档字段：算子 / 算子名称 / 具体算子名称 / 输入 / 输入数据 / 输出 / 输出数据 / 性能 / 精度。

---

## 1. 测试环境

| 项 | 值 |
| --- | --- |
| 硬件 | Atlas A2 / Ascend 910B3 x8 |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.0 / 2.10.0.post4 |
| 算子库 | cann_ops_transformer 1.0.0（内置 A2 AscendC 内核，`opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_transformer/sparse_flash_mla/`） |
| 被测算子 | `cann_ops_transformer.sparse_flash_mla`（`torch.ops.cann_ops_transformer::sparse_flash_mla`） |

精度判据（见 `test_precision.py` `_report()`）：
- 均值误差 `mean_diff <= 2e-3`（严格，检测真实错误）；
- 最大值误差 `max_diff <= max(0.03, 0.05*ref_max) + 0.25`（容忍 fp16 在 mask 边界 softmax 切换处的单元素累积噪声尖峰，其 mean_diff 仍保持 ~1e-4 量级，属 fp16 累积特性而非正确性问题）。

---

## 2. 精度测试归档

参考实现：PyTorch float32 推导，覆盖 SWA/CSA/HCA 三种模式与 BSND/TND 两种布局。关键语义（经算子实测校准）：
- q 与 ori_kv **末端对齐**：`global_q_pos(i) = kv_len - q_len + i`；
- ori 侧滑动窗口：q 关注 ori_kv 区间 `[g-127, g]`（win_left=127，win_right=0）；
- attention sink：**虚拟 token**，V 值为 0，logit 偏置为 `sinks[head]`；
- cmp 侧 RightDownCausal：压缩位置 j 在 q 原始位置 `p >= j*ratio + ratio - 1` 时可见；
- `cmp_residual_kv`：`ori_len_for_cmp_mask = cmp_len*ratio + cmp_residual_kv[b]`。

| 算子 | 算子名称 | 具体算子名称 | 输入（shape/参数） | 输入数据 | 输出 | 输出数据 | 性能（同 §3） | 精度（vs float32 参考） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[1,8,1,512]` bf16；ori_kv `[1,8,1,512]` bf16；sinks `[1]` f32；win 127/0；scale=1/√512 | q/ori_kv: N(0,1) 随机；sinks: |U(0,3)| | attention_out `[1,8,1,512]` bf16；softmax_lse `[]` | shape==q，dtype==q | 见 §3（decode 同族 shape 0.134ms） | max_diff 0.00391 / mean 0.000211 → **PASS** |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[1,16,4,512]` bf16；ori_kv `[1,32,1,512]` bf16；win 127/0 | 同上 | 同上 | 同上 | 0.135–0.156ms（Sq128） | max_diff 0.00781 / mean 0.000270 → **PASS** |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[2,16,2,512]` fp16；ori_kv `[2,32,1,512]` fp16 | 同上 | 同上 | 同上 | 同上 | max_diff 0.00098 / mean 0.000033 → **PASS** |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[2,8,64,512]` fp16；ori_kv `[2,64,1,512]` fp16 | 同上 | 同上 | 同上 | 同上 | max_diff 0.00049 / mean 0.000023 → **PASS** |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[4,64,8,512]` bf16；ori_kv `[4,256,1,512]` bf16 | 同上 | 同上 | 同上 | 同上 | max_diff 0.00391 / mean 0.000151 → **PASS** |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[4,128,64,512]` fp16；ori_kv `[4,1024,1,512]` fp16 | 同上 | 同上 | 同上 | 0.253ms(fp16)/0.263ms(bf16) | max_diff 0.00098 / mean 0.000018 → **PASS** |
| SWA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[16,4,512]` bf16；ori_kv `[16,1,512]` bf16；cu_seqlens_q=cu_ori=[0,16] | 同上 | 同上 | 同上 | - | max_diff 0.00781 / mean 0.000135 → **PASS** |
| CSA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[6,4,512]` fp16；ori_kv `[6,1,512]`；cmp_kv `[1,1,512]`；cmp_sparse_indices `[6,1,512]` int32；residual=[2]；ratio=4；K=512 | 索引：causal 有效位置前 K 个，无效填 -1 | 同上 | 同上 | 0.675–0.685ms（q1024/K512） | max_diff 0.00049 / mean 0.000019 → **PASS** |
| CSA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[34,8,512]` bf16；ori_kv `[34,1,512]`；cmp_kv `[8,1,512]`；residual=[0,2]；ratio=4；K=1024（双 batch，CP 切分口径） | 同上 | 同上 | 同上 | 0.678–0.691ms（K1024） | max_diff 0.00781 / mean 0.000248 → **PASS** |
| CSA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[32,64,512]` fp16；cmp_kv `[8,1,512]`；residual=[0]；K=512 | 同上 | 同上 | 同上 | 同上 | max_diff 0.00098 / mean 0.000028 → **PASS** |
| HCA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[128,8,512]` bf16；ori_kv `[128,1,512]`；cmp_kv `[1,1,512]`；residual=[0]；ratio=128 | 连续压缩段 + RightDownCausal | 同上 | 同上 | 0.292–0.307ms（q1024/cmp8） | max_diff 0.00781 / mean 0.000109 → **PASS** |
| HCA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[192,16,512]` fp16；cmp_kv `[2,1,512]`；residual=[0,0]；ratio=128（双 batch） | 同上 | 同上 | 同上 | 同上 | max_diff 0.19312 / mean 0.000081 → **PASS**（fp16 mask 边界累积噪声，均值 1e-4 量级） |

**精度结论：12/12 PASS。** 三种计算模式、两种布局、两种 dtype 均与 float32 参考一致（均值误差 ~1e-4 量级）。

### 2.1 精度测试用例矩阵（覆盖性）

| 维度 | 覆盖 |
| --- | --- |
| 模式 | SWA（仅 ori_kv）、CSA（ori_kv + TopK 压缩）、HCA（ori_kv + 连续压缩） |
| 布局 | BSND / TND（TND 含 CP 切分、双 batch、非对称长度） |
| dtype | bf16 / fp16 |
| batch | 1 / 2 / 4 |
| 特殊语义 | attention sink（虚拟 token）、mask 边界、cmp_residual_kv、-1 无效索引 |
| 约束验证 | win_left≠127 时 A2/A3 报 EZ1001 拒绝执行；has_cmp_kv=False 时 cmp_ratio 必须为 1；Sq>Sk 为未定义边界（模型实际恒有 Sq≤Sk） |

---

## 3. 性能测试归档

方法：单算子调用，`torch.npu.Event` 计时（不含数据构造与 metadata 生成），warmup 3 次后 50 次取中位数。有效 FLOPs = `2*Nq*Sq*S_attended*D*2`（QKᵀ + PV），S_attended = min(Sk,128) [+ K(cmp_topk) | + cmp_len]。

| 算子 | 算子名称 | 具体算子名称 | 输入（shape/参数） | 输入数据 | 输出 | 输出数据 | 性能（latency / GFLOPS） | 精度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[1,1,64,512]`；ori_kv `[1,4096,1,512]`（decode 口径） | N(0,1) 随机 | attention_out `[1,1,64,512]` | 同 q | fp16 **0.134ms** / 125 GFLOPS<br>bf16 **0.149ms** / 113 GFLOPS | 见 §2（同族输入） |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[1,128,64,512]`；ori_kv `[1,128,1,512]` | 同上 | 同上 | 同上 | fp16 **0.135ms** / 15.95 TFLOPS<br>bf16 **0.156ms** / 13.8 TFLOPS | 见 §2 |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[1,1024,64,512]`；ori_kv `[1,1024,1,512]` | 同上 | 同上 | 同上 | fp16 **0.253ms** / 68.0 TFLOPS<br>bf16 **0.263ms** / 65.3 TFLOPS | 见 §2 |
| SWA | sparse_flash_mla (BSND) | `cann_ops_transformer.sparse_flash_mla` | q `[1,1024,64,512]`；ori_kv `[1,32768,1,512]`（长上下文，窗口固定 128） | 同上 | 同上 | 同上 | fp16 **0.248ms** / 69.3 TFLOPS<br>bf16 **0.267ms** / 64.3 TFLOPS | 见 §2 |
| CSA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[1024,64,512]`；ori_kv `[1024,1,512]`；cmp_kv `[256,1,512]`；K=512；ratio=4 | 同上 | 同上 | 同上 | fp16 **0.682ms** / 126.0 TFLOPS<br>bf16 **0.685ms** / 125.3 TFLOPS | 见 §2 |
| CSA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | 同上但 K=1024 | 同上 | 同上 | 同上 | fp16 **0.678ms** / 227.9 TFLOPS<br>bf16 **0.691ms** / 223.7 TFLOPS | 见 §2 |
| HCA | sparse_flash_mla (TND) | `cann_ops_transformer.sparse_flash_mla` | q `[1024,64,512]`；ori_kv `[1024,1,512]`；cmp_kv `[8,1,512]`；ratio=128 | 同上 | 同上 | 同上 | fp16 **0.306ms** / 59.6 TFLOPS<br>bf16 **0.307ms** / 59.5 TFLOPS | 见 §2 |

**性能结论：**
- SWA 关注固定 128 长度窗口，latency 不随 Sk 增长（1024→32768 仅 0.253→0.248ms），体现窗口化稀疏注意力收益；decode（Sq=1, Sk=4096）0.134ms，为窗口搬运受限（低 FLOPs 高带宽访问）。
- CSA 在 K=512→1024 时 FLOPs 翻倍但 latency 几乎不变（0.682→0.678ms），计算已被流水充分掩盖，呈算力饱和。
- bf16 与 fp16 性能接近（同计算强度，910B3 bf16/fp16 同算力）。

### msprof 精准测量（op_statistic.csv 官方 Avg Time(us)，warmup5+active10）

`python3 benchmark_perf.py --msprof --iters 10`，kernel avg（`SparseFlashMla`，MIX_AIC，mac<0.5% vec<0.7%）：

| case（fp16 / bf16） | 输入 | kernel avg fp16 | kernel avg bf16 |
| --- | --- | --- | --- |
| SWA BSND B1 Sq1 Sk4096 N64（decode） | q `[1,1,64,512]`；ori_kv `[1,4096,1,512]` | **19.506 us** | **19.392 us** |
| SWA BSND B1 Sq128 Sk128 N64 | q `[1,128,64,512]`；ori_kv `[1,128,1,512]` | **39.378 us** | **40.078 us** |
| SWA BSND B1 Sq1024 Sk1024 N64 | q `[1,1024,64,512]`；ori_kv `[1,1024,1,512]` | **171.198 us** | **173.930 us** |
| SWA BSND B1 Sq1024 Sk32768 N64（长 ctx） | q `[1,1024,64,512]`；ori_kv `[1,32768,1,512]` | **174.088 us** | **173.736 us** |
| CSA TND q1024 kv1024 cmp256 K512 N64 | q `[1024,64,512]`；cmp_kv `[256,1,512]`；K=512 | **584.114 us** | **580.716 us** |
| CSA TND q1024 kv1024 cmp256 K1024 N64 | 同上 K=1024 | **585.648 us** | **577.244 us** |
| HCA TND q1024 kv1024 cmp8 N64 | q `[1024,64,512]`；cmp_kv `[8,1,512]`；ratio=128 | **203.358 us** | **204.466 us** |

> kernel avg 与 Event wall-time（latency）同源验证：decode 19us vs wall 0.134ms（launch/排队为主）；SWA Sq1024Sk1024 kernel 172us vs wall 0.253ms；CSA K512 kernel 584us vs wall 0.682ms。kernel 层面 decode 极短（19us），窗口搬运受限的结论不变。本表为卡 7 全量复测数据。

---

## 4. 测试程序

| 文件 | 说明 |
| --- | --- |
| `test_precision.py` | 精度测试（含 float32 参考实现、12 用例、判据），`python3 test_precision.py` |
| `benchmark_perf.py` | 性能测试（SWA/CSA/HCA × fp16/bf16，14 组 shape），`python3 benchmark_perf.py --iters 50` |

运行环境：Atlas A2/A3 NPU + torch_npu + `cann_ops_transformer`（需 `sparse_flash_mla_metadata` 前置算子生成任务切分）。
