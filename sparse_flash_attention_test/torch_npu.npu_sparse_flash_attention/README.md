# `torch_npu.npu_sparse_flash_attention` 测试说明

> DeepSeek-V4.1 稀疏注意力（SparseFlashAttention 家族，通用离散 TopK 形态）算子 `torch_npu.npu_sparse_flash_attention` 的本机跑通、tiling 失败根因、语义与性能验证归档。
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
| 被测算子 | `torch_npu.npu_sparse_flash_attention`（`torch.ops.npu.npu_sparse_flash_attention`，注册 `torch_npu/utils/custom_ops.py:50`） |
| 内核 | A2（910B3）为 **MLA 内核**：`opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/sparse_flash_attention/arch22/sparse_flash_attention_kernel_mla.h` |

---

## 2. tiling 失败根因（561002 = `ACLNN_ERR_INNER_TILING_ERROR`）

初始所有配置（D=128 等）报 `Check NnopbaseRunForWorkspace failed`，aclnn 错误码 **561002 = `ACLNN_ERR_INNER_TILING_ERROR`**（`aclnn/opdev/op_errno.h:20`）。经对照 A2 内核源码与 tiling 校验信息逐项定位，A2 上该算子为 **MLA 内核**，存在 4 条硬约束：

| 约束 | 说明 | 证据 |
| --- | --- | --- |
| **head_dim 必须 = 512、kv 头 = 1** | MLA 内核硬编码 `headDim = 512 / headDimAlign = 512 / headDimRope = 64 / kvHeadNum = 1`；D=128 等配置 tiling 必失败 | `sparse_flash_attention_kernel_mla.h:99-102` |
| **query_rope / key_rope 必须传真实张量** | 传 None 时报 `EZ0037 Invalid_Argument: Parameter query_rope ... Shape of tensor queryRope is nullptr` | 运行时 EZ 错误日志 |
| **pre_tokens / next_tokens 必须 = 9223372036854775807（INT64_MAX）** | 传 128 报 `EZ0026 Invalid_Argument: Parameter pre_tokens ... preTokens must be 9223372036854775807` | 运行时 EZ 错误日志 |
| **attention_mode 必须 = 2** | `attention_mode=0/1` tiling 失败，`attention_mode=2`（连续 decode）通过 | 实测矩阵 |

**本机可运行配置（生产范式，与 vllm-ascend `AscendSFAImpl` 调用一致）：**
```
layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
pre_tokens=next_tokens=INT64_MAX, scale_value=1/sqrt(512), sparse_block_size=1,
query_rope [Tq, N, 64], key_rope [block_num, block_size, 1, 64]
```
BSND 布局 + attention_mode=2 亦可运行，但输出含滑动窗口 band 语义，与下述稀疏-only 语义不同（未作为生产路径验证）。

---

## 3. 已验证语义（浮点参考对照）

与 DSV4.1 离散 TopK 稀疏注意力一致：
- `score(i,j) = (q_nope[i]·k[j] + q_rope[i]·k_rope[j]) / sqrt(512)`（rope 项参与打分）；
- 每 query 只对 `sparse_indices[i]` 中的有效 KV 位置做 softmax；
- 输出 `attention_out` 为加权 V 的 nope 部分，shape `[Tq, N, D]`（D=512，不含 rope 位）。

**精度实测（9 用例套件，vs float32 参考，TND+PA_BSND+attention_mode=2，判据 max≤0.03 ∧ mean≤2e-3）：**

| 用例（label, Tq/Tkv/SS/dtype） | max_diff | mean_diff | 结果 |
| --- | --- | --- | --- |
| SWA decode SS=8 bf16（1/128/8） | 0.00777 | 0.001172 | **PASS** |
| SWA decode SS=128 bf16（1/4096/128） | 0.00777 | 0.001172 | **PASS** |
| SWA decode SS=128 bf16 long（1/32768/128） | 0.00777 | 0.001172 | **PASS** |
| SWA prefill SS=8 bf16（64/128/8） | 0.01226 | 0.001225 | **PASS** |
| SWA prefill SS=32 bf16（256/256/32） | 0.01228 | 0.000767 | **PASS** |
| SWA prefill SS=128 bf16（1024/1024/128） | 0.01434 | 0.000454 | **PASS** |
| SWA prefill SS=128 bf16 long（1024/32768/128） | 0.00989 | 0.000453 | **PASS** |
| SWA prefill SS=8 fp16（64/128/8） | 0.00157 | 0.000149 | **PASS** |
| SWA prefill SS=128 fp16（1024/1024/128） | 0.00172 | 0.000057 | **PASS** |

9/9 PASS。decode max_diff 恒定 0.00777（稀疏位置索引固定，仅 1 query）；prefill 随 SS 增大 max_diff 略升（更多 softmax 候选，mean 反而下降）；fp16 精度优于 bf16（fp16 中间计算精度更高）。

**本机性能（910B3，`torch.npu.Event`，N=1 头）：**

| 场景 | 输入 | Event wall-time | msprof kernel avg | mac/vec |
| --- | --- | --- | --- | --- |
| decode（SS=8） | q `[1,1,512]`；kv PA_BSND `[1,128,1,512]`；Tkv=128 | 0.0887ms | **16.246 us** | 0.002% / 0.006% |
| decode（SS=128） | q `[1,1,512]`；kv PA_BSND `[32,128,1,512]`；Tkv=4096 | 0.0895ms | **18.390 us** | 0.003% / 0.006% |
| decode 长上下文 | 同上 Tkv=32768 | 0.0880ms | **19.266 us** | 0.003% / 0.006% |
| prefill（SS=8） | q `[64,1,512]`；kv PA_BSND `[1,128,1,512]`；Tkv=128 | 0.0877ms | **29.280 us** | 0.049% / 0.053% |
| prefill（SS=32） | q `[256,1,512]`；kv PA_BSND `[2,128,1,512]`；Tkv=256 | 0.0886ms | **67.544 us** | 0.076% / 0.078% |
| prefill（SS=128） | q `[1024,1,512]`；kv PA_BSND `[8,128,1,512]`；Tkv=1024 | 0.2730ms | **268.440 us** | 0.117% / 0.083% |
| prefill 长上下文 | 同上 Tkv=32768 | 0.2732ms | **269.182 us** | 0.116% / 0.081% |
| prefill（SS=128 fp16） | 同 SS=128 但 fp16 | 0.2735ms | **269.634 us** | 0.117% / 0.082% |

结论：kernel avg 极短（decode 16-19us、prefill SS=8 29us），wall-time 远大于 kernel（launch/排队为主，decode 占比 >80%）；latency 不随 Tkv 增长（离散 TopK 优势），随 SS（候选数）近线性增长（29→68→268us）；mac<0.15% → memory/vector bound。本表 msprof 为卡 7 全量复测数据，8 case 全表见根目录 `TEST_SUMMARY.md`。

---

## 4. 测试程序

| 文件 | 说明 |
| --- | --- |
| `test.py` | torch_npu 原生 `npu_sparse_flash_attention` 套件测试（9 精度用例 + 8 性能 case）：`python3 test.py --check-precision`（精度）；`python3 test.py --bench`（Event wall-time）；`python3 test.py --bench --msprof`（kernel avg + 利用率）；默认无参数全跑 |

运行环境：Atlas A2 NPU + torch_npu（`npu_sparse_flash_attention` 已随 CANN 分发）。

---

## 5. 与其它同族算子的关系

- vllm-ascend `_C_ascend.npu_sparse_flash_attention`（打包自定义算子）本环境**未启用（可编译）**：其底层 `aclnnSparseFlashAttention` 调用范式（TND/PA_BSND/attention_mode=2）与本测试完全一致，已由本算子验证（详见 `../README.md` §1.4）。
- `mixed_quant_sparse_flash_mla`（ds41 命名空间，DSV4.1 生产量化算子）需 A5 + CANN 9.2.0，本机不可运行。
