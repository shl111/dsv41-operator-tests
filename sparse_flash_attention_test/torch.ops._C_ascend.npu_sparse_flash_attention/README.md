# `torch.ops._C_ascend.npu_sparse_flash_attention` 测试说明

> DeepSeek-V4.1 稀疏注意力（SparseFlashAttention 家族，通用离散 TopK 形态）算子 `torch.ops._C_ascend.npu_sparse_flash_attention`（vllm-ascend 打包的自定义算子）的激活验证、精度与性能测试归档。
> 概念关系（通用 SparseFlashAttention vs MLA 特化 SparseFlashMLA）见 `../README.md` §1.0。
>
> 归档字段：算子 / 算子名称 / 具体算子名称 / 输入 / 输入数据 / 输出 / 输出数据 / 性能 / 精度。

---

## 1. 测试环境与激活

| 项 | 值 |
| --- | --- |
| 硬件 | Atlas A2 / Ascend 910B3 x8 |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.0 / 2.10.0.post4 |
| vllm-ascend | 源码挂载 `/vllm-workspace/vllm-ascend`，C++ 扩展已编译（`vllm_ascend/vllm_ascend_C.*.so`） |
| 被测算子 | `torch.ops._C_ascend.npu_sparse_flash_attention`（注册于 `csrc/torch_binding.cpp:3138`） |

**激活方式（`enable_custom_op()`，`vllm_ascend/utils.py:544`）：**
- 检查 `VLLM_BATCH_INVARIANT` 与硬件 capability（A2 支持 `RUNTIME_CUSTOM_OPS`）→ 不拦截；
- `bootstrap_custom_op_env()` 将 `_cann_ops_custom/vendors/custom_transformer` 注入 `ASCEND_CUSTOM_OPP_PATH`；
- `import vllm_ascend.vllm_ascend_C` 注册 `torch.ops._C_ascend`，`import vllm_ascend.meta_registration`。

> **关键约束：`enable_custom_op()` 必须在 `torch.npu.set_device()` 之前调用**（`utils.py:546` 注释），否则 CANN RTS 组件提前初始化，SparseFlashAttention tiling 语义异常导致精度 FAIL（实测 max_diff 0.79→修复后 0.012）。

实测：`enable_custom_op()` 返回 `True`；`hasattr(torch.ops._C_ascend, 'npu_sparse_flash_attention')` = `True`；vendor 的 `op_impl/ai_core/tbe/kernel/ascend910b/` 已含 `SparseFlashAttention_04d19c…` / `SparseFlashMla_…` 编译产物。

---

## 2. 与 torch_npu 原生算子的关系

`_C_ascend.npu_sparse_flash_attention` 底层调用与 torch_npu 原生 `npu_sparse_flash_attention` 相同的 CANN `aclnnSparseFlashAttention` 内核。A/B 实测（相同输入、相同 TND/PA_BSND 生产范式）**两算子输出逐元素一致（max_diff = 0.0）**。

与原生算子的差异点：
- 调用入口不同（`torch.ops._C_ascend` vs `torch_npu`）；
- `_C_ascend` 注册签名带 `return_softmax_lse` 默认 `False`，未暴露 `pre_tokens/next_tokens`（内部固定 INT64_MAX）；
- 同为 A2 MLA 内核约束：`head_dim=512`、`kv 头=1`、`query_rope/key_rope` 必传真实张量、`attention_mode=2`；
- sfa_v1.py:224（dense 模型 NoPE 路径）传 `query_rope=None/key_rope=None` 在本机 **不可运行**（`EZ0037 queryRope shape nullptr`）——本测试按 `device_op.py:560` 的 MLA 生产范式传真实 `q_pe`/`kv_cache[1]`。

---

## 3. 已验证语义（浮点参考对照）

与 DSV4.1 离散 TopK 稀疏注意力一致（PA_BSND 下 `sparse_indices` 为逻辑 token 位置，经 `block_table` 映射物理 block）：
- `score(i,j) = (q_nope[i]·k[j] + q_rope[i]·k_rope[j]) / sqrt(512)`（rope 项参与打分）；
- 每 query 只对 `sparse_indices[i]` 中的有效 KV 位置做 softmax；
- 输出 `attention_out` 为加权 V 的 nope 部分，shape `[Tq, N, D]`（D=512，不含 rope 位）。

**精度实测（9 用例套件，vs float32 参考，TND+PA_BSND+attention_mode=2，`block_table=arange(nb)`，判据 max≤0.03 ∧ mean≤2e-3）：**

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

9/9 PASS，数值与 torch_npu 原生**完全一致**（同内核，逐元素一致 max_diff=0.0）。

> 注：`block_table` 必须为 `arange(nb)`（0..nb-1 物理 block 映射），若填 0 会导致 Tkv>128 时逻辑位置 128+ 错误映射到 block 0（实测 max_diff 1.14 FAIL）。

**本机性能（910B3，`torch.npu.Event`，N=1 头）：**

| 场景 | 输入 | Event wall-time | msprof kernel avg | mac/vec |
| --- | --- | --- | --- | --- |
| decode（SS=8） | q `[1,1,512]`；kv PA_BSND `[1,128,1,512]`；Tkv=128 | 0.0887ms | **17.294 us** | 0.002% / 0.006% |
| decode（SS=128） | q `[1,1,512]`；kv PA_BSND `[32,128,1,512]`；Tkv=4096 | 0.0895ms | **19.180 us** | 0.003% / 0.006% |
| decode 长上下文 | 同上 Tkv=32768 | 0.0880ms | **19.200 us** | 0.003% / 0.006% |
| prefill（SS=8） | q `[64,1,512]`；kv PA_BSND `[1,128,1,512]`；Tkv=128 | 0.0877ms | **27.464 us** | 0.053% / 0.057% |
| prefill（SS=32） | q `[256,1,512]`；kv PA_BSND `[2,128,1,512]`；Tkv=256 | 0.0886ms | **69.746 us** | 0.073% / 0.075% |
| prefill（SS=128） | q `[1024,1,512]`；kv PA_BSND `[8,128,1,512]`；Tkv=1024 | 0.2730ms | **289.842 us** | 0.110% / 0.078% |
| prefill 长上下文 | 同上 Tkv=32768 | 0.2732ms | **291.204 us** | 0.109% / 0.076% |
| prefill（SS=128 fp16） | 同 SS=128 但 fp16 | 0.2735ms | **291.174 us** | 0.109% / 0.077% |

**性能要点：**
- decode（SS=128）0.08~0.10ms，prefill（Tq=1024/SS=128）0.28ms，latency 均不随 Tkv 增长（离散 TopK 优势）；
- 与 torch_npu 原生 `npu_sparse_flash_attention` 同内核，kernel avg 基本一致（偏差 <4%，launch 路径差异）；8 case 全表见根目录 `TEST_SUMMARY.md`。

---

## 4. 测试程序

| 文件 | 说明 |
| --- | --- |
| `test.py` | `_C_ascend.npu_sparse_flash_attention` 套件测试（9 精度用例 + 8 性能 case）：`python3 test.py --check-precision`（精度）；`python3 test.py --bench`（Event wall-time）；`python3 test.py --bench --msprof`（kernel avg + 利用率）；默认无参数全跑 |

运行环境：Atlas A2 NPU + torch_npu + vllm-ascend（需 `enable_custom_op()` 在 `set_device()` 前激活）。
