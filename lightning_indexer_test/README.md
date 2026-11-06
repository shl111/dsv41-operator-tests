# npu_lightning_indexer (LightningIndexer) 测试归档

DeepSeek-V4.1 推理链路中，`npu_lightning_indexer` 用于**稀疏注意力索引选择**：给定 query、共享单头 key 与逐 token weights，输出每个 query 最相关 topK 的 key 位置索引，供后续 SparseFlashAttention / SparseFlashMLA 消费。

> 精度 + 性能统一汇总见根目录 [`TEST_SUMMARY.md`](../TEST_SUMMARY.md)。

## 检索证据

| 来源 | 位置 | 说明 |
| --- | --- | --- |
| torch_npu 注册 | `torch_npu/utils/custom_ops.py:255` | `npu_lightning_indexer` 已注册 |
| torch_npu 文档 | `torch_npu/_op_plugin_docs.py:1328` | API 签名见下 |
| vllm-ascend 生产调用 | `vllm_ascend/device/device_op.py:502` | decode 用 `torch_npu.npu_lightning_indexer` |
| vllm-ascend 生产调用 | `vllm_ascend/device/device_op.py:485` | prefill 用 `torch.ops._C_ascend.npu_lightning_indexer_quant` |
| CANN A2 内核 | `opp/built-in/op_impl/.../ascendc/lightning_indexer/arch22/` | `lightning_indexer_kernel.h` 等 |
| DSV4.1 生产参数 | `models/deepseek_v4_1/models/configuration_deepseek.py:353-361` | 见 §1.4 |

## 同族接口实测矩阵（Atlas A2）

torch_npu 共注册 6 个 LightningIndexer 族接口（`custom_ops.py:36,80,112,187,255,332`），逐一实测：

| 接口 | 注册位置 | 用途 | A2 实测结果 |
| --- | --- | --- | --- |
| `npu_lightning_indexer` | `custom_ops.py:255` | 推理：topK 稀疏索引选择（主接口，DSV4.1 decode 用） | **可运行**，输出 `(B,S1,1,K)` int32（本归档主体，见下） |
| `npu_dense_lightning_indexer_softmax_lse` | `custom_ops.py:187` | 训练前置：softmax max/sum 提前计算以省显存 | **可运行**，输出两个 `(B,N2,S1)` float32（query_index/key_index 传普通张量即可） |
| `npu_quant_lightning_indexer` | `custom_ops.py:112` | 推理：Per-Token-Head 量化输入版（vllm prefill 用 quant 变体） | **不可运行**：`ACLNN_ERR_PARAM_INVALID`（161002）。文档参数为 A3 推理系列/A5（int8/fp8），A2 不支持 |
| `torch.ops._C_ascend.npu_lightning_indexer_quant`（vllm-ascend 自定义内核） | `vllm_ascend/device/device_op.py:485`（C8 路径）；内核 `_cann_ops_custom/vendors/.../ascendc/lightning_indexer_quant/` | 推理：vllm prefill 用 int8 量化版（kernel json 输入 query/key 为 int8、weights fp16） | **不可运行（A2）**：`enable_custom_op()` 激活后调用报 `EZ1009 dtype/format 不一致`。虽有 ascend910b 内核二进制，但 A2 实测失败（format 校验不通过），需 A3/A5 |
| `npu_lightning_indexer_grad` | `custom_ops.py:80` | 训练反向 | 无文档，未单独验证 |
| `npu_dense_lightning_indexer_grad_kl_loss` | `custom_ops.py:36` | 训练反向 + KL loss（dense warmup） | **不可运行**：tiling 失败（`blockDim=0`，rtsLaunchKernel 失败），A2 训练内核不支持 |
| `npu_sparse_lightning_indexer_grad_kl_loss` | `custom_ops.py:332` | 训练反向 + KL loss（sparse warmup） | **不可运行**：aclnn 调用失败，A2 训练内核不支持 |

结论：**推理相关可运行接口为 `npu_lightning_indexer`（主）与 `npu_dense_lightning_indexer_softmax_lse`（其训练前置）**；grad_kl_loss 系为训练专用且 A2 平台不支持；quant 版为 A3/A5 推理专用。



## API 签名

```
torch_npu.npu_lightning_indexer(
    query, key, weights, *,
    actual_seq_lengths_query=None, actual_seq_lengths_key=None,
    block_table=None,
    layout_query='BSND', layout_key='BSND',
    sparse_count=2048, sparse_mode=3
) -> Tensor        # [B, S1, 1, sparse_count] int32 索引
```

- `query` `[B,S1,N1,D]` 或 `[T1,N1,D]`；`key` `[B,S2,1,D]` 或 TND/PA_BSND；`weights` `[B,S1,N1]` 或 `[T1,N1]`（key 共享单 head）
- `sparse_count`：每 query 保留的索引数（topK）
- `sparse_mode`：0=无 mask；3=rightDownCausal

## 打分语义（参考实现推导）

```
score(s2) = Σ_h w[s1,h] · ⟨q[s1,h], k[s2]⟩      # key 单 head 共享
```

输出为每 query 的 `topK` 索引集合，无效位置填 `-1`。

### sparse_mode=0（无 mask）
全部 key 参与打分，取 topK。

### sparse_mode=3（rightDownCausal）
每 query `s1` 有效 key 前缀数 = `S2 - S1 + s1 + 1`（索引 `0..S2-S1+s1` 有效），超出部分不参与，不足 K 时补 `-1`。

### topK 为稀疏近似（两阶段合并，非全局精确）
A2 内核用 `SparseTopK` 分块维护 running topK（`BASE_TOPK=2048`，`s2BaseSize=512`，每 4 块 merge），
因此随机数据下**输出是近似的 topK**（边界 tie 由 bf16 精度与合并顺序决定），生产链路仅按索引集合消费。
验证口径：
- **打分公式**：确定性 one-hot 数据（无 tie）精确验证集合匹配
- **边界 tie**：随机数据在 tie 位可能与 float32 参考索引不同（精度特性，非 bug）
- **合法性与 mask**：随机 shape 校验索引范围、mode3 前缀、-1 填充

## A2 平台约束

| 约束 | 现象 |
| --- | --- |
| `head_dim` 必须 128 | 否则 `EZ0009` (input query's last dim head_dim only support 128) |
| key 必须 `N2=1` | 多 head key 报 tiling 错误 |
| TND 布局 `actual_seq_lengths_query` | 必须传**前缀和**（cumulative） |
| 布局 | query/key 均支持 BSND / TND / PA_BSND |

## 测试运行

```bash
python3 test_lightning_indexer.py            # 精度 + 语义（确定性 + 随机）
python3 test_lightning_indexer.py --bench    # 追加生产 shape 性能
python3 test_lightning_indexer.py --bench --msprof   # msprof 精准测 kernel avg
python3 test_lightning_indexer.py --dtype fp16
```

## 测试结果（Atlas A2 实测）

| 项目 | 结果 |
| --- | --- |
| deterministic K=2/4（BSND，打分+mask 集合精确） | PASS |
| TND/PA_BSND 生产范式（one-hot 集合精确） | PASS |
| random 20 trials（范围/mask/-1 合法性） | PASS |
| decode T1=1,T2=4096,N1=32,K=2048 | 0.07 ms/iter（kernel avg **18.942 us**） |
| decode T1=1,T2=32768,N1=32,K=2048 | 0.07 ms/iter（kernel avg **30.942 us**） |
| prefill T1=1024,T2=1024,N1=32,K=2048 | 0.15 ms/iter（kernel avg **145.194 us**） |
| prefill T1=1024,T2=32768,N1=32,K=2048 | 3.51 ms/iter（kernel avg **3506.430 us**） |

> msprof 数据为 `op_statistic.csv` 官方 Avg Time(us)（卡 7 复测，warmup5+active10，MIX_AIC，mac<0.3% vec<1% memory/vector bound）。Event wall-time（ms/iter）远大于 kernel avg：主体差异为 host launch / 排队开销（`npu_lightning_indexer` kernel 极短但 launch 开销大，尤其 decode 场景 launch 占比接近 100%）。

## 判定口径

- **集合匹配**：确定性数据（one-hot + 显式 weights）下 got 与参考集合完全一致
- **合法范围**：随机数据所有输出索引 ∈ `[0,S2)` 或 `-1`，mode3 索引不超有效前缀
- **性能**：单次毫秒级（见上表），decode 远低于 prefill（打分量级 O(S1×S2)）
