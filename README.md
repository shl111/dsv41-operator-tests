# DeepSeek-V4.1 推理算子测试归档

DeepSeek-V4.1（DSV4.1）NPU 推理链路上 4 个核心算子族的检索、精度与性能测试归档。

- **测试环境**：Atlas A2（Ascend 910B3），CANN 9.1.0，torch_npu 2.10.0.post4，cann_ops_transformer 1.0.0
- **数据口径**：kernel avg 以 msprof `op_statistic.csv` 官方 `Avg Time(us)` 为准（warmup5 在 profiler 外 + active10 在 profiler 内，Level1 + PipeUtilization）；Event wall-time（`torch.npu.Event`）含 host launch/排队开销
- **最终版本**：kernel avg / mac/vec 为卡 7 全量复测数据（`ASCEND_RT_VISIBLE_DEVICES=7 python3 run_all_msprof.py`，12 step 全部 exit=0）

## 内容

| 算子族 | 来源 | 目录 |
| --- | --- | --- |
| mhc_post（MHC Post 后处理） | CANN / torch_npu / vllm-ascend 3 来源 | [`mhc_post_test/`](mhc_post_test/) |
| mhc_pre_sinkhorn（MHC Pre + Sinkhorn，forward + backward） | CANN / vllm-ascend 2 来源 + backward | [`mhc_pre_sinkhorn_test/`](mhc_pre_sinkhorn_test/) |
| lightning_indexer（稀疏注意力索引选择） | torch_npu | [`lightning_indexer_test/`](lightning_indexer_test/) |
| sparse_flash_attention（SFA 算子族） | CANN sparse_flash_mla / torch_npu / vllm-ascend 3 来源 | [`sparse_flash_attention_test/`](sparse_flash_attention_test/) |

**统一汇总**：[`TEST_SUMMARY.md`](TEST_SUMMARY.md)（输入/输出数据 + 精度 + 性能，每 case 一行）

## 一键复测

```bash
# 卡 7 全量复测（12 step：精度 + msprof kernel avg）
ASCEND_RT_VISIBLE_DEVICES=7 python3 run_all_msprof.py
```
