# torch.ops._C_ascend.npu_hc_pre_v3（vllm-ascend HcPre 自定义内核）测试归档

DeepSeek-V4.1 **生产实际调用**的 MHC 前向算子（`deepseek_v41/model.py:859`），与 CANN `cann_ops_transformer.mhc_pre_sinkhorn` 同语义（RMSNorm + 三路投影 + Sinkhorn 归一化），vllm-ascend 打包的 AscendC 自定义内核。

## 检索证据

| 来源 | 位置 | 说明 |
| --- | --- | --- |
| 生产调用 | `vllm_ascend/models/deepseek_v41/model.py:859` | `hc_pre()` → `_C_ascend.npu_hc_pre_v3(...)`，每层 attn/ffn 前调用 |
| GE 算子定义 | `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_proto/inc/hc_pre_proto.h` | `HcPre`：输入 x/hc_fn/hc_scale/hc_base/pre_mix，输出 y/post/comb_frag/pre，attr `hc_sinkhorn_iters=20` |
| AscendC 内核 | `vllm_ascend/_cann_ops_custom/vendors/.../ascendc/hc_pre/` | `hc_pre.cpp`（`__DAV_C310__`=950 分支 + 默认 A2/A3 分支） |
| A2 内核二进制 | `.../kernel/ascend910b/hc_pre/HcPre_*.json` + `.o` | 已编译 |
| C++ 注册 | `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` | `npu_hc_pre_v3` / `npu_hc_post` 注册 |

## API 签名

```python
y, post, comb_frag, pre = torch.ops._C_ascend.npu_hc_pre_v3(
    x, hc_fn, hc_scale, hc_base, pre_mix,
    hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6)
```

- `x` `(B,S,N,C)` bf16；`hc_fn` `((2N+N²), N*C)` fp32；`hc_scale` `(3)` fp32；`hc_base` `(2N+N²)` fp32；`pre_mix` `(B,S,N)` fp32 可空
- 输出：`y` `(B,S,C)` bf16（→attention/MLP 输入）、`post` `(B,S,N)` fp32（=2·sigmoid）、`comb_frag` `(B,S,N,N)` fp32（Sinkhorn 双随机矩阵）、`pre` `(B,S,N)` fp32（=sigmoid+eps）

## 与 CANN mhc_pre_sinkhorn 双来源对照（同输入）

| 输出 | 一致性 |
| --- | --- |
| `comb_frag` ↔ `hRes`（Sinkhorn 矩阵） | **逐元素一致**（max diff 6.4e-6） |
| `post` ↔ `hPost`（2·sigmoid） | **一致**（max diff 2.5e-6） |
| `y` ↔ `hin` | **语义不同**：vllm `y`=Σ(pre·x) 沿 N 维加权（含 pre_mix 影响）；CANN `hin`=x@H^pre^T |

## A2 实测结果

| 项目 | 结果 |
| --- | --- |
| 精度 | PASS（post/comb 双来源一致，Sinkhorn 列和=1 精确） |
| 性能 S=1024/C=1280 | 0.134 ms/iter（CANN 版 0.185ms，**快 28%**） |
| 性能 S=2048/C=1280 | 0.241 ms/iter（CANN 版 0.326ms） |
| 性能 S=2048/C=2560 | 0.388 ms/iter |
| msprof kernel avg | 131.654 / 237.770 / 384.394 us（三 shape，`op_statistic.csv` 官方值，卡 7 复测，mac~0.2-0.3% vec~0.6% memory/vector bound） |
| 激活前提 | `enable_custom_op()` 必须在 `set_device()` 前调用 |

## 运行

```bash
python3 test.py --bench    # 双来源一致性 + 性能
python3 test.py --bench --msprof   # msprof 精准测 kernel avg
```

## 判定口径

- **一致性**：与 CANN `mhc_pre_sinkhorn` 同输入下 post/comb max diff < 5e-4
- **Sinkhorn 性质**：comb_frag 列和精确 1，行和 ≈ 1
- **性能**：单次毫秒级，优于 CANN 融合版（vllm-ascend 内核 A2 调优）
