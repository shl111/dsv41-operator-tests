# torch_npu.npu_mhc_post

MHC Post 后处理算子（torch_npu 原生接口）。

| 字段 | 内容 |
| --- | --- |
| 算子 | MHC Post |
| 算子名称 | `torch_npu.npu_mhc_post` |
| 具体算子名称 | `torch_npu.npu_mhc_post`（注册 `custom_ops.py:188`） |
| 输入 | `x` `(B,S,N,D)` bf16 残差流；`h_res` `(B,S,N,N)` fp32；`h_out` `(B,S,D)` bf16 collapsed；`h_post` `(B,S,N)` fp32 |
| 输入数据 | torch.randn（bf16/fp32） |
| 输出 | `y` `(B,S,N,D)` bf16 |
| 输出数据 | `y[i,:] = h_post[i]·h_out[:] + Σ_j h_res[j,i]·x[j,:]` |
| 性能 | S=1024 0.075ms / S=2048 0.077ms（A2，Event wall-time）；kernel avg 27.112 / 48.408us（msprof op_statistic，卡 7 复测） |
| 精度 | vs float32 参考 0.026，bf16 噪声 |

## 运行

```bash
python3 test.py            # 语义精度
python3 test.py --bench    # 追加性能
python3 test.py --bench --msprof   # msprof 精准测 kernel avg + 利用率
```

## 判定口径

- vs float32 参考 max diff < 0.05（bf16 输出量化噪声）
- 与 CANN `cann_ops_transformer.mhc_post` 同输入输出**逐元素一致**（max diff = 0.0）
