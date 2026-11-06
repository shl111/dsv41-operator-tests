# cann_ops_transformer.mhc_post

MHC Post 后处理算子（CANN 融合版）。

| 字段 | 内容 |
| --- | --- |
| 算子 | MHC Post |
| 算子名称 | `cann_ops_transformer.mhc_post` |
| 具体算子名称 | `cann_ops_transformer.ops.mhc_post` |
| 输入 | `x` `(B,S,N,D)`/`(T,N,D)` bf16 残差流；`h_res` `(B,S,N,N)`/`(T,N,N)` fp32；`h_out` `(B,S,D)`/`(T,D)` bf16 collapsed；`h_post` `(B,S,N)`/`(T,N)` fp32 |
| 输入数据 | torch.randn（bf16/fp32） |
| 输出 | `y` `(B,S,N,D)`/`(T,N,D)` bf16 |
| 输出数据 | `y[i,:] = h_post[i]·h_out[:] + Σ_j h_res[j,i]·x[j,:]` |
| 性能 | BSND S=1024 0.092ms / S=2048 0.095ms（A2，Event wall-time）；kernel avg 27.456 / 48.536us（msprof op_statistic，卡 7 复测） |
| 精度 | vs float32 参考 0.026（BSND）/ 0.030（TND），bf16 噪声 |

## 运行

```bash
python3 test.py            # 语义精度（BSND + TND）
python3 test.py --bench    # 追加性能
python3 test.py --bench --msprof   # msprof 精准测 kernel avg + 利用率
```

## 判定口径

- vs float32 参考 max diff < 0.05（bf16 输出量化噪声，mean 极小）
- BSND / TND 布局均支持
