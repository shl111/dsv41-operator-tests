# torch.ops._C_ascend.npu_hc_post

MHC Post 后处理算子（vllm-ascend 自定义内核，**DSV4.1 生产实际调用**）。

| 字段 | 内容 |
| --- | --- |
| 算子 | MHC Post |
| 算子名称 | `torch.ops._C_ascend.npu_hc_post` |
| 具体算子名称 | `torch.ops._C_ascend.npu_hc_post`（生产调用 `vllm_ascend/models/deepseek_v41/model.py:872`；GE 算子 `hc_post_proto.h`：`(x, residual, post, comb) → y`） |
| 输入 | `x` `(B,S,D)` bf16 collapsed sub-layer 输出；`residual` `(B,S,N,D)` bf16 hc 流；`post` `(B,S,N)` fp32；`comb` `(B,S,N,N)` fp32 |
| 输入数据 | torch.randn（bf16/fp32） |
| 输出 | `y` `(B,S,N,D)` bf16 |
| 输出数据 | `y[i,:] = post[i]·x[:] + Σ_j comb[j,i]·residual[j,:]` |
| 性能 | S=1024 **0.045ms** / S=2048 **0.046ms**（A2，三来源中最快，~2× 于原生，Event wall-time）；kernel avg 22.700 / 39.452us（msprof op_statistic，卡 7 复测） |
| 精度 | vs float32 参考 0.015，bf16 噪声；与 CANN/torch_npu 输出逐元素一致（diff=0.0） |

## 运行

```bash
python3 test.py            # 语义精度（生产范式）
python3 test.py --bench    # 追加性能
python3 test.py --bench --msprof   # msprof 精准测 kernel avg + 利用率
```

## 判定口径

- vs float32 参考 max diff < 0.05（bf16 输出量化噪声）
- 激活前提：`enable_custom_op()` 须在 `set_device()` 前调用
- 布局为生产范式（collapsed 单流 + 残差流），与 CANN BSND 布局参数映射不同
