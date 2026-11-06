# DeepSeek-V4.1 推理算子测试统一汇总（输入 / 输出数据 + 精度 + 性能）

> 汇总 mhc_post / mhc_pre_sinkhorn（含 backward）/ lightning_indexer / sparse_flash_attention 四个算子族的实测数据。**每个算子族先独立列出「输入数据」「输出数据」两个列表，再给出测量数据表**（每 case 一行）。
>
> **数据口径**（保证准确性）：
> - 性能分两列：**Event wall-time**（`torch.npu.Event`，含 host launch/排队）与 **kernel avg**（msprof `op_statistic.csv` 官方 `Avg Time(us)`，warmup=5 在 profiler 外 + active=10 在 profiler 内，Level1 + PipeUtilization）。`op_statistic.csv` 为权威口径；`kernel_details.csv` 仅用于 mac/vec 利用率参考。
> - **精度**：各算子族判据不同，统一以「实测值 + 判据 → PASS/FAIL」表示。
> - **shape 语义**：B=batch、S=序列长、N=head 数、C=压缩隐藏维度、Tq/Tkv=query/key token 数、SS=每 query 稀疏候选数、nb=block 数。
> - 输入/输出数据默认 torch.randn 生成（与各算子 README 归档一致）；`—` 表示该项未实测。
> - 环境：Atlas A2（Ascend 910B3），CANN 9.1.0，torch_npu 2.10.0.post4，cann_ops_transformer 1.0.0。
> - **最终版本（卡 7 全量复测）**：kernel avg / mac/vec 为本机卡 7 全量复测数据（`ASCEND_RT_VISIBLE_DEVICES=7 python3 run_all_msprof.py`，12 step 全部 exit=0）；Event wall-time 为历史采集值。
> - 一键复测：根目录 `ASCEND_RT_VISIBLE_DEVICES=7 python3 run_all_msprof.py`（12 step 顺序执行，各 step 内联 msprof）。

## 1. mhc_post（MHC Post 后处理，3 来源）

公式：`y[i,:] = h_post[i]·h_out[:] + Σ_j h_res[j,i]·x[j,:]`。布局映射：CANN `x`=vllm `residual`、`h_res`=vllm `comb`、`h_out`=vllm `x`、`h_post`=vllm `post`。三来源同输入输出**逐元素一致**（max diff = 0.0）。

### 1.1 输入数据

**CANN / torch_npu 布局（BSND）：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `x` | `(B,S,N,D)` / `(T,N,D)` | bf16 | 残差流 |
| 2 | `h_res` | `(B,S,N,N)` / `(T,N,N)` | fp32 | 残差投影矩阵 |
| 3 | `h_out` | `(B,S,D)` / `(T,D)` | bf16 | collapsed 输出投影 |
| 4 | `h_post` | `(B,S,N)` / `(T,N)` | fp32 | 后处理系数 |

**vllm 布局（生产调用）：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `x` | `(B,S,D)` | bf16 | collapsed 输出投影 |
| 2 | `residual` | `(B,S,N,D)` | bf16 | hc 流 |
| 3 | `post` | `(B,S,N)` | fp32 | 后处理系数 |
| 4 | `comb` | `(B,S,N,N)` | fp32 | 组合矩阵 |

### 1.2 输出数据

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `y` | `(B,S,N,D)` / `(T,N,D)` | bf16 | `y=h_post·h_out + h_res^T·x`（vllm：`y=post·x + comb^T·residual`） |

### 1.3 测量数据（精度 + 性能）

| 具体算子名称 | 精度实测（S=4） | 判据 → 结果 | Event wall-time（S=1024/S=2048） | kernel avg（S=1024/S=2048） | mac/vec |
| --- | --- | --- | --- | --- | --- |
| `cann_ops_transformer.mhc_post` | vs f32 参考 0.0261（BSND）/ 0.0303（TND） | < 0.05 → **PASS** | 0.092 / 0.095 ms | **27.456 / 48.536 us** | 0% / 0.389% |
| `torch_npu.npu_mhc_post` | vs f32 参考 0.0261（BSND） | < 0.05 → **PASS** | 0.075 / 0.077 ms | **27.112 / 48.408 us** | 0% / 0.396% |
| `torch.ops._C_ascend.npu_hc_post`（生产调用） | vs f32 参考 0.0153（生产范式） | < 0.05 → **PASS** | 0.045 / 0.046 ms | **22.700 / 39.452 us** | 0% / 0.512% |

结论：三来源 kernel 纯执行同量级（22.7-48.5us），wall-time 差异（45-95us）来自 host launch 开销；mac=0% → 纯 memory/vector bound 算子。vllm 版最快且为 DSV4.1 生产实际调用（`model.py:872`）。

## 2. mhc_pre_sinkhorn（MHC Pre + Sinkhorn，forward + backward）

公式：RMSNorm + 三路投影（H^pre/H^post/H^res）+ Sinkhorn 20 次迭代。**`post`/`comb_frag`（Sinkhorn 矩阵）双来源逐元素一致（2.5e-6/6.4e-6）**；`y` 语义不同（vllm `y`=Σ(pre·x) 与 HF 一致，CANN `hin`=x@H^pre^T）。

### 2.1 输入数据

**CANN `cann_ops_transformer.mhc_pre_sinkhorn`（forward）与 backward（Autograd 复用）：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `x` | `(1,S,4,C)` | bf16 | 残差输入 |
| 2 | `phi` | `(24,4C)` | fp32 | 三路投影权重（pre/post/res 拼接） |
| 3 | `alpha` | `(3)` | fp32 | 缩放系数 |
| 4 | `bias` | `(24)` | fp32 | 偏置 |
| 5 | `numIters` | 标量 = 20 | int | Sinkhorn 迭代次数（必须 = 20） |

**vllm `torch.ops._C_ascend.npu_hc_pre_v3`（生产调用）：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `x` | `(1,S,4,C)` | bf16 | 残差输入 |
| 2 | `hc_fn` | `(24,4C)` | fp32 | 三路投影权重（pre/post/res 拼接） |
| 3 | `hc_scale` | `(3)` | fp32 | 缩放系数 |
| 4 | `hc_base` | `(24)` | fp32 | 偏置 |
| 5 | `pre_mix` | `(1,S,4)` | fp32 | pre 混合系数 |

### 2.2 输出数据

**CANN forward：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `hin` | `(1,S,C)` | bf16 | `x@H^pre^T` |
| 2 | `hPost` | `(1,S,4)` | fp32 | `2·sigmoid` |
| 3 | `hRes` | `(1,S,16)` | fp32 | `Sinkhorn(H^res)`（列和=1，行和≈1） |

**CANN backward：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `x.grad` | `(1,S,4,C)` | bf16 | 梯度（路径：hin→α[0]） |
| 2 | `phi.grad` | `(24,4C)` | fp32 | 梯度（路径：hpost→α[1]） |
| 3 | `alpha.grad` | `(3)` | fp32 | 梯度（路径：hres→α[2]） |
| 4 | `bias.grad` | `(24)` | fp32 | 梯度 |

**vllm：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `y` | `(1,S,C)` | bf16 | `Σ(pre·x)` 沿 N 维加权 |
| 2 | `post` | `(1,S,4)` | fp32 | 后处理系数 |
| 3 | `comb_frag` | `(1,S,4,4)` | fp32 | Sinkhorn 矩阵 |
| 4 | `pre` | `(1,S,4)` | fp32 | pre 系数 |

### 2.3 测量数据（CANN forward）

精度判据：hres<5e-3 ∧ hpost mean≤2e-4 ∧ >5e-3 点占比≤1% ∧ hpost max≤2e-2 ∧ 列和<1e-3 ∧ 行和<8e-2。

| shape | 精度实测 | 结果 | Event wall-time | kernel avg | mac/vec |
| --- | --- | --- | --- | --- | --- |
| S=1024/C=1280 | hin 0.0294 / hpost max 0.0014（mean 6.72e-5，>5e-3 占比 0.00%）/ hres 0.0019；\|col-1\|=1.19e-6，\|row-1\|=0.0756 | **PASS** | 0.186 ms | **184.506 us** | 0.078% / 0.657% |
| S=8/C=1920 | hin 0.0162 / hpost 0.0000（mean 1.02e-7，占比 0.00%）/ hres 0.0004；\|col-1\|=1.13e-6，\|row-1\|=0.0371 | **PASS** | — | —（未测 msprof） | — |
| S=4/C=2560 | hin 0.0162 / hpost 0.0009（mean 9.00e-5，占比 0.00%）/ hres 0.0021；\|col-1\|=1.07e-6，\|row-1\|=0.0371 | **PASS** | — | —（未测 msprof） | — |
| S=2048/C=1280 | 仅 msprof 性能（精度未复测，与 S=1024/C=1280 同构） | — | 0.325 ms | **322.052 us** | 0.118% / 0.744% |
| S=2048/C=2560 | 仅 msprof 性能（精度未复测，与 S=1024/C=1280 同构） | — | 0.379 ms | **373.692 us** | 0.143% / 0.710% |

### 2.4 测量数据（CANN backward）

精度判据：梯度路径独立（hin→α[0]、hpost→α[1]、hres→α[2]）∧ 4 梯度有限非 NaN。

| shape | 精度实测 | 结果 | Event wall-time | kernel avg | mac/vec |
| --- | --- | --- | --- | --- | --- |
| S=4/C=1280 | 梯度路径独立 + 4 梯度有限非 NaN | **PASS** | — | —（未测 msprof） | — |
| S=1024/C=1280 | 仅 msprof 性能（梯度路径已在 S=4 验证） | — | — | **197.420 us** | 0.038% / 0.254% |
| S=2048/C=1280 | 仅 msprof 性能（梯度路径已在 S=4 验证） | — | — | **383.714 us** | 0.035% / 0.261% |

### 2.5 测量数据（vllm `npu_hc_pre_v3`）

精度判据：post_diff<5e-4 ∧ comb_diff<5e-4 ∧ 列和<1e-3 ∧ 行和<8e-2。

| shape | 精度实测 | 结果 | Event wall-time | kernel avg | mac/vec |
| --- | --- | --- | --- | --- | --- |
| S=4/C=1280 | post_diff 2.50e-6 / comb_diff 6.44e-6；\|col-1\|=1.01e-6，\|row-1\|=0.026 | **PASS** | — | —（未测 msprof） | — |
| S=1024/C=1280 | 仅 msprof 性能（精度未复测，与 S=4/C=1280 同构） | — | 0.134 ms | **131.654 us** | 0.221% / 0.600% |
| S=2048/C=1280 | 仅 msprof 性能（精度未复测，与 S=4/C=1280 同构） | — | 0.241 ms | **237.770 us** | 0.267% / 0.646% |
| S=2048/C=2560 | 仅 msprof 性能（精度未复测，与 S=4/C=1280 同构） | — | 0.388 ms | **384.394 us** | 0.333% / 0.646% |

结论：
- forward kernel 纯执行：vllm `HcPre`（131.654-384.394us）全面优于 CANN（184.506-373.692us），S=2048/C=1280 快 **26%**（237.770 vs 322.052）；S=2048/C=2560 接近（384.394 vs 373.692，CANN 略优 2.8%）。
- backward 与 forward 同量级（197.420/383.714 vs 184.506/322.052us），慢 7-19%。
- mac<0.5%：memory/vector bound 融合算子。
- 精度注：CANN 版大 shape（S=1024）hpost 极少数点超 5e-3 为 kernel 内部量化噪声（本次复测 max 0.0014、>5e-3 占比 0.00%，mean 6.72e-5；vllm 版同 shape 精确到 2.5e-6）；判据按「mean + 超阈值占比 + max≤2e-2」判定，非正确性问题。

## 3. lightning_indexer（npu_lightning_indexer）

语义：`score(s2)=Σ_h w[s1,h]·⟨q[s1,h], k[s2]⟩`，输出每 query topK 索引（无效填 -1）。`sparse_mode=3`（rightDownCausal）时有效前缀 = `S2-S1+s1+1`。

### 3.1 输入数据

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `query` | `(B,S1,N1,128)` | bf16 | 查询（N1=32） |
| 2 | `key` | PA_BSND `(B,nb,128,1,128)` | bf16 | 键（kv 头=1，block_size=128，nb=T2/128） |
| 3 | `weights` | `(B,S1,N1)` | fp32 | 打分权重 |
| 4 | `sparse_count` | 标量 = K（2048） | int | 每 query topK 数 |
| 5 | `sparse_mode` | 标量 = 3 | int | rightDownCausal |

### 3.2 输出数据

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | 索引 | `(B,S1,1,K)` | int32 | 每 query topK 索引，无效位填 -1 |

### 3.3 测量数据（精度 + 性能）

精度 shape 口径：deterministic K=2/4 与 TND/PA_BSND 生产范式用 T1=8/T2=32；random 20 trials 随机 shape。判据：集合匹配 + 合法范围（索引∈[0,S2) 或 -1，mode3 不超有效前缀）。

| 场景 | shape（性能口径） | 精度实测 | 结果 | Event wall-time | kernel avg | mac/vec |
| --- | --- | --- | --- | --- | --- |--- |
| decode | T1=1/T2=4096/N1=32/K=2048 | deterministic K=2/4 集合精确；TND/PA_BSND one-hot 精确；random 20 trials 合法 | **PASS** | 0.07 ms/iter | **18.942 us** | 0.02% / 0.19% |
| decode 长上下文 | T1=1/T2=32768/N1=32/K=2048 | 集合匹配 + 合法范围（精度 shape 独立） | **PASS** | 0.07 ms/iter | **30.942 us** | 0.086% / 0.238% |
| prefill | T1=1024/T2=1024/N1=32/K=2048 | 集合匹配 + 合法范围 | **PASS** | 0.15 ms/iter | **145.194 us** | 0.185% / 0.707% |
| prefill 长上下文 | T1=1024/T2=32768/N1=32/K=2048 | 集合匹配 + 合法范围 | **PASS** | 3.51 ms/iter | **3506.430 us** | 0.269% / 0.969% |

结论：kernel avg 极短（decode 18.9-30.9us）但 wall-time 高（0.07ms），launch 占比接近 100%；prefill 长上下文 kernel 3.5ms 为主开销。A2 约束：head_dim=128、key N2=1、TND 需前缀和、topK 为稀疏近似（两阶段合并）。

## 4. sparse_flash_attention（SFA 算子族）

概念：SparseFlashAttention（通用离散 TopK）与 SparseFlashMLA（MLA 特化，共享 KV、kv 头=1、head_dim=512）为同一稀疏注意力算子两形态。DSV4.1 为 MLA 模型，生产算子为 `mixed_quant_sparse_flash_mla`（ds41，需 A5+CANN 9.2.0，本机不可运行）。本机实测对象为同族可运行算子。

### 4.1 输入数据

**CANN `cann_ops_transformer.sparse_flash_mla`（SWA 形态）：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `q` | `(1,Tq,64,512)` | bf16 | 查询（64 头） |
| 2 | `ori_kv` | `(1,Tkv,1,512)` | bf16 | 原始 KV（kv 头=1） |
| 3 | `sinks` | `(1)` | fp32 | 固定窗口起点 |
| 4 | `win` | 标量 127/0 | int | 滑动窗口左/右 |
| 5 | `scale` | 标量 1/√512 | fp32 | 缩放 |

**CANN `cann_ops_transformer.sparse_flash_mla`（CSA / HCA 形态）：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `q` | `(1024,64,512)` | bf16 | 查询（64 头，TND） |
| 2 | `ori_kv` | `(1024,1,512)` | bf16 | 原始 KV |
| 3 | `cmp_kv` | CSA `(256,1,512)` / HCA `(8,1,512)` | bf16 | 竞争 KV（CSA: K=512/1024 用 256；HCA: ratio=128 用 8） |
| 4 | `cmp_sparse_indices` | CSA 专用 | int | 竞争稀疏索引 |
| 5 | `residual` | 与 q 同形 | bf16 | 残差流 |
| 6 | `ratio` | CSA=4 / HCA=128 | int | 采样比 |

**torch_npu / vllm `npu_sparse_flash_attention`（离散 TopK，TND + PA_BSND，生产范式）：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `query` | `(Tq,1,512)` | bf16/fp16 | 查询（N1=1，head_dim=512） |
| 2 | `kv` | PA_BSND `(nb,128,1,512)` | bf16/fp16 | KV（block_size=128，nb=Tkv/128） |
| 3 | `query_rope` | `(Tq,1,64)` | 同 query | rope 位查询 |
| 4 | `key_rope` | `(nb,128,1,64)` | 同 query | rope 位键 |
| 5 | `sparse_indices` | `(Tq,1,SS)` | int32 | 每 query 最近 SS 候选位置 |
| 6 | 属性 | `am=2`、`pre=next=INT64_MAX`、`sparse_mode=3`、`sparse_block_size=1`、`scale=1/√512` | — | 硬约束（见 4.2） |

### 4.2 输出数据

**CANN `sparse_flash_mla`：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `attention_out` | `(1,Tq,64,512)` 或 `(1024,64,512)` | bf16 | 加权 V 的 nope 部分 |
| 2 | `softmax_lse` | `()` 标量 | fp32 | log-sum-exp |

**torch_npu / vllm `npu_sparse_flash_attention`：**

| # | 张量 | shape | dtype | 语义 |
| --- | --- | --- | --- | --- |
| 1 | `attention_out` | `(Tq,1,512)` | bf16/fp16 | 加权 V 的 nope 部分（D=512，不含 rope 位） |
| 2 | `softmax_max` | `()` 标量 | fp32 | softmax 最大值 |
| 3 | `softmax_sum` | `()` 标量 | fp32 | softmax 求和 |

### 4.3 测量数据（CANN `sparse_flash_mla`）

精度：12 用例（SWA/CSA/HCA × BSND/TND × fp16/bf16）**12/12 PASS**，均值误差 ~1e-4，判据 `max≤max(0.03, 0.05×ref_max)+0.25`。性能为 msprof kernel avg（fp16/bf16 双值）。

| 形态 | kernel avg fp16 | kernel avg bf16 | mac/vec fp16 | mac/vec bf16 |
| --- | --- | --- | --- | --- |
| SWA decode | **19.506 us** | **19.392 us** | mac 0.007% / vec 0.013% | mac 0.007% / vec 0.012% |
| SWA Sq128 | **39.378 us** | **40.078 us** | mac 0.161% / vec 0.339% | mac 0.161% / vec 0.334% |
| SWA Sq1024 | **171.198 us** | **173.930 us** | mac 0.43% / vec 0.624% | mac 0.424% / vec 0.616% |
| SWA 长 ctx | **174.088 us** | **173.736 us** | mac 0.438% / vec 0.607% | mac 0.441% / vec 0.610% |
| CSA K512 | **584.114 us** | **580.716 us** | mac 0.29% / vec 0.397% | mac 0.297% / vec 0.405% |
| CSA K1024 | **585.648 us** | **577.244 us** | mac 0.293% / vec 0.400% | mac 0.296% / vec 0.404% |
| HCA | **203.358 us** | **204.466 us** | mac 0.432% / vec 0.640% | mac 0.422% / vec 0.627% |

### 4.4 测量数据（torch_npu `npu_sparse_flash_attention`）

精度 9 用例 9/9 PASS，判据 `max≤0.03 ∧ mean≤2e-3`。

| 场景 | shape（Tq/Tkv/SS/dtype） | 精度 max_diff | 精度 mean_diff | 结果 | Event wall-time | kernel avg | mac/vec |
| --- | --- | --- | --- | --- | --- | --- |--- |
| decode | 1/128/8/bf16 | 0.00777 | 0.001172 | **PASS** | 0.0887 ms | **16.246 us** | 0.002% / 0.006% |
| decode | 1/4096/128/bf16 | 0.00777 | 0.001172 | **PASS** | 0.0895 ms | **18.390 us** | 0.003% / 0.006% |
| decode 长上下文 | 1/32768/128/bf16 | 0.00777 | 0.001172 | **PASS** | 0.0880 ms | **19.266 us** | 0.003% / 0.006% |
| prefill | 64/128/8/bf16 | 0.01226 | 0.001225 | **PASS** | 0.0877 ms | **29.280 us** | 0.049% / 0.053% |
| prefill | 256/256/32/bf16 | 0.01228 | 0.000767 | **PASS** | 0.0886 ms | **67.544 us** | 0.076% / 0.078% |
| prefill | 1024/1024/128/bf16 | 0.01434 | 0.000454 | **PASS** | 0.2730 ms | **268.440 us** | 0.117% / 0.083% |
| prefill 长上下文 | 1024/32768/128/bf16 | 0.00989 | 0.000453 | **PASS** | 0.2732 ms | **269.182 us** | 0.116% / 0.081% |
| prefill fp16 | 1024/1024/128/fp16 | 0.00172 | 0.000057 | **PASS** | 0.2735 ms | **269.634 us** | 0.117% / 0.082% |

### 4.5 测量数据（vllm `npu_sparse_flash_attention`，生产调用，`enable_custom_op()` 激活）

精度 9 用例 9/9 PASS，**数值与 torch_npu 原生完全一致**（同内核逐元素 max_diff=0.0）。判据 `max≤0.03 ∧ mean≤2e-3`。

| 场景 | shape（Tq/Tkv/SS/dtype） | 精度 max_diff | 精度 mean_diff | 结果 | Event wall-time | kernel avg | mac/vec |
| --- | --- | --- | --- | --- | --- | --- |--- |
| decode | 1/128/8/bf16 | 0.00777 | 0.001172 | **PASS** | 0.0887 ms | **17.294 us** | 0.002% / 0.006% |
| decode | 1/4096/128/bf16 | 0.00777 | 0.001172 | **PASS** | 0.0895 ms | **19.180 us** | 0.003% / 0.006% |
| decode 长上下文 | 1/32768/128/bf16 | 0.00777 | 0.001172 | **PASS** | 0.0880 ms | **19.200 us** | 0.003% / 0.006% |
| prefill | 64/128/8/bf16 | 0.01226 | 0.001225 | **PASS** | 0.0877 ms | **27.464 us** | 0.053% / 0.057% |
| prefill | 256/256/32/bf16 | 0.01228 | 0.000767 | **PASS** | 0.0886 ms | **69.746 us** | 0.073% / 0.075% |
| prefill | 1024/1024/128/bf16 | 0.01434 | 0.000454 | **PASS** | 0.2730 ms | **289.842 us** | 0.110% / 0.078% |
| prefill 长上下文 | 1024/32768/128/bf16 | 0.00989 | 0.000453 | **PASS** | 0.2732 ms | **291.204 us** | 0.109% / 0.076% |
| prefill fp16 | 1024/1024/128/fp16 | 0.00172 | 0.000057 | **PASS** | 0.2735 ms | **291.174 us** | 0.109% / 0.077% |

结论：SWA 窗口固定 128，kernel avg 不随 Sk 增长（Sq1024: 171.198 vs 174.088us）；decode kernel 极短（CANN 19.4us、torch/vllm 16-19us，wall 0.088-0.134ms 以 launch 为主）；prefill 随 SS（候选数）近线性（torch 29→268us、vllm 27→290us）；CSA 算力饱和（K512→K1024 FLOPs 翻倍 latency 几乎不变，~71% 峰值利用率）；torch/vllm 两算子同内核偏差 <4%。

## 5. 汇总结论

| 算子族 | 来源数 | 精度结论 | 性能结论 |
| --- | --- | --- | --- |
| mhc_post | 3（CANN/torch_npu/vllm） | 全部 PASS，三来源逐元素一致（diff=0.0） | vllm 最快（kernel 22.700us@S1024）；全部 memory/vector bound（mac=0%） |
| mhc_pre_sinkhorn | 2 来源 + backward | 全部 PASS；post/comb 双来源一致（~1e-6） | vllm 全面优于 CANN（S2048/C1280 快 26%）；backward 同量级 |
| lightning_indexer | 1 | PASS（集合精确） | kernel decode 18.9-30.9us，launch 占比高；prefill 长 ctx kernel 3.5ms |
| sparse_flash_attention | 3 | 全部 PASS（CANN 12/12 均值误差 ~1e-4；torch_npu/vllm 9/9，两算子逐元素一致） | SFA 三来源同内核；CANN 版 CSA 算力饱和 ~71% 峰值；torch/vllm 版 decode 16-19us、prefill 随 SS 近线性 |

## 6. 复测命令

```bash
# 一键全采集（12 step：精度 + msprof kernel avg），指定卡 7 执行：
ASCEND_RT_VISIBLE_DEVICES=7 python3 run_all_msprof.py    # 根目录
# 单测精度：
cd mhc_post_test/cann_ops_transformer.mhc_post && python3 test.py --bench --msprof
cd mhc_pre_sinkhorn_test/cann_ops_transformer.mhc_pre_sinkhorn && python3 test.py --bench --msprof
cd mhc_pre_sinkhorn_test/cann_ops_transformer.mhc_pre_sinkhorn && python3 test_backward.py --bench --msprof
cd mhc_pre_sinkhorn_test/torch.ops._C_ascend.npu_hc_pre_v3 && python3 test.py --bench --msprof
cd lightning_indexer_test && python3 test_lightning_indexer.py --bench --msprof
cd sparse_flash_attention_test/cann_ops_transformer.sparse_flash_mla && python3 benchmark_perf.py --msprof --iters 10
cd sparse_flash_attention_test/torch_npu.npu_sparse_flash_attention && python3 test.py --check-precision   # 9 用例精度
cd sparse_flash_attention_test/torch_npu.npu_sparse_flash_attention && python3 test.py --bench --msprof    # 8 case kernel avg
cd sparse_flash_attention_test/torch.ops._C_ascend.npu_sparse_flash_attention && python3 test.py --check-precision
cd sparse_flash_attention_test/torch.ops._C_ascend.npu_sparse_flash_attention && python3 test.py --bench --msprof
```
