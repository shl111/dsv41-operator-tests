# coding=utf-8
"""LightningIndexer 来源矩阵实测（Atlas A2）

对比同一 LightningIndexer 的两个来源:
  1. torch_npu.npu_lightning_indexer (torch_npu 原生)            -> A2 可运行
  2. torch.ops._C_ascend.npu_lightning_indexer_quant
     (vllm-ascend 自定义内核, int8 量化版)                      -> A2 不可运行

运行: python3 test_lightning_indexer_source_matrix.py
"""

import sys

import torch
import torch_npu


def main():
    torch_npu.npu.set_device(0)

    # 来源1: torch_npu 原生 (主接口, 已确认可运行)
    B, S1, S2, N1, D, K = 1, 4, 8, 2, 128, 3
    q = torch.randn(B, S1, N1, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(B, S2, 1, D, dtype=torch.bfloat16, device="npu")
    w = torch.randn(B, S1, N1, dtype=torch.bfloat16, device="npu")
    idx = torch_npu.npu_lightning_indexer(q, k, w, sparse_count=K, sparse_mode=0)[0]
    torch_npu.npu.synchronize()
    print(f"1. torch_npu.npu_lightning_indexer: {tuple(idx.shape)} {idx.dtype} RUNNABLE")

    # 来源2: vllm-ascend _C_ascend 自定义内核 (int8 quant 版)
    sys.path.insert(0, "/vllm-workspace/vllm-ascend")
    from vllm_ascend.utils import enable_custom_op
    enable_custom_op()
    has_ascend = hasattr(torch.ops, "_C_ascend")
    print(f"   _C_ascend 加载: {has_ascend}, "
          f"npu_lightning_indexer_quant: {hasattr(torch.ops._C_ascend, 'npu_lightning_indexer_quant') if has_ascend else False}")
    try:
        T1, T2, N1b, DB, BS, Kb = 8, 16, 4, 128, 128, 4
        nb = (T2 + BS - 1) // BS
        qq = torch.randint(-16, 16, (T1, N1b, DB), dtype=torch.int8, device="npu")
        kk = torch.randint(-16, 16, (nb, BS, 1, DB), dtype=torch.int8, device="npu")
        ww = torch.randn(T1, N1b, dtype=torch.float16, device="npu")
        q_scale = torch.randn(T1, N1b, dtype=torch.float32, device="npu")
        k_scale = torch.randn(T2, DB, dtype=torch.float32, device="npu")
        aslq = torch.tensor([T1], dtype=torch.int32, device="npu")
        aslk = torch.tensor([T2], dtype=torch.int32, device="npu")
        bt = torch.arange(nb, dtype=torch.int32, device="npu").unsqueeze(0)
        idx2 = torch.ops._C_ascend.npu_lightning_indexer_quant(
            query=qq, key=kk, weights=ww, query_dequant_scale=q_scale, key_dequant_scale=k_scale,
            actual_seq_lengths_query=aslq, actual_seq_lengths_key=aslk, block_table=bt,
            query_quant_mode=0, key_quant_mode=0, layout_query="TND", layout_key="PA_BSND",
            sparse_count=Kb, sparse_mode=3)
        torch_npu.npu.synchronize()
        print(f"2. _C_ascend.npu_lightning_indexer_quant: {tuple(idx2.shape)} RUNNABLE")
    except Exception as e:
        print(f"2. _C_ascend.npu_lightning_indexer_quant: NOT RUNNABLE on A2 ({str(e)[:90]})")

    print("== matrix done ==")


if __name__ == "__main__":
    main()
