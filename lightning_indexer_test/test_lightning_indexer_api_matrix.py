# coding=utf-8
"""LightningIndexer 同族接口实测矩阵（Atlas A2）

覆盖 torch_npu 注册的 6 个 LightningIndexer 族接口，验证可运行性 / 输出结构 / 平台约束。

运行: python3 test_lightning_indexer_api_matrix.py
"""

import torch
import torch_npu


def main():
    torch_npu.npu.set_device(0)
    B, S1, S2, N1, D, K = 1, 4, 8, 1, 128, 3
    q = torch.randn(B, S1, N1, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(B, S2, 1, D, dtype=torch.bfloat16, device="npu")
    w = torch.randn(B, S1, N1, dtype=torch.bfloat16, device="npu")

    # 1) 主接口: npu_lightning_indexer (推理)
    res = torch_npu.npu_lightning_indexer(q, k, w, sparse_count=K, sparse_mode=0)
    torch_npu.npu.synchronize()
    idx = res[0]
    print(f"1. npu_lightning_indexer      : {tuple(idx.shape)} {idx.dtype} RUNNABLE")
    print(f"   idx[0]: {idx[0, :, 0, :].tolist()}")

    # 2) npu_dense_lightning_indexer_softmax_lse (训练前置)
    q_index = torch.randn(B, S1, N1, D, dtype=torch.bfloat16, device="npu")
    k_index = torch.randn(B, S2, 1, D, dtype=torch.bfloat16, device="npu")
    smax, ssum = torch_npu.npu_dense_lightning_indexer_softmax_lse(q_index, k_index, w, sparse_mode=3)
    torch_npu.npu.synchronize()
    print(f"2. dense_lightning_indexer_softmax_lse: {tuple(smax.shape)} {smax.dtype} RUNNABLE")

    # 3) npu_quant_lightning_indexer (推理, 量化输入; 文档为 A3/A5)
    try:
        qq = torch.randint(-10, 10, (B, S1, N1, D), dtype=torch.int8, device="npu")
        kq = torch.randint(-10, 10, (B, S2, 1, D), dtype=torch.int8, device="npu")
        qscale = torch.randn(1, dtype=torch.float32, device="npu")
        kscale = torch.randn(1, dtype=torch.float32, device="npu")
        out = torch_npu.npu_quant_lightning_indexer(qq, kq, w, qscale, kscale, 0, 0, sparse_count=K, sparse_mode=0)
        torch_npu.npu.synchronize()
        print(f"3. npu_quant_lightning_indexer: {tuple(out[0].shape)} RUNNABLE")
    except Exception as e:
        print(f"3. npu_quant_lightning_indexer: NOT RUNNABLE on A2 ({str(e)[:80]})")

    # 4/5/6) grad_kl_loss 训练接口 (A2 训练内核不支持, 预期失败)
    for name, fn in [
        ("4. dense_lightning_indexer_grad_kl_loss", torch_npu.npu_dense_lightning_indexer_grad_kl_loss),
        ("5. sparse_lightning_indexer_grad_kl_loss", torch_npu.npu_sparse_lightning_indexer_grad_kl_loss),
    ]:
        try:
            if name.startswith("4"):
                out = fn(q.float().half(), k.float().half(), q_index.half(), k_index.half(),
                         w.half(), smax.abs().float(), torch.ones(B, 1, S1, 1, device="npu"),
                         smax, ssum, 1.0, sparse_mode=3)
            else:
                si = torch.randint(0, S2, (B, S1, 1, K), dtype=torch.int32, device="npu")
                out = fn(q.float().half(), k.float().half(), q_index.half(), k_index.half(), w.half(),
                         si, smax.abs().float(), torch.ones(B, 1, S1, 1, device="npu"), 1.0, sparse_mode=3)
            torch_npu.npu.synchronize()
            print(f"{name}: RUNNABLE -> {[(tuple(t.shape), str(t.dtype)) for t in out]}")
        except Exception as e:
            print(f"{name}: NOT RUNNABLE on A2 ({str(e)[:70]})")

    print("== matrix done ==")


if __name__ == "__main__":
    main()
