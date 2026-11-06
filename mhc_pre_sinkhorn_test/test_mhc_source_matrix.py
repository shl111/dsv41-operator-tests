# coding=utf-8
"""mhc_pre_sinkhorn 来源矩阵实测（Atlas A2）

对比同一 MHC 算子的两个来源:
  1. cann_ops_transformer.ops.mhc_pre_sinkhorn  (CANN 融合版)  -> A2 可运行
  2. torch_npu.npu_mhc_pre / npu_mhc_sinkhorn    (torch_npu 分步版) -> A2 不可运行 (Apt 后端)

运行: python3 test_mhc_source_matrix.py
"""

import torch
import torch_npu
from cann_ops_transformer.ops import mhc_pre_sinkhorn

B, S, N, C = 1, 4, 4, 1280


def main():
    torch_npu.npu.set_device(0)
    x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
    phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
    alpha = torch.randn(3, dtype=torch.float32).npu()
    bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()

    # 来源1: CANN 融合版
    try:
        hin, h_post, h_res = mhc_pre_sinkhorn(x, phi, alpha, bias, N, 20)
        torch_npu.npu.synchronize()
        print(f"1. cann_ops_transformer.mhc_pre_sinkhorn: "
              f"hin{tuple(hin.shape)} hPost{tuple(h_post.shape)} hRes{tuple(h_res.shape)} RUNNABLE")
    except Exception as e:
        print(f"1. cann_ops_transformer.mhc_pre_sinkhorn: FAIL ({str(e)[:100]})")

    # 来源2: torch_npu 原生分步版
    for name, call in [
        ("npu_mhc_pre", lambda: torch_npu.npu_mhc_pre(x, phi, alpha, bias, out_flag=0)),
        ("npu_mhc_sinkhorn", lambda: torch_npu.npu_mhc_sinkhorn(
            torch.randn(B, S, N, N, dtype=torch.float32).npu(), num_iters=20, out_flag=0)),
        ("npu_mhc_post", lambda: torch_npu.npu_mhc_post(
            torch.randn(B, S, N, C, dtype=torch.bfloat16).npu(),
            torch.randn(B, S, N, N, dtype=torch.float32).npu(),
            torch.randn(B, S, C, dtype=torch.bfloat16).npu(),
            torch.randn(B, S, N, dtype=torch.float32).npu())),
    ]:
        try:
            out = call()
            torch_npu.npu.synchronize()
            print(f"2. {name}: RUNNABLE -> {[(tuple(t.shape), str(t.dtype)) for t in out]}")
        except Exception as e:
            print(f"2. {name}: NOT RUNNABLE on A2 ({str(e)[:80]})")

    print("== matrix done ==")


if __name__ == "__main__":
    main()
