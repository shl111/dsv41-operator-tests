#!/usr/bin/env python3
"""顺序执行所有算子测试的 msprof 精准测量 (--bench --msprof)。"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("mhc_post CANN", f"{ROOT}/mhc_post_test/cann_ops_transformer.mhc_post", ["python3", "-u", "test.py", "--bench", "--msprof"]),
    ("mhc_post torch_npu", f"{ROOT}/mhc_post_test/torch_npu.npu_mhc_post", ["python3", "-u", "test.py", "--bench", "--msprof"]),
    ("mhc_post vllm", f"{ROOT}/mhc_post_test/torch.ops._C_ascend.npu_hc_post", ["python3", "-u", "test.py", "--bench", "--msprof"]),
    ("mhc_pre_sinkhorn CANN", f"{ROOT}/mhc_pre_sinkhorn_test/cann_ops_transformer.mhc_pre_sinkhorn", ["python3", "-u", "test.py", "--bench", "--msprof"]),
    ("mhc_pre_sinkhorn backward", f"{ROOT}/mhc_pre_sinkhorn_test/cann_ops_transformer.mhc_pre_sinkhorn", ["python3", "-u", "test_backward.py", "--bench", "--msprof"]),
    ("mhc_pre_sinkhorn vllm hc_pre_v3", f"{ROOT}/mhc_pre_sinkhorn_test/torch.ops._C_ascend.npu_hc_pre_v3", ["python3", "-u", "test.py", "--bench", "--msprof"]),
    ("lightning_indexer", f"{ROOT}/lightning_indexer_test", ["python3", "-u", "test_lightning_indexer.py", "--bench", "--msprof"]),
    ("SFA torch_npu", f"{ROOT}/sparse_flash_attention_test/torch_npu.npu_sparse_flash_attention", ["python3", "-u", "test.py", "--check-precision"]),
    ("SFA torch_npu msprof", f"{ROOT}/sparse_flash_attention_test/torch_npu.npu_sparse_flash_attention", ["python3", "-u", "test.py", "--bench", "--msprof"]),
    ("SFA vllm", f"{ROOT}/sparse_flash_attention_test/torch.ops._C_ascend.npu_sparse_flash_attention", ["python3", "-u", "test.py", "--check-precision"]),
    ("SFA vllm msprof", f"{ROOT}/sparse_flash_attention_test/torch.ops._C_ascend.npu_sparse_flash_attention", ["python3", "-u", "test.py", "--bench", "--msprof"]),
    ("SFA CANN benchmark_perf", f"{ROOT}/sparse_flash_attention_test/cann_ops_transformer.sparse_flash_mla", ["python3", "-u", "benchmark_perf.py", "--msprof", "--iters", "10"]),
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    for i, (tag, cwd, cmd) in enumerate(STEPS):
        if n >= 0 and i != n:
            continue
        print(f"\n########## [{i}] {tag} ##########", flush=True)
        p = subprocess.run(cmd, cwd=cwd, capture_output=False)
        print(f"########## [{i}] {tag} exit={p.returncode} ##########", flush=True)


if __name__ == "__main__":
    main()
