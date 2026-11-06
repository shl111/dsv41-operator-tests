# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# This file is a part of the CANN Open Software.
# See LICENSE in the root of the software repository for the full text of the License.
"""sparse_flash_mla (SparseFlashAttention) 性能测试

方法:
  - 单算子调用，torch.npu.Event 计时（不含数据构造与 metadata 生成）
  - warmup 后连续执行 N 次取中位数/均值
  - 计算有效 FLOPs 与 GFLOPS: FLOPs = 2*Nq*Sq*S_attended*D*2 (QK^T + PV)
    S_attended = min(Sk, win_size) [+ K(cmp_topk) | + cmp_len]

运行: python3 benchmark_perf.py [--iters 50] [--warmup 10]   (需 Atlas A2/A3 NPU)
"""

import argparse
import csv
import math
import os
import statistics
import time

import torch
import torch_npu

import cann_ops_transformer

D = 512


def msprof_bench(fn, tag, save_path, warmup=5, active=10):
    """带 warmup 的 msprof 精准测量 (Level1 + PipeUtilization)。"""
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    # 清理本 step 旧产物, 避免 os.walk 读到上一次运行留下的 kernel_details.csv
    import shutil
    shutil.rmtree(save_path, ignore_errors=True)
    ecfg = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    )
    os.makedirs(save_path, exist_ok=True)
    try:
        with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU,
                            torch_npu.profiler.ProfilerActivity.CPU],
                with_stack=False, record_shapes=False, profile_memory=False,
                experimental_config=ecfg,
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=active, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(save_path)) as prof:
            for _ in range(active):
                fn()
                torch_npu.npu.synchronize()
                prof.step()
            prof.step()
    except Exception as e:
        print(f"  [{tag}] WARN profiler raised {e!r}; awaiting async analyse...")
    out = None
    for _ in range(720):
        cand = [os.path.join(dp, "ASCEND_PROFILER_OUTPUT") for dp, dn, _ in os.walk(save_path)
                if "ASCEND_PROFILER_OUTPUT" in dn
                and os.path.isfile(os.path.join(dp, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv"))]
        if cand:
            out = cand[0]
            break
        time.sleep(1)
    if not out:
        print(f"  [{tag}] kernel_details.csv not produced (msprof analyse failed?)")
        return
    # 官方聚合统计: op_statistic.csv 直接给出该算子 Avg/Min/Max/Total（权威口径）
    ost = os.path.join(out, "op_statistic.csv")
    if not os.path.exists(ost):
        print(f"  [{tag}] op_statistic.csv missing (msprof analyse failed?)")
        return
    with open(ost) as f:
        ops = list(csv.DictReader(f))
    # 利用率: kernel_details.csv 的 mac/vec 按 kernel 名首见采样
    util = {}
    kd = os.path.join(out, "kernel_details.csv")
    if os.path.exists(kd):
        with open(kd) as f:
            rows = list(csv.DictReader(f))
        for x in rows:
            n = x["Name"]
            if n in util or "aic_mac_ratio" not in x:
                continue
            if x["aic_mac_ratio"] not in ("", "N/A"):
                util[n] = (x["aic_mac_ratio"], x.get("aiv_vec_ratio", ""))
    print(f"== {tag}: {len(ops)} op type(s), msprof op_statistic ==")
    for o in sorted(ops, key=lambda v: -float(v["Avg Time(us)"])):
        uu = ""
        for n, (m, v) in util.items():
            if o["OP Type"] in n or n in o["OP Type"]:
                uu = f" mac={m}% vec={v}%"
                break
        print(f"  {o['OP Type']:<44} avg={float(o['Avg Time(us)']):9.3f} us  "
              f"min={float(o['Min Time(us)']):9.3f}  max={float(o['Max Time(us)']):9.3f}  "
              f"total={float(o['Total Time(us)']):9.3f} us  x{o['Count']}  core={o['Core Type']}{uu}")
    print(f"  TOTAL device: {sum(float(o['Total Time(us)']) for o in ops):.3f} us")


def _timeit(fn, iters):
    torch_npu.npu.synchronize()
    times = []
    for _ in range(iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch_npu.npu.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def bench_swa_bsnd(B, Sq, Sk, Nq, dtype, iters, msprof=False):
    q = torch.randn(B, Sq, Nq, D, dtype=dtype, device="npu")
    ori_kv = torch.randn(B, Sk, 1, D, dtype=dtype, device="npu")
    sinks = torch.zeros(Nq, dtype=torch.float32, device="npu")
    md = cann_ops_transformer.sparse_flash_mla_metadata(
        Nq, 1, D, batch_size=B, max_seqlen_q=Sq, max_seqlen_ori_kv=Sk,
        ori_topk=0, cmp_topk=0, cmp_ratio=1, ori_mask_mode=4, cmp_mask_mode=3,
        ori_win_left=127, ori_win_right=0, layout_q="BSND", layout_kv="BSND",
        has_ori_kv=True, has_cmp_kv=False)

    def run():
        cann_ops_transformer.sparse_flash_mla(
            q, ori_kv=ori_kv, sinks=sinks, metadata=md,
            softmax_scale=1.0 / math.sqrt(D), cmp_ratio=1, ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=127, ori_win_right=0, layout_q="BSND", layout_kv="BSND",
            return_softmax_lse=False)
    if msprof:
        msprof_bench(run, f"sparse_flash_mla SWA B{B} Sq{Sq} Sk{Sk} N{Nq}",
                     f"./prof_msprof/swa_B{B}_Sq{Sq}_Sk{Sk}")
        return None, 0.0
    for _ in range(3):
        run()
    ms = _timeit(run, iters)
    s_att = min(Sk, 128)
    flops = 2.0 * B * Sq * Nq * s_att * D * 2
    return ms, flops


def bench_csa_tnd(q_lens, ori_lens, cmp_lens, residuals, Nq, K, dtype, iters, msprof=False):
    q = torch.randn(sum(q_lens), Nq, D, dtype=dtype, device="npu")
    ori_kv = torch.randn(sum(ori_lens), 1, D, dtype=dtype, device="npu")
    cmp_kv = torch.randn(sum(cmp_lens), 1, D, dtype=dtype, device="npu")
    sinks = torch.zeros(Nq, dtype=torch.float32, device="npu")
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="npu")
    cu_ori = torch.tensor([0] + list(torch.tensor(ori_lens).cumsum(0)), dtype=torch.int32, device="npu")
    cu_cmp = torch.tensor([0] + list(torch.tensor(cmp_lens).cumsum(0)), dtype=torch.int32, device="npu")
    residual = torch.tensor(residuals, dtype=torch.int32, device="npu")
    cmp_indices = torch.full((sum(q_lens), 1, K), -1, dtype=torch.int32, device="npu")
    for row in range(sum(q_lens)):
        n_sel = min(int(((torch.arange(cmp_lens[0], device="npu").float() * 4 + 3) <= row).sum()), K)
        if n_sel > 0:
            cmp_indices[row, 0, :n_sel] = torch.arange(n_sel, device="npu")
    md = cann_ops_transformer.sparse_flash_mla_metadata(
        Nq, 1, D, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_ori, cu_seqlens_cmp_kv=cu_cmp,
        max_seqlen_q=max(q_lens), max_seqlen_ori_kv=max(ori_lens), max_seqlen_cmp_kv=max(cmp_lens),
        cmp_residual_kv=residual, ori_topk=0, cmp_topk=K, cmp_ratio=4,
        ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
        layout_q="TND", layout_kv="TND", has_ori_kv=True, has_cmp_kv=True)

    def run():
        cann_ops_transformer.sparse_flash_mla(
            q, ori_kv=ori_kv, cmp_kv=cmp_kv, cmp_sparse_indices=cmp_indices,
            cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_ori, cu_seqlens_cmp_kv=cu_cmp,
            cmp_residual_kv=residual, sinks=sinks, metadata=md,
            softmax_scale=1.0 / math.sqrt(D), cmp_ratio=4, ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=127, ori_win_right=0, layout_q="TND", layout_kv="TND",
            return_softmax_lse=False)
    if msprof:
        msprof_bench(run, f"sparse_flash_mla CSA q{sum(q_lens)} K{K} N{Nq}",
                     f"./prof_msprof/csa_q{sum(q_lens)}_K{K}")
        return None, 0.0
    for _ in range(3):
        run()
    ms = _timeit(run, iters)
    s_att = min(ori_lens[0], 128) + K
    flops = 2.0 * sum(q_lens) * Nq * s_att * D * 2
    return ms, flops


def bench_hca_tnd(q_lens, ori_lens, cmp_lens, residuals, Nq, dtype, iters, msprof=False):
    q = torch.randn(sum(q_lens), Nq, D, dtype=dtype, device="npu")
    ori_kv = torch.randn(sum(ori_lens), 1, D, dtype=dtype, device="npu")
    cmp_kv = torch.randn(sum(cmp_lens), 1, D, dtype=dtype, device="npu")
    sinks = torch.zeros(Nq, dtype=torch.float32, device="npu")
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device="npu")
    cu_ori = torch.tensor([0] + list(torch.tensor(ori_lens).cumsum(0)), dtype=torch.int32, device="npu")
    cu_cmp = torch.tensor([0] + list(torch.tensor(cmp_lens).cumsum(0)), dtype=torch.int32, device="npu")
    residual = torch.tensor(residuals, dtype=torch.int32, device="npu")
    md = cann_ops_transformer.sparse_flash_mla_metadata(
        Nq, 1, D, cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_ori, cu_seqlens_cmp_kv=cu_cmp,
        max_seqlen_q=max(q_lens), max_seqlen_ori_kv=max(ori_lens), max_seqlen_cmp_kv=max(cmp_lens),
        cmp_residual_kv=residual, ori_topk=0, cmp_topk=0, cmp_ratio=128,
        ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
        layout_q="TND", layout_kv="TND", has_ori_kv=True, has_cmp_kv=True)

    def run():
        cann_ops_transformer.sparse_flash_mla(
            q, ori_kv=ori_kv, cmp_kv=cmp_kv,
            cu_seqlens_q=cu_q, cu_seqlens_ori_kv=cu_ori, cu_seqlens_cmp_kv=cu_cmp,
            cmp_residual_kv=residual, sinks=sinks, metadata=md,
            softmax_scale=1.0 / math.sqrt(D), cmp_ratio=128, ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=127, ori_win_right=0, layout_q="TND", layout_kv="TND",
            return_softmax_lse=False)
    if msprof:
        msprof_bench(run, f"sparse_flash_mla HCA q{sum(q_lens)} N{Nq}",
                     f"./prof_msprof/hca_q{sum(q_lens)}")
        return None, 0.0
    for _ in range(3):
        run()
    ms = _timeit(run, iters)
    s_att = min(ori_lens[0], 128) + cmp_lens[0]
    flops = 2.0 * sum(q_lens) * Nq * s_att * D * 2
    return ms, flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--msprof", action="store_true",
                        help="msprof 精准测 kernel 时间+利用率 (自动 warmup)")
    args = parser.parse_args()
    torch_npu.npu.set_device(0)
    torch.manual_seed(0)
    iters = args.iters
    msprof = args.msprof

    rows = []
    # (label, fn, flops)
    cases = []
    for dtype, dt in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        # SWA decode: q=1 token vs long window kv
        cases.append((f"SWA BSND {dt} B1 Sq1 Sk4096 N64",
                      lambda: bench_swa_bsnd(1, 1, 4096, 64, dtype, iters, msprof)))
        cases.append((f"SWA BSND {dt} B1 Sq128 Sk128 N64",
                      lambda: bench_swa_bsnd(1, 128, 128, 64, dtype, iters, msprof)))
        cases.append((f"SWA BSND {dt} B1 Sq1024 Sk1024 N64",
                      lambda: bench_swa_bsnd(1, 1024, 1024, 64, dtype, iters, msprof)))
        cases.append((f"SWA BSND {dt} B1 Sq1024 Sk32768 N64 (long ctx)",
                      lambda: bench_swa_bsnd(1, 1024, 32768, 64, dtype, iters, msprof)))
        # CSA TND: sliding window + topk compressed
        cases.append((f"CSA TND {dt} q1024 kv1024 cmp256 K512 N64",
                      lambda: bench_csa_tnd([1024], [1024], [256], [0], 64, 512, dtype, iters, msprof)))
        cases.append((f"CSA TND {dt} q1024 kv1024 cmp256 K1024 N64",
                      lambda: bench_csa_tnd([1024], [1024], [256], [0], 64, 1024, dtype, iters, msprof)))
        # HCA TND
        cases.append((f"HCA TND {dt} q1024 kv1024 cmp8 N64",
                      lambda: bench_hca_tnd([1024], [1024], [8], [0], 64, dtype, iters, msprof)))

    print(f"\n{'case':<46}{'latency(ms)':>12}{'GFLOPS':>12}")
    print("-" * 72)
    for label, fn in cases:
        try:
            ms, flops = fn()
            if ms is None:
                continue  # msprof 模式已打印 kernel 摘要
            gflops = flops / (ms * 1e-3) / 1e9
            print(f"{label:<46}{ms:>12.3f}{gflops:>12.1f}")
            rows.append((label, ms, gflops))
        except Exception as ex:  # noqa: BLE001
            print(f"{label:<46}{'ERROR':>12}{str(ex)[:40]:>12}")

    print("\nDONE")


if __name__ == "__main__":
    main()
