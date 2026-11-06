import argparse
import csv
import math
import os
import time

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

# 精度用例矩阵（TND + PA_BSND, 离散 TopK 稀疏注意力, A2 MLA 内核约束: N1=1/head_dim=512）
# (label, Tq, Tkv, SS, dtype)
PREC_CASES = [
    ("SWA decode  SS=8   bf16", 1, 128, 8, "bf16"),
    ("SWA decode  SS=128 bf16", 1, 4096, 128, "bf16"),
    ("SWA decode  SS=128 bf16 long", 1, 32768, 128, "bf16"),
    ("SWA prefill SS=8   bf16", 64, 128, 8, "bf16"),
    ("SWA prefill SS=32  bf16", 256, 256, 32, "bf16"),
    ("SWA prefill SS=128 bf16", 1024, 1024, 128, "bf16"),
    ("SWA prefill SS=128 bf16 long", 1024, 32768, 128, "bf16"),
    ("SWA prefill SS=8   fp16", 64, 128, 8, "fp16"),
    ("SWA prefill SS=128 fp16", 1024, 1024, 128, "fp16"),
]

# 性能 case 矩阵（msprof kernel avg + Event wall-time 双口径）
PERF_CASES = [
    ("SWA decode  SS=8   bf16", 1, 128, 8, "bf16"),
    ("SWA decode  SS=128 bf16", 1, 4096, 128, "bf16"),
    ("SWA decode  SS=128 bf16 long", 1, 32768, 128, "bf16"),
    ("SWA prefill SS=8   bf16", 64, 128, 8, "bf16"),
    ("SWA prefill SS=32  bf16", 256, 256, 32, "bf16"),
    ("SWA prefill SS=128 bf16", 1024, 1024, 128, "bf16"),
    ("SWA prefill SS=128 bf16 long", 1024, 32768, 128, "bf16"),
    ("SWA prefill SS=128 fp16", 1024, 1024, 128, "fp16"),
]

IMAX = (1 << 63) - 1
SCALE = 1.0 / math.sqrt(512)
BS = 128  # PA_BSND block_size


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


def build_tnd_pa(dtype, Tq, Tkv, N1, N2, SS, seed=42, block_size=BS):
    """构造 TND query + PA_BSND kv（sparse_indices 为每 query 最近 SS 个候选的离散 TopK）。"""
    g = torch.Generator().manual_seed(seed)
    q_np = torch.randn(Tq, N1, 512, dtype=torch.float32, generator=g)
    kv_np = torch.randn((Tkv + block_size - 1) // block_size, block_size, N2, 512,
                        dtype=torch.float32, generator=g)
    qr_np = torch.randn(Tq, N1, 64, dtype=torch.float32, generator=g)
    kr_np = torch.randn((Tkv + block_size - 1) // block_size, block_size, N2, 64,
                        dtype=torch.float32, generator=g)
    si = torch.full((Tq, N2, SS), -1, dtype=torch.int32)
    for t in range(Tq):
        lo = max(0, t + 1 - SS)
        sel = list(range(lo, min(t + 1, Tkv)))[:SS]
        for j, idx in enumerate(sel):
            si[t, 0, j] = idx
    nb = (Tkv + block_size - 1) // block_size
    bt = torch.arange(nb, dtype=torch.int32).unsqueeze(0)
    aslq = torch.tensor([Tq], dtype=torch.int32)
    aslk = torch.tensor([Tkv], dtype=torch.int32)
    return (q_np, kv_np, qr_np, kr_np, si, bt, aslq, aslk)


def run_op(q, kv, qr, kr, si, bt, aslq, aslk):
    # sfa_v1.py:224 生产范式: TND/PA_BSND, sparse_mode=3, attention_mode=2
    # A2 MLA 内核约束: query_rope/key_rope 必传真实张量 (EZ0037)，故传入 rope
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        query=q, key=kv, value=kv, sparse_indices=si, scale_value=SCALE,
        sparse_block_size=1, block_table=bt,
        actual_seq_lengths_query=aslq, actual_seq_lengths_kv=aslk,
        query_rope=qr, key_rope=kr,
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2,
        return_softmax_lse=False,
    )


def reference(q_np, kv_np, qr_np, kr_np, si, Tq, Tkv):
    """float32 参考：每 query 只对 sparse_indices 有效位做 softmax（与算子语义一致）。"""
    kv_flat = kv_np.reshape(-1, kv_np.shape[2], kv_np.shape[3])[:Tkv]
    kr_flat = kr_np.reshape(-1, kr_np.shape[2], kr_np.shape[3])[:Tkv]
    k = kv_flat[:, 0]
    krv = kr_flat[:, 0]
    v = k
    outs = []
    for t in range(Tq):
        idxs = [int(si[t, 0, j]) for j in range(si.shape[2]) if int(si[t, 0, j]) >= 0]
        sc = (q_np[t, 0] @ k[idxs].T + qr_np[t, 0] @ krv[idxs].T) * SCALE
        p = torch.softmax(sc, dim=-1)
        outs.append(p @ v[idxs])
    return torch.stack(outs)


def run_precision_cases():
    """9 用例精度套件（参照 CANN sparse_flash_mla test_precision.py 结构）。"""
    results = []
    for label, Tq, Tkv, SS, dt in PREC_CASES:
        dtype = torch.bfloat16 if dt == "bf16" else torch.float16
        q_np, kv_np, qr_np, kr_np, si, bt, aslq, aslk = build_tnd_pa(
            dtype, Tq, Tkv, 1, 1, SS)
        q = q_np.to(dtype=dtype, device="npu")
        kv = kv_np.to(dtype=dtype, device="npu")
        qr = qr_np.to(dtype=dtype, device="npu")
        kr = kr_np.to(dtype=dtype, device="npu")
        si = si.to(device="npu")
        bt = bt.to(device="npu")
        aslq = aslq.to(device="npu")
        aslk = aslk.to(device="npu")
        try:
            out = run_op(q, kv, qr, kr, si, bt, aslq, aslk)
            torch_npu.npu.synchronize()
        except Exception as e:
            print(f"[{label:<32}] ERROR {str(e)[:100]} -> FAIL")
            results.append(False)
            continue
        att = out[0][:, 0, :].float().cpu()
        ref = reference(q_np, kv_np, qr_np, kr_np, si, Tq, Tkv)
        diff = (att - ref).abs()
        ok = diff.max().item() <= 0.03 and diff.mean().item() <= 2e-3
        results.append(ok)
        print(f"[{label:<32}] max_diff={diff.max().item():.5f} "
              f"mean_diff={diff.mean().item():.6f} -> {'PASS' if ok else 'FAIL'}")
    passed = sum(1 for r in results if r)
    print(f"== PRECISION SUMMARY: {passed}/{len(results)} PASS ==")
    return all(results)


def run_perf_cases(msprof=False):
    """性能 case 矩阵：msprof kernel avg 或 Event wall-time 双口径。"""
    for label, Tq, Tkv, SS, dt in PERF_CASES:
        dtype = torch.bfloat16 if dt == "bf16" else torch.float16
        q_np, kv_np, qr_np, kr_np, si, bt, aslq, aslk = build_tnd_pa(
            dtype, Tq, Tkv, 1, 1, SS)
        q = q_np.to(dtype=dtype, device="npu")
        kv = kv_np.to(dtype=dtype, device="npu")
        qr = qr_np.to(dtype=dtype, device="npu")
        kr = kr_np.to(dtype=dtype, device="npu")
        si = si.to(device="npu")
        bt = bt.to(device="npu")
        aslq = aslq.to(device="npu")
        aslk = aslk.to(device="npu")
        fn = lambda: run_op(q, kv, qr, kr, si, bt, aslq, aslk)
        tag = f"Tq{Tq}_Tkv{Tkv}_SS{SS}_{dt}"
        if msprof:
            msprof_bench(fn, f"vllm npu_sparse_flash_attention {label}",
                         f"./prof_msprof/{tag}")
        else:
            iters = 50
            for _ in range(3):
                fn()
            torch_npu.npu.synchronize()
            s = torch.npu.Event(True)
            e = torch.npu.Event(True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch_npu.npu.synchronize()
            print(f"[{label:<32}] {s.elapsed_time(e) / iters:.4f} ms/iter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-precision", action="store_true",
                    help="跑 9 用例精度套件")
    ap.add_argument("--bench", action="store_true",
                    help="跑性能 case 矩阵（Event wall-time）")
    ap.add_argument("--msprof", action="store_true",
                    help="性能矩阵用 msprof 精准测 kernel avg + 利用率")
    args = ap.parse_args()

    dtype = torch.bfloat16
    # enable_custom_op() 必须在 set_device() 之前调用（utils.py:546），
    # 否则 CANN RTS 提前初始化，SparseFlashAttention tiling 语义异常导致精度 FAIL。
    assert enable_custom_op(), "vllm-ascend custom op enable failed"
    torch_npu.npu.set_device(0)
    ok = True
    if args.check_precision:
        ok = ok and run_precision_cases()
    if args.bench:
        run_perf_cases(msprof=args.msprof)
    if not args.check_precision and not args.bench:
        # 默认: 全跑（精度 + Event 性能）
        ok = ok and run_precision_cases()
        run_perf_cases(msprof=False)
    print(f"== OVERALL: {'PASS' if ok else 'FAIL'} ==")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
