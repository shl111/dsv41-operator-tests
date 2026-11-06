# coding=utf-8
"""npu_lightning_indexer (LightningIndexer) 精度与语义测试

被测算子: torch_npu.npu_lightning_indexer (A2 MLA 同族内核)
参考实现: 依据 CANN 文档与 A2 AscendC 内核源码推导的 PyTorch float32 参考:
  - 打分: score(s2) = Σ_h w[s1,h] * <q[s1,h], k[s2]>   (weights [T1,N1], key 共享单 head)
  - 输出: [B, S1, 1, sparse_count] int32 索引, 每 query 的 topK 位置
  - mode0 (defaultMask): 无 mask, 全 key 参与打分
  - mode3 (rightDownCausal): 每 query 有效 key 前缀数 = S2 - S1 + i + 1 (索引 0..S2-S1+i), 超出无效
  - 无效位填 -1

运行: python3 test_lightning_indexer.py   (需 Atlas A2 NPU + torch_npu)
"""

import argparse
import csv
import math
import os
import random
import time

import torch
import torch_npu

NEG = -1e9


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


def ref_score(q, k, w):
    """score[S1,S2] = Σ_h w[s1,h] * <q[s1,h], k[s2]>, key 单 head。"""
    qf = q.float().cpu()
    kf = k.float().cpu()
    wf = w.float().cpu()
    if kf.dim() == 4:
        k1 = kf.squeeze(2)[0]  # [S2, D] (N2=1, B=1)
    else:
        k1 = kf[:, 0]
    return (qf[0] * wf[0].unsqueeze(-1)).sum(1) @ k1.T  # [S1, S2]


def ref_indices(sc, K, mode, S1, S2):
    """每 query topK 索引集合 (+ mode3 mask + -1 填充)。"""
    outs = []
    for i in range(S1):
        row = sc[i].clone()
        if mode == 3:
            lim = S2 - S1 + i + 1  # 有效 key 数 (0..S2-S1+i)
            row[lim:] = NEG
        sv, si = row.sort(descending=True)
        valid = [int(si[k].item()) for k in range(K) if sv[k].item() > NEG + 1]
        outs.append(sorted(valid) + [-1] * (K - len(valid)))
    return outs


def _report(name, got, ref, tol_abs=0.03, tol_rel=0.05):
    ok = got == ref
    print(f"{name}: got={got} ref={ref} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_deterministic(dtype, K):
    """确定性 one-hot + 显式 weights: 打分公式与 mode3 mask 精确验证。"""
    torch.manual_seed(0)
    S1, S2, N1, D = 4, 8, 2, 128
    q = torch.zeros(1, S1, N1, D, dtype=dtype, device="npu")
    k = torch.zeros(1, S2, 1, D, dtype=dtype, device="npu")
    w = torch.zeros(1, S1, N1, dtype=dtype, device="npu")
    for s1 in range(S1):
        for h in range(N1):
            q[0, s1, h, s1 * N1 + h] = 1.0
            w[0, s1, h] = s1 + 1 + h
    for s2 in range(S2):
        k[0, s2, 0, s2] = 1.0
    out0 = torch_npu.npu_lightning_indexer(q, k, w, sparse_count=K, sparse_mode=0)
    out3 = torch_npu.npu_lightning_indexer(q, k, w, sparse_count=K, sparse_mode=3)
    torch_npu.npu.synchronize()
    got0 = out0[0][0, :, 0, :].tolist()
    got3 = out3[0][0, :, 0, :].tolist()
    sc = ref_score(q, k, w)
    ref0 = ref_indices(sc, K, 0, S1, S2)
    ref3 = ref_indices(sc, K, 3, S1, S2)
    ok0 = all(sorted(r) == sorted(g) for r, g in zip(ref0, got0))
    ok3 = all(sorted(r) == sorted(g) for r, g in zip(ref3, got3))
    print(f"[deterministic K={K}] mode0 PASS={ok0}, mode3 PASS={ok3}")
    if not ok0:
        for i, (g, r) in enumerate(zip(got0, ref0)):
            if sorted(g) != sorted(r):
                print(f"  mode0 i={i} got={g} ref={r}")
    if not ok3:
        for i, (g, r) in enumerate(zip(got3, ref3)):
            if sorted(g) != sorted(r):
                print(f"  mode3 i={i} got={g} ref={r}")
    return ok0 and ok3


def run_random_range(dtype, trials=20):
    """随机 shape: 索引合法性、无越界、mode3 有效前缀内、无效 -1。"""
    allok = True
    for t in range(trials):
        S1 = random.randint(2, 8)
        S2 = random.randint(S1, 16)
        N1 = random.randint(1, 4)
        D = 128
        K = random.randint(1, 6)
        q = torch.randn(1, S1, N1, D, dtype=dtype, device="npu")
        k = torch.randn(1, S2, 1, D, dtype=dtype, device="npu")
        w = torch.randn(1, S1, N1, dtype=dtype, device="npu")
        for mode in (0, 3):
            out = torch_npu.npu_lightning_indexer(q, k, w, sparse_count=K, sparse_mode=mode)
            torch_npu.npu.synchronize()
            got = out[0][0, :, 0, :].tolist()
            for i in range(S1):
                valid = [v for v in got[i] if v != -1]
                if len(valid) != min(K, S2 if mode == 0 else S2 - S1 + i + 1):
                    allok = False
                    print(f"  trial{t} mode{mode} i={i}: valid count {len(valid)} != expected")
                for v in valid:
                    if not (0 <= v < S2):
                        allok = False
                        print(f"  trial{t} mode{mode} i={i}: index {v} out of range [0,{S2})")
                    if mode == 3 and v > S2 - S1 + i:
                        allok = False
                        print(f"  trial{t} mode{mode} i={i}: index {v} beyond causal prefix {S2-S1+i}")
    print(f"[random {trials} trials] range/mask validity: {'PASS' if allok else 'FAIL'}")
    return allok


def run_tnd_pa(dtype):
    """生产范式 TND/PA_BSND: 确定性验证 + 范围验证。"""
    torch.manual_seed(0)
    T1, T2, N1, D, BS = 8, 32, 4, 128, 128
    q = torch.zeros(T1, N1, D, dtype=dtype, device="npu")
    k = torch.zeros((T2 + BS - 1) // BS, BS, 1, D, dtype=dtype, device="npu")
    w = torch.zeros(T1, N1, dtype=dtype, device="npu")
    for s1 in range(T1):
        for h in range(N1):
            q[s1, h, s1 * N1 + h] = 1.0
            w[s1, h] = s1 + 1 + h
    for s2 in range(T2):
        k[s2 // BS, s2 % BS, 0, s2] = 1.0
    aslq = torch.tensor([T1], dtype=torch.int32, device="npu")
    aslk = torch.tensor([T2], dtype=torch.int32, device="npu")
    bt = torch.arange((T2 + BS - 1) // BS, dtype=torch.int32, device="npu").unsqueeze(0)
    K = 4
    out0 = torch_npu.npu_lightning_indexer(
        q, k, w, actual_seq_lengths_query=aslq, actual_seq_lengths_key=aslk, block_table=bt,
        layout_query="TND", layout_key="PA_BSND", sparse_count=K, sparse_mode=0)
    out3 = torch_npu.npu_lightning_indexer(
        q, k, w, actual_seq_lengths_query=aslq, actual_seq_lengths_key=aslk, block_table=bt,
        layout_query="TND", layout_key="PA_BSND", sparse_count=K, sparse_mode=3)
    torch_npu.npu.synchronize()
    got0 = out0[0][:, 0, :].tolist()
    got3 = out3[0][:, 0, :].tolist()
    sc = ref_score(q.unsqueeze(0), k, w.unsqueeze(0))
    ref0 = ref_indices(sc, K, 0, T1, T2)
    ref3 = ref_indices(sc, K, 3, T1, T2)
    ok0 = all(sorted(r) == sorted(g) for r, g in zip(ref0, got0))
    ok3 = all(sorted(r) == sorted(g) for r, g in zip(ref3, got3))
    print(f"[TND/PA_BSND K={K}] mode0 PASS={ok0}, mode3 PASS={ok3}")
    if not ok0:
        for i, (g, r) in enumerate(zip(got0, ref0)):
            if sorted(g) != sorted(r):
                print(f"  mode0 i={i} got={g} ref={r}")
    if not ok3:
        for i, (g, r) in enumerate(zip(got3, ref3)):
            if sorted(g) != sorted(r):
                print(f"  mode3 i={i} got={g} ref={r}")
    return ok0 and ok3


def run_bench(dtype, msprof=False):
    """生产 shape 性能（TND/PA_BSND, sparse_count=2048）。"""
    import math
    cases = [("decode T2=4096", 1, 4096), ("decode T2=32768", 1, 32768),
             ("prefill T2=1024", 1024, 1024), ("prefill T2=32768", 1024, 32768)]
    for name, T1, T2 in cases:
        N1, D, K, BS = 32, 128, 2048, 128
        nb = (T2 + BS - 1) // BS
        q = torch.randn(T1, N1, D, dtype=dtype, device="npu")
        k = torch.randn(nb, BS, 1, D, dtype=dtype, device="npu")
        w = torch.randn(T1, N1, dtype=dtype, device="npu")
        aslq = torch.tensor([T1], dtype=torch.int32, device="npu")
        aslk = torch.tensor([T2], dtype=torch.int32, device="npu")
        bt = torch.arange(nb, dtype=torch.int32, device="npu").unsqueeze(0)
        fn = lambda: torch_npu.npu_lightning_indexer(
            q, k, w, actual_seq_lengths_query=aslq, actual_seq_lengths_key=aslk, block_table=bt,
            layout_query="TND", layout_key="PA_BSND", sparse_count=K, sparse_mode=3)
        if msprof:
            msprof_bench(fn, f"lightning_indexer {name}", f"./prof_msprof/{name.replace(' ', '_')}")
        else:
            for _ in range(3):
                fn()
            torch_npu.npu.synchronize()
            s = torch.npu.Event(True)
            e = torch.npu.Event(True)
            s.record()
            iters = 20
            for _ in range(iters):
                fn()
            e.record()
            torch_npu.npu.synchronize()
            print(f"[bench {name}] {s.elapsed_time(e) / iters:.4f} ms/iter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--msprof", action="store_true",
                    help="msprof 精准测 kernel 时间+利用率 (自动 warmup)")
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch_npu.npu.set_device(0)

    results = []
    results.append(("deterministic K=2", run_deterministic(dtype, 2)))
    results.append(("deterministic K=4", run_deterministic(dtype, 4)))
    results.append(("TND/PA_BSND production", run_tnd_pa(dtype)))
    results.append(("random range/mask", run_random_range(dtype, args.trials)))
    if args.bench:
        run_bench(dtype, msprof=args.msprof)
    total = len(results)
    passed = sum(ok for _, ok in results)
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print(f"== SUMMARY: {passed}/{total} PASS ==")


if __name__ == "__main__":
    main()
