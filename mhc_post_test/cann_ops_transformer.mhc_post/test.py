# coding=utf-8
"""cann_ops_transformer.mhc_post (CANN 融合版) 语义 / 精度 / 性能测试

被测算子: cann_ops_transformer.ops.mhc_post
语义 (CANN 文档): y[i,:] = H^post[i]·h_out[:] + Σ_j H^res[j,i]·x[j,:]
布局: x (B,S,N,D)/(T,N,D) 残差流; h_res (B,S,N,N)/(T,N,N); h_out (B,S,D)/(T,D); h_post (B,S,N)/(T,N)

运行:
  python3 test.py                # 语义精度
  python3 test.py --bench        # Event wall-time
  python3 test.py --bench --msprof   # msprof 精准: kernel 时间 + 利用率 (自动 warmup)
"""

import argparse
import csv
import os
import time

import torch
import torch_npu
from cann_ops_transformer.ops import mhc_post

N, D = 4, 128


def msprof_bench(fn, tag, save_path, warmup=5, active=10):
    """带 warmup 的 msprof 精准测量 (Level1 + PipeUtilization)。

    warmup 在 profiler 外 (图编译/JIT/首调预热), active 次稳态进 profiler 窗口,
    loop 后补一次 prof.step() 收尾。msprof analyse 异步, 轮询等待 kernel_details.csv。
    """
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


def ref(x, h_res, h_out, h_post):
    return (torch.einsum('bsn,bsd->bsnd', h_post.float().cpu(), h_out.float().cpu())
            + torch.einsum('bsjn,bsjd->bsnd', h_res.float().cpu(), x.float().cpu()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--msprof", action="store_true",
                    help="msprof 精准测 kernel 时间+利用率 (自动 warmup)")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    torch.manual_seed(0)
    B, S = 1, 4
    x = torch.randn(B, S, N, D, dtype=torch.bfloat16).npu()
    h_res = torch.randn(B, S, N, N, dtype=torch.float32).npu()
    h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
    h_post = torch.randn(B, S, N, dtype=torch.float32).npu()
    y = mhc_post(x, h_res, h_out, h_post)
    torch_npu.npu.synchronize()
    d = (y.float().cpu() - ref(x, h_res, h_out, h_post)).abs().max().item()
    print(f"[BSND] {tuple(y.shape)} {y.dtype} vs_ref={d:.4f} -> {'PASS' if d < 0.05 else 'FAIL'}")

    x3 = torch.randn(B * S, N, D, dtype=torch.bfloat16).npu()
    hr3 = torch.randn(B * S, N, N, dtype=torch.float32).npu()
    ho3 = torch.randn(B * S, D, dtype=torch.bfloat16).npu()
    hp3 = torch.randn(B * S, N, dtype=torch.float32).npu()
    y3 = mhc_post(x3, hr3, ho3, hp3)
    torch_npu.npu.synchronize()
    r3 = (torch.einsum('tn,td->tnd', hp3.float().cpu(), ho3.float().cpu())
          + torch.einsum('tjn,tjd->tnd', hr3.float().cpu(), x3.float().cpu()))
    d3 = (y3.float().cpu() - r3).abs().max().item()
    print(f"[TND ] {tuple(y3.shape)} vs_ref={d3:.4f} -> {'PASS' if d3 < 0.05 else 'FAIL'}")

    if args.bench:
        for S in (1024, 2048):
            xb = torch.randn(1, S, N, D, dtype=torch.bfloat16).npu()
            hb = torch.randn(1, S, D, dtype=torch.bfloat16).npu()
            pb = torch.randn(1, S, N, dtype=torch.float32).npu()
            cb = torch.randn(1, S, N, N, dtype=torch.float32).npu()
            fn = lambda: mhc_post(xb, cb, hb, pb)
            if args.msprof:
                msprof_bench(fn, f"cann mhc_post S={S}", f"./prof_msprof/S{S}")
            else:
                for _ in range(3):
                    fn()
                torch_npu.npu.synchronize()
                s = torch.npu.Event(True)
                e = torch.npu.Event(True)
                s.record()
                iters = 100 if S == 1024 else 50
                for _ in range(iters):
                    fn()
                e.record()
                torch_npu.npu.synchronize()
                print(f"  [bench S={S}] {s.elapsed_time(e) / iters:.4f} ms/iter")
    print("== done ==")


if __name__ == "__main__":
    main()
