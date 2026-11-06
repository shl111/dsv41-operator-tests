# coding=utf-8
"""mhc_pre_sinkhorn_backward (反向) 精度 / 梯度路径 / 性能测试

被测算子: cann_ops_transformer.ops.mhc_pre_sinkhorn_backward
（CANN mhc_pre_sinkhorn 融合前向的反向算子，Autograd 自动调用）

梯度路径验证: 三个输出 (hin/hpost/hres) 各自独立的参数梯度分支——
  hin  只依赖 alpha[0] (pre 路径)
  hpost 只依赖 alpha[1] (post 路径)
  hres 只依赖 alpha[2] (res 路径 + Sinkhorn)

运行: python3 test_mhc_pre_sinkhorn_backward.py [--bench]
"""

import argparse
import csv
import os
import time

import torch
import torch_npu
from cann_ops_transformer.ops import mhc_pre_sinkhorn

N = 4


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


def grad_alpha_for(loss_sel, seed):
    torch.manual_seed(seed)
    B, S, C = 1, 4, 1280
    x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu().requires_grad_()
    phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
    alpha = torch.randn(3, dtype=torch.float32).npu().requires_grad_()
    bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()
    hin, hpost, hres = mhc_pre_sinkhorn(x, phi, alpha, bias, N, 20)
    if loss_sel == "hin":
        loss = hin.float().sum()
    elif loss_sel == "hpost":
        loss = hpost.sum()
    else:
        loss = hres.sum()
    loss.backward()
    torch_npu.npu.synchronize()
    return alpha.grad.cpu().tolist()


def run_grad_paths():
    """三个输出路径的 alpha 梯度独立性。"""
    g_hin = grad_alpha_for("hin", 0)
    g_hpost = grad_alpha_for("hpost", 0)
    g_hres = grad_alpha_for("hres", 0)
    print(f"[grad paths] d(hin)/da   ={g_hin}")
    print(f"            d(hpost)/da ={g_hpost}")
    print(f"            d(hres)/da  ={g_hres}")
    ok_hin = abs(g_hin[0]) > 1 and abs(g_hin[1]) < 1e-4 and abs(g_hin[2]) < 1e-4
    ok_hpost = abs(g_hpost[1]) > 0.1 and abs(g_hpost[0]) < 1e-4 and abs(g_hpost[2]) < 1e-4
    ok_hres = abs(g_hres[0]) < 1e-4 and abs(g_hres[1]) < 1e-4
    print(f"  hin->alpha[0] 独立: {ok_hin}")
    print(f"  hpost->alpha[1] 独立: {ok_hpost}")
    print(f"  hres->alpha[2] 独立: {ok_hres} (alpha[2] 影响 Sinkhorn 输入)")
    return ok_hin and ok_hpost and ok_hres


def run_grad_finite(seed=1):
    """随机上游梯度下 4 个梯度均有限非 NaN。"""
    torch.manual_seed(seed)
    B, S, C = 1, 4, 1280
    x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu().requires_grad_()
    phi = torch.randn(24, N * C, dtype=torch.float32).npu().requires_grad_()
    alpha = torch.randn(3, dtype=torch.float32).npu().requires_grad_()
    bias = torch.randn(24, dtype=torch.float32).npu().requires_grad_()
    hin, hpost, hres = mhc_pre_sinkhorn(x, phi, alpha, bias, N, 20)
    loss = (hin.float() ** 2).sum() + (hpost ** 2).sum() + (hres ** 2).sum()
    loss.backward()
    torch_npu.npu.synchronize()
    checks = {
        "x.grad": torch.isfinite(x.grad).all().item(),
        "phi.grad": torch.isfinite(phi.grad).all().item(),
        "alpha.grad": torch.isfinite(alpha.grad).all().item(),
        "bias.grad": torch.isfinite(bias.grad).all().item(),
    }
    for k, v in checks.items():
        print(f"[finite] {k}: {v}")
    return all(checks.values())


def run_bench(msprof=False):
    for S, C in ((1024, 1280), (2048, 1280)):
        torch.manual_seed(0)
        x = torch.randn(1, S, N, C, dtype=torch.bfloat16).npu().requires_grad_()
        phi = torch.randn(24, N * C, dtype=torch.float32).npu().requires_grad_()
        alpha = torch.randn(3, dtype=torch.float32).npu().requires_grad_()
        bias = torch.randn(24, dtype=torch.float32).npu().requires_grad_()
        hin, hpost, hres = mhc_pre_sinkhorn(x, phi, alpha, bias, N, 20)
        loss = (hin.float() ** 2).sum() + (hpost ** 2).sum() + (hres ** 2).sum()
        if msprof:
            fn = lambda: (loss.backward(retain_graph=True), setattr(x, 'grad', None))
            msprof_bench(fn, f"cann mhc_pre_sinkhorn_backward S={S} C={C}", f"./prof_msprof/S{S}_C{C}")
        else:
            for _ in range(3):
                loss.backward(retain_graph=True)
                x.grad = None
            torch_npu.npu.synchronize()
            s = torch.npu.Event(True)
            e = torch.npu.Event(True)
            s.record()
            for _ in range(20):
                loss.backward(retain_graph=True)
                x.grad = None
            e.record()
            torch_npu.npu.synchronize()
            print(f"  [bench S={S} C={C}] {s.elapsed_time(e) / 20:.4f} ms/iter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--msprof", action="store_true",
                    help="msprof 精准测 kernel 时间+利用率 (自动 warmup)")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    ok1 = run_grad_paths()
    ok2 = run_grad_finite()
    if args.bench:
        run_bench(msprof=args.msprof)
    print(f"== SUMMARY: {sum([ok1, ok2])}/2 PASS ==")


if __name__ == "__main__":
    main()
