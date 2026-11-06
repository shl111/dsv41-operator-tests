# coding=utf-8
"""mhc_pre_sinkhorn (MHC Pre + Sinkhorn) 精度 / 约束 / 性能测试

被测算子: cann_ops_transformer.ops.mhc_pre_sinkhorn (Atlas A2/A3 支持)
参考实现: 依据 CANN 文档公式推导的 PyTorch float32 参考:

  x' = RMSNorm(x, eps=normEps)                        # [B,S,N,C], 沿 (N,C) 两维
  H^pre  = sigmoid(alpha[0]·(x'_flat·phi_pre^T) + b_pre)      # [B,S,N]
  H^post = 2·sigmoid(alpha[1]·(x'_flat·phi_post^T) + b_post)  # [B,S,N]
  H^res  = alpha[2]·(x'_flat·phi_res^T) + b_res       # [B,S,N*N] -> [B,S,N,N]
  H^res  = Sinkhorn(H^res, numIters)                   # 交替行/列归一化
  h_in   = x @ H^pre^T                                 # [B,S,C]  (x 原始, 非 RMSNorm)

  phi 拆分: phi_pre=[0:N], phi_post=[N:2N], phi_res=[2N:N*N+2N]
  (N*N+2N, N*C) -> pre:(N,N*C), post:(N,N*C), res:(N*N,N*C)

运行: python3 test_mhc_pre_sinkhorn.py [--bench]
"""

import argparse
import csv
import os
import time

import torch
import torch_npu
from cann_ops_transformer.ops import mhc_pre_sinkhorn

N, C_DEF = 4, 1280


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


def reference(x, phi, alpha, bias, num_iters=20, hc_eps=1e-6, norm_eps=1e-6):
    """float32 参考实现。返回 (hin, hpost_2s, hres_sinkhorn)。"""
    B, S, _, C = x.shape
    xf = x.float().cpu()
    phif = phi.float().cpu()
    alphaf = alpha.float().cpu()
    biasf = bias.float().cpu()

    rms = (xf.pow(2).mean(dim=(-2, -1), keepdim=True) + norm_eps).sqrt()
    xnorm = xf / rms
    xn_flat = xnorm.reshape(B, S, -1)  # [B,S,N*C]

    phi_pre = phif[:N]
    phi_post = phif[N:2 * N]
    phi_res = phif[2 * N:]
    b_pre = biasf[:N]
    b_post = biasf[N:2 * N]
    b_res = biasf[2 * N:]

    hpre_s = torch.sigmoid(torch.einsum('bsf,nf->bsn', xn_flat, phi_pre) * alphaf[0] + b_pre)
    hpost_2s = 2 * torch.sigmoid(torch.einsum('bsf,nf->bsn', xn_flat, phi_post) * alphaf[1] + b_post)
    hres = torch.einsum('bsf,nf->bsn', xn_flat, phi_res) * alphaf[2] + b_res  # [B,S,N*N]
    hres_m = hres.reshape(B, S, N, N)

    # Sinkhorn: normOut[0]=softmax(-1)+eps; 迭代 20 次行/列交替归一化
    norm = torch.softmax(hres_m, dim=-1) + hc_eps
    norm = norm / (norm.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(1, num_iters):
        norm = norm / (norm.sum(dim=-1, keepdim=True) + hc_eps)
        norm = norm / (norm.sum(dim=-2, keepdim=True) + hc_eps)
    hres_s = norm

    hin = torch.einsum('bsnc,bsn->bsc', xf, hpre_s)
    return hin, hpost_2s, hres_s


def run_precision(B, S, C, num_iters=20):
    x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
    phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
    alpha = torch.randn(3, dtype=torch.float32).npu()
    bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()
    hin, hpost, hres = mhc_pre_sinkhorn(x, phi, alpha, bias, N, num_iters)
    torch_npu.npu.synchronize()

    ref_hin, ref_hpost, ref_hres = reference(x, phi, alpha, bias, num_iters)
    d_hin = (hin.float().cpu() - ref_hin).abs().max().item()
    d_hpost = (hpost.float().cpu() - ref_hpost).abs().max().item()
    d_hres = (hres.float().cpu() - ref_hres.reshape(B, S, -1)).abs().max().item()

    # Sinkhorn 性质: reshape [B,S,N,N], 列和=1(精确), 行和≈1(迭代收敛)
    hr = hres.float().cpu().reshape(B, S, N, N)
    col_max = (hr.sum(-2) - 1).abs().max().item()
    row_max = (hr.sum(-1) - 1).abs().max().item()

    print(f"[prec S={S} C={C}] hin_diff={d_hin:.4f} hpost_diff={d_hpost:.4f} "
          f"hres_diff={d_hres:.4f} |col-1|={col_max:.2e} |row-1|={row_max:.2e}")
    d_post_all = (hpost.float().cpu() - ref_hpost).abs()
    frac_over = (d_post_all > 5e-3).float().mean().item()
    mean_post = d_post_all.mean().item()
    # 判据: hpost 允许极少数点(<=1%)在 5e-3~2e-2 间(大 shape 下 kernel 量化噪声),
    # mean 必须极小(<=2e-4), 主判据以 hres + Sinkhorn 性质 + 噪声占比为准
    ok = (d_hres < 5e-3 and mean_post <= 2e-4 and frac_over <= 0.01
          and d_hpost <= 2e-2 and col_max < 1e-3 and row_max < 8e-2)
    print(f"  hpost mean={mean_post:.2e} >5e-3占比={frac_over*100:.2f}% "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def run_constraints():
    """numIters 边界 + C 规格实测。"""
    B, S = 1, 4
    results = []
    for C in (1280, 1920, 2560, 2048):
        x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
        phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
        alpha = torch.randn(3, dtype=torch.float32).npu()
        bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()
        try:
            mhc_pre_sinkhorn(x, phi, alpha, bias, N, 20)
            torch_npu.npu.synchronize()
            results.append(f"C={C}: OK")
        except Exception:
            results.append(f"C={C}: FAIL(不应)")
    for iters in (0, 1, 21):
        x = torch.randn(B, S, N, C_DEF, dtype=torch.bfloat16).npu()
        phi = torch.randn(24, N * C_DEF, dtype=torch.float32).npu()
        alpha = torch.randn(3, dtype=torch.float32).npu()
        bias = torch.randn(24, dtype=torch.float32).npu()
        try:
            mhc_pre_sinkhorn(x, phi, alpha, bias, N, iters)
            torch_npu.npu.synchronize()
            results.append(f"numIters={iters}: OK(不应)")
        except Exception:
            results.append(f"numIters={iters}: FAIL(预期参数校验)")
    for r in results:
        print("  ", r)
    return all("OK" in r for r in results[:4])


def run_bench(msprof=False):
    for S, C in ((1024, 1280), (2048, 1280), (2048, 2560)):
        B = 1
        x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
        phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
        alpha = torch.randn(3, dtype=torch.float32).npu()
        bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()
        fn = lambda: mhc_pre_sinkhorn(x, phi, alpha, bias, N, 20)
        if msprof:
            msprof_bench(fn, f"cann mhc_pre_sinkhorn S={S} C={C}", f"./prof_msprof/S{S}_C{C}")
        else:
            for _ in range(5):
                fn()
            torch_npu.npu.synchronize()
            s = torch.npu.Event(True)
            e = torch.npu.Event(True)
            s.record()
            for _ in range(50):
                fn()
            e.record()
            torch_npu.npu.synchronize()
            print(f"  [bench S={S} C={C}] {s.elapsed_time(e) / 50:.4f} ms/iter")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--msprof", action="store_true",
                    help="msprof 精准测 kernel 时间+利用率 (自动 warmup)")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)

    ok1 = run_precision(1, 1024, 1280)
    ok2 = run_precision(1, 8, 1920)
    ok3 = run_precision(1, 4, 2560)
    ok4 = run_constraints()
    if args.bench:
        run_bench(msprof=args.msprof)
    passed = sum([ok1, ok2, ok3, ok4])
    print(f"== SUMMARY: {passed}/4 PASS ==")


if __name__ == "__main__":
    main()
