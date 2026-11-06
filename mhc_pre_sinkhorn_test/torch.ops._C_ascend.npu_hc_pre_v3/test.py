# coding=utf-8
"""_C_ascend.npu_hc_pre_v3 (vllm-ascend 自定义 HcPre 内核) 精度/语义/性能测试

被测算子: torch.ops._C_ascend.npu_hc_pre_v3 (DSV4.1 生产实际调用, deepseek_v41/model.py:859)
参考对照: 与 CANN cann_ops_transformer.mhc_pre_sinkhorn 双来源一致性 (相同输入下输出对照)

运行: python3 test.py [--bench]
"""

import argparse
import csv
import os
import time

import torch
import torch_npu


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--msprof", action="store_true",
                    help="msprof 精准测 kernel 时间+利用率 (自动 warmup)")
    args = ap.parse_args()
    torch_npu.npu.set_device(0)
    import sys
    sys.path.insert(0, "/vllm-workspace/vllm-ascend")
    from vllm_ascend.utils import enable_custom_op
    from cann_ops_transformer.ops import mhc_pre_sinkhorn
    enable_custom_op()

    torch.manual_seed(0)
    B, S, N, C = 1, 4, 4, 1280
    phi = torch.randn(N * N + 2 * N, N * C, dtype=torch.float32).npu()
    alpha = torch.randn(3, dtype=torch.float32).npu()
    bias = torch.randn(N * N + 2 * N, dtype=torch.float32).npu()
    x = torch.randn(B, S, N, C, dtype=torch.bfloat16).npu()
    pre_mix = torch.randn(B, S, N, dtype=torch.float32).npu()

    hin_c, hpost_c, hres_c = mhc_pre_sinkhorn(x, phi, alpha, bias, N, 20)
    y_v, post_v, comb_v, pre_v = torch.ops._C_ascend.npu_hc_pre_v3(
        x, phi, alpha, bias, pre_mix, hc_mult=N, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6)
    torch_npu.npu.synchronize()

    d_post = (post_v.float().cpu() - hpost_c.float().cpu()).abs().max().item()
    d_comb = (comb_v.float().cpu().reshape(B, S, -1) - hres_c.float().cpu()).abs().max().item()
    # comb_frag Sinkhorn 性质: 列和=1, 行和≈1
    cfm = comb_v.float().cpu().reshape(B, S, N, N)
    col_max = (cfm.sum(-2) - 1).abs().max().item()
    row_max = (cfm.sum(-1) - 1).abs().max().item()
    # y 语义: Σ pre·x (与 CANN hin 不同, 记录差异)
    y_range = (y_v.float().cpu().min().item(), y_v.float().cpu().max().item())
    print(f"[prec] post_diff={d_post:.2e} comb_diff={d_comb:.2e} "
          f"|col-1|={col_max:.2e} |row-1|={row_max:.2e}")
    print(f"  y(vllm) range={y_range} vs hin(cann) range=({hin_c.float().cpu().min().item():.2f}, "
          f"{hin_c.float().cpu().max().item():.2f}) — y=Σ(pre·x) 语义不同")
    ok = d_post < 5e-4 and d_comb < 5e-4 and col_max < 1e-3 and row_max < 8e-2
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    if args.bench:
        for S, C in ((1024, 1280), (2048, 1280), (2048, 2560)):
            xb = torch.randn(1, S, N, C, dtype=torch.bfloat16).npu()
            phi_b = torch.randn(24, N * C, dtype=torch.float32).npu()
            pm = torch.randn(1, S, N, dtype=torch.float32).npu()
            fn = lambda: torch.ops._C_ascend.npu_hc_pre_v3(
                xb, phi_b, alpha, bias, pm, hc_mult=N, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6)
            if args.msprof:
                msprof_bench(fn, f"vllm npu_hc_pre_v3 S={S} C={C}", f"./prof_msprof/S{S}_C{C}")
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

    print("== done ==")


if __name__ == "__main__":
    main()
