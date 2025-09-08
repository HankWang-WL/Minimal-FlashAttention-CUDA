# --- add project root to sys.path so sibling packages are importable ---
import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ----------------------------------------------------------------------

# Defaults: 快速、不刷大量 log
os.environ.setdefault("FLASH_BLOCK_N", "64")   # safe tile width
os.environ.setdefault("FLASH_SMEM_KB", "48")   # per-block shared mem budget
os.environ.setdefault("FLASH_DEBUG", "0")      # 0 = 不列印每次啟動

import argparse
from pathlib import Path
import csv
import time

import torch
import pandas as pd
import matplotlib.pyplot as plt

from baselines.sdpa_pytorch import sdpa_pytorch

# Import your custom CUDA FlashAttention (must succeed)
try:
    from cuda.launcher import custom_sdpa_flash
    HAS_CUSTOM = True
    CUSTOM_IMPORT_ERR = ""
except Exception as e:
    HAS_CUSTOM = False
    CUSTOM_IMPORT_ERR = repr(e)

@torch.no_grad()
def measure_latency_and_memory(fn, *tensors, iters=50, warmup=10, synchronize=True, **kwargs):
    device = tensors[0].device
    # warmup
    for _ in range(warmup):
        _ = fn(*tensors, **kwargs)
    if synchronize and device.type == "cuda":
        torch.cuda.synchronize()
    # measure
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _ = fn(*tensors, **kwargs)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        avg_ms = total_ms / iters
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        t0 = time.time()
        for _ in range(iters):
            _ = fn(*tensors, **kwargs)
        total_ms = (time.time() - t0) * 1000.0
        avg_ms = total_ms / iters
        peak_mb = 0.0
    return avg_ms, peak_mb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--dhead", type=int, default=64)
    # 預設：輕量 L 範圍 & 次數（跑很快）
    ap.add_argument("--L_min", type=int, default=512)
    ap.add_argument("--L_max", type=int, default=2048)
    ap.add_argument("--step", type=int, default=512)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--full", action="store_true", help="跑完整配置：L_max=4096, iters=50, warmup=10")
    args = ap.parse_args()

    # 如果想要完整重跑，再加 --full
    if args.full:
        if args.L_max < 4096: args.L_max = 4096
        if args.iters < 50:   args.iters = 50
        if args.warmup < 10:  args.warmup = 10
        # 也可以打開 debug 看 launcher 配置
        os.environ["FLASH_DEBUG"] = os.environ.get("FLASH_DEBUG", "0")

    if not HAS_CUSTOM:
        raise RuntimeError(
            "Failed to import custom CUDA kernel (cuda.launcher). "
            f"Error: {CUSTOM_IMPORT_ERR}\n"
            "Tip: install CuPy matching your Torch CUDA (11.x -> cupy-cuda11x, 12.x -> cupy-cuda12x)"
        )

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    rows = []
    L_values = list(range(args.L_min, args.L_max + 1, args.step))



    pyt_backends = ["flash", "math"]



    for L in L_values:
        q = torch.randn(args.B, args.H, L, args.dhead, device=device, dtype=dtype)
        k = torch.randn(args.B, args.H, L, args.dhead, device=device, dtype=dtype)
        v = torch.randn(args.B, args.H, L, args.dhead, device=device, dtype=dtype)

        # ---- First: run your custom CUDA (force FP32 & B=1,H=1) ----
        q32, k32, v32 = q.float(), k.float(), v.float()
        if q32.shape[0] > 1 or q32.shape[1] > 1:
            q32, k32, v32 = q32[:1, :1], k32[:1, :1], v32[:1, :1]

        try:
            avg_ms, peak_mb = measure_latency_and_memory(
                custom_sdpa_flash, q32, k32, v32,
                iters=args.iters, warmup=args.warmup, is_causal=args.causal
            )
            rows.append({"L": L, "backend": "custom_cuda", "avg_ms": avg_ms, "peak_mem_MB": peak_mb})
            print(f"[Custom]        L={L:4d} avg={avg_ms:.3f} ms  peak={peak_mb:.1f} MB")
        except Exception as e:
            raise RuntimeError(f"Custom kernel failed at L={L}: {e}")

        # ---- Then: PyTorch flash baseline ----
        for backend in pyt_backends:
            try:
                avg_ms, peak_mb = measure_latency_and_memory(
                    sdpa_pytorch, q, k, v,
                    iters=args.iters, warmup=args.warmup, backend=backend, is_causal=args.causal
                )
                rows.append({"L": L, "backend": backend, "avg_ms": avg_ms, "peak_mem_MB": peak_mb})
                print(f"[PyTorch-{backend}] L={L:4d} avg={avg_ms:.3f} ms  peak={peak_mb:.1f} MB")
            except Exception as e:
                print(f"[PyTorch-{backend}] L={L:4d} FAILED: {e}")

    # Save CSV
    csv_path = Path(args.outdir)/f"sdpa_Lsweep_B{args.B}_H{args.H}_d{args.dhead}_{'fp16' if dtype==torch.float16 else 'fp32'}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["L", "backend", "avg_ms", "peak_mem_MB"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {csv_path}")

    # Plot
    df = pd.DataFrame(rows)

    # Latency
    plt.figure(figsize=(7,5))
    for name, sub in df.groupby("backend"):
        sub = sub.sort_values("L")
        plt.plot(sub["L"], sub["avg_ms"], marker="o", label=name)
    plt.xlabel("Sequence length L"); plt.ylabel("Avg latency (ms)"); plt.title("SDPA Latency vs L (custom vs flash)")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(args.outdir)/"latency_vs_L.png", dpi=150)

    # Memory
    plt.figure(figsize=(7,5))
    for name, sub in df.groupby("backend"):
        sub = sub.sort_values("L")
        plt.plot(sub["L"], sub["peak_mem_MB"], marker="o", label=name)
    plt.xlabel("Sequence length L"); plt.ylabel("Peak GPU memory (MB)"); plt.title("SDPA Peak Memory vs L (custom vs flash)")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(args.outdir)/"memory_vs_L.png", dpi=150)

    print(f"Saved plots to: {args.outdir}/latency_vs_L.png and memory_vs_L.png")
    plt.show()

if __name__ == "__main__":
    main()
