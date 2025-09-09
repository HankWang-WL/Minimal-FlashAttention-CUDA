# cuda/launcher.py — minimal build-from-.cu launcher
import os
from pathlib import Path
import cupy as cp
import torch

_kernel = None
_printed_cfg = False  # print launch config at most once

def _collect_include_paths():
    """Collect CUDA include dirs for NVRTC (<math.h> etc.)."""
    incl = []
    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        p = Path(cuda_path) / "include"
        if p.exists():
            incl.append(str(p))
    for v in ["v12.4", "v12.3", "v12.2", "v12.1", "v12.0", "v11.8", "v11.7", "v11.6", "v11.5", "v11.4", "v11.3", "v11.2"]:
        p = Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{v}/include")
        if p.exists():
            incl.append(str(p))
    for p in [Path("/usr/local/cuda/include"), Path("/opt/cuda/include")]:
        if p.exists():
            incl.append(str(p))
    # dedup
    seen, uniq = set(), []
    for d in incl:
        if d not in seen:
            uniq.append(d); seen.add(d)
    return uniq

def _get_nvrtc_options():
    """Compose NVRTC options; ignore duplicate -arch if user sets CUPY_NVRTC_COMPILER_OPTIONS."""
    opts = ["--std=c++14", "--use_fast_math"]
    for inc in _collect_include_paths():
        opts.append(f"-I{inc}")
    extra = os.getenv("CUPY_NVRTC_COMPILER_OPTIONS", "")
    if extra:
        import shlex
        for tok in shlex.split(extra):
            low = tok.lower()
            if low.startswith("-arch") or "gpu-architecture" in low:
                continue
            opts.append(tok)
    return tuple(opts)

def _load_kernel():
    """Load and cache CUDA kernel from external .cu using CuPy RawModule."""
    global _kernel
    if _kernel is not None:
        return _kernel

    cu_name = os.getenv("FLASH_KERNEL_SRC", "sdpa_tiled16.cu")
    fn_name = os.getenv("FLASH_KERNEL_FN", "sdpa_tiled16")
    path = Path(__file__).with_name(cu_name)
    if not path.exists():
        raise FileNotFoundError(f"CUDA source not found: {path}")

    src = path.read_text(encoding="utf-8")
    try:
        module = cp.RawModule(code=src, options=_get_nvrtc_options(), name_expressions=(fn_name,))
        fn = module.get_function(fn_name)
    except cp.cuda.compiler.CompileException as e:
        # 簡潔地把 NVRTC 的訊息回傳，避免冗長提示
        raise RuntimeError(f"NVRTC compile failed for {cu_name}/{fn_name}.") from e

    _kernel = fn
    return fn

def _pick_block_n(D: int, limit_bytes: int):
    """Choose BLOCK_N so (K + V + scores0 + scores1) fits under limit."""
    pad = 1
    BLOCK_D = min(D, 64)
    candidates = []
    if os.getenv("FLASH_BLOCK_N"):
        try:
            candidates.append(int(os.getenv("FLASH_BLOCK_N")))
        except:
            pass
    # Pascal 常見較佳：32；其後嘗試較大
    for n in (32, 64, 80, 96, 128):
        if n not in candidates:
            candidates.append(n)
    for n in candidates:
        bytes_needed = (n*(BLOCK_D+pad) + n*(BLOCK_D+pad) + 2*n) * 4  # K + V + scores0 + scores1
        if bytes_needed <= limit_bytes:
            return n, BLOCK_D, pad
    return 64, BLOCK_D, pad

@torch.no_grad()
def custom_sdpa_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None, is_causal=True):
    """Flash-style SDPA launcher (MVP: B=1, H=1, FP32, D<=64)."""
    # 輕量檢查（保留最基本的假設）
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.float32 and k.dtype == torch.float32 and v.dtype == torch.float32
    B, H, L, D = q.shape
    assert B == 1 and H == 1

    out = torch.empty_like(q)
    q_cu = cp.asarray(q.contiguous()); k_cu = cp.asarray(k.contiguous())
    v_cu = cp.asarray(v.contiguous()); out_cu = cp.asarray(out)

    limit_kb = int(os.getenv("FLASH_SMEM_KB", "48"))  # Pascal SM ~48KB
    limit_bytes = limit_kb * 1024
    BLOCK_M = int(os.getenv("FLASH_BLOCK_M", "2"))
    BLOCK_N, BLOCK_D, pad = _pick_block_n(D, limit_bytes)

    # 內部 kernel 自己掃 K/V tiles → grid.x 必須為 1
    grid = (1, (L + BLOCK_M - 1) // BLOCK_M, 1)
    block = (32, 2, 1)  # 64 threads
    shared_mem_bytes = (BLOCK_N*(BLOCK_D+pad) + BLOCK_N*(BLOCK_D+pad) + 2*BLOCK_N) * 4

    global _printed_cfg
    if os.getenv("FLASH_DEBUG") == "1" and not _printed_cfg:
        print(f"[launcher] FN={os.getenv('FLASH_KERNEL_FN','sdpa_tiled16')} "
              f"SRC={os.getenv('FLASH_KERNEL_SRC','sdpa_tiled16.cu')}")
        print(f"[launcher] BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_D={BLOCK_D} pad={pad} "
              f"smem={shared_mem_bytes/1024:.1f}KB grid=({grid[0]},{grid[1]})")
        _printed_cfg = True

    kernel = _load_kernel()
    args = (q_cu, k_cu, v_cu, out_cu,
            cp.int32(L), cp.int32(D),
            cp.int32(BLOCK_M), cp.int32(BLOCK_N), cp.int32(BLOCK_D),
            cp.int32(1 if is_causal else 0))
    kernel(grid, block, args, shared_mem=shared_mem_bytes)
    cp.cuda.runtime.deviceSynchronize()
    return out
