# FlashSDPA‑Mini: A Minimal FlashAttention (SDPA) Kernel

This repository contains a from‑scratch CUDA implementation of **tile‑wise, online‑softmax** scaled dot‑product attention. The focus is on clarity and memory efficiency:

* Avoid materializing the full L×L score matrix.
* Stage K/V tiles in shared memory with bank‑conflict padding.
* Warp‑collaborative dot product, apply online softmax, and accumulate weighted V in registers.
* Write the output tensor once at the end.

---

## 📚 What is FlashAttention?

Standard attention computes `softmax(QKᵀ / sqrt(d)) V`. With sequence length **L**, the intermediate score matrix `QKᵀ` has size **L×L** and costs **O(L²)** memory.

**FlashAttention** changes the computation order:

* **Tile over K/V**: split the sequence into tiles, load one tile into shared memory at a time.
* **Online softmax**: maintain running `(m, l)` (max and normalized sum) per query row to merge results tile by tile without storing all scores.
* **Accumulate immediately**: apply weights to V on the fly, keeping partial sums in registers.

Memory scales like **O(L·d + tile)** instead of **O(L²)**, while producing the same outputs.

---

## ✨ What’s inside

* **Custom CUDA kernel**

  * K/V staged per tile in shared memory with +1 padding
  * Warp‑level dot products with shuffle reductions
  * Online softmax and causal masking
  * Register accumulation, single write of output
* **Baselines**

  * PyTorch `scaled_dot_product_attention` (flash / math)
* **Benchmark and plots**

  * Sweep L, record latency and memory
  * Plots in `results/latency_vs_L.png` and `results/memory_vs_L.png`

---

## 📦 Project layout

```
.
├─ baselines/
│  └─ sdpa_pytorch.py         # PyTorch SDPA wrapper (flash / math)
├─ bench/
│  └─ bench_l_sweep.py        # L-sweep benchmark → CSV + plots
├─ cuda/
│  ├─ launcher.py             # CuPy NVRTC loader + kernel launcher
│  └─ sdpa_tiled16.cu         # CUDA Flash-style SDPA kernel
├─ results/                   # Auto-generated CSV/plots
├─ README.md
└─ requirements.txt
```

---

## 🚀 Quick start

**Prerequisites**

* NVIDIA GPU + driver
* CUDA Toolkit (for NVRTC headers)
* Python 3.10+ with CUDA build of `torch` and matching `cupy-cudaXX`

**Install**

```bash
pip install -r requirements.txt
```

**Run (fast preset)**

```bash
python -m bench.bench_l_sweep --device cuda
```

Produces:

* `results/sdpa_Lsweep_B1_H1_d64_fp32.csv`
* `results/latency_vs_L.png`
* `results/memory_vs_L.png`

**Run (full sweep)**

```bash
python -m bench.bench_l_sweep --device cuda --full
```

> Current assumptions: `B=H=1`, `D<=64`, FP32.

---

## 🧩 Kernel design

* Tiling over K/V with shared memory (+1 stride padding)
* Warp‑collaborative dot products with shuffles
* Online softmax with running `(m,l)`
* Causal mask inline
* Register accumulation, one final store of `Out` (`[L, D]`)

---

## ⚙️ Optimization plan on GTX1060 (plain, detailed)

> Constraints: Pascal (GTX1060) has no `cp.async`; reduced precision (FP16/BF16) is not used here. Expect **no gain** from shared‑mem double‑buffering without async copies.

### O1 — Practical wins you should do now

1. **Float4 vectorized loads/stores for K/V**
   **What**: read/write 4 floats at a time (`float4`).
   **Why**: fewer memory transactions → higher effective bandwidth; better L2/TEX utilization.
   **How**:

   * Preconditions: `d_head % 4 == 0` (true for 64) and pointers are 16‑byte aligned.
   * Load: `const float4* K4 = reinterpret_cast<const float4*>(K + base);` then index by `d/4`.
   * Option A: keep `float4` in registers and multiply per lane; Option B: unpack to 4 scalars after load.
     **Gotchas**: handle tail only if `D` not divisible by 4 (not needed for 64).

2. **Per‑lane register caching of Q channels (`d0`, `d1`)**
   **What**: each lane loads its two Q elements **once** and reuses across all `t` in the tile.
   **Why**: avoids reloading `Qrow[d0]`/`Qrow[d1]` for every K row; reduces L1 pressure.
   **How**: hoist outside the inner `t` loop:

   ```cpp
   const int d0 = lane, d1 = lane + 32;
   const float q0 = (d0 < D) ? Qrow[d0] : 0.f;
   const float q1 = (d1 < D) ? Qrow[d1] : 0.f;
   // use q0/q1 inside all dot products
   ```

3. **Keep `grid.x = 1`; tune `BLOCK_N` for shared‑mem limit/occupancy**
   **What**: the kernel already loops over K/V tiles; multiple blocks in `x` would duplicate work.
   **Why**: simpler scheduling; avoids redundant tile loads.
   **How (BLOCK\_N)**: on Pascal 48KB SMEM/block is common. With padding, bytes per tile ≈
   `bytes = 4 * [ BLOCK_N*(BLOCK_D+1)   // K                + BLOCK_N*(BLOCK_D+1) // V                + 2*BLOCK_N ]         // scores0/1`
   For `D=64` ⇒ `BLOCK_D=64`, a good default is **BLOCK\_N=32** (often higher occupancy than 64).

4. **Single final store of `Out`**
   **What**: keep accumulators (`acc0`, `acc1`) in registers; rescale when `(m,l)` updates; write once at the end.
   **Why**: removes per‑tile global writes; cuts DRAM traffic.
   **How**: this is already how the code works—keep it that way when refactoring.

5. **Coalesced & aligned accesses**
   **What**: ensure `Q/K/V/Out` are contiguous and aligned.
   **Why**: to let `float4` vectorization actually coalesce.
   **How**: keep tensors contiguous on host (`contiguous()`), align base pointers (16B), and index rows as `[row*D + d]`.

6. **(Optional) Register blocking (2×2)**
   **What**: have each lane produce >1 output channel beyond `d0,d1` (e.g., also `d0+64`, `d1+64` if you split D).
   **Why**: increase math per byte if registers allow.
   **How**: add extra `acc*` registers and update them in the same loop body; watch register pressure.

### O2 — Next gains to try (Pascal‑friendly)

1. **Keep short‑tile scores in registers**
   **What**: when `tile_len ≤ 32`, store per‑lane scores in registers instead of shared memory.
   **Why**: removes shared‑mem traffic and syncs for score reads.
   **How**: use a small per‑lane array (e.g., `float s_local[32];`) and warp‑reduce to get `tile_max`/`sum`. Ensure no register spill.

2. **Unroll inner loops and leverage FMA**
   **What**: unroll the `d` loop (e.g., ×4/×8) so the compiler fuses multiply‑add.
   **Why**: better ILP and scheduling on Pascal cores.
   **How**: `#pragma unroll` with care; check SASS and profiler for spills.

3. **Pointer alignment & strict coalescing**
   **What**: make sure K/V row starts are 16B‑aligned; avoid strided or misaligned segments.
   **Why**: sustains throughput after switching to `float4`.

4. **Register pressure control**
   **What**: prevent spills when adding vectorization/unrolling.
   **How**: trim temporaries; consider `__launch_bounds__(64)`; verify occupancy vs. registers in Nsight Compute.

5. **Split‑D for `D>64`**
   **What**: process D in 64‑wide chunks while maintaining the same `(m,l)` per row.
   **Why**: generalizes beyond 64‑dim heads without changing the core loop.

6. **Bank‑conflict audit after vectorization**
   **What**: confirm `+1` padding still eliminates conflicts with your new access pattern.
   **How**: verify stall reasons in Nsight Compute (shared‑mem bank conflict counters).


---

## 🧪 Headline results (GTX1060, FP32, d=64, B=H=1)

| L (seq len) | **Flash (PyTorch)** | **Math (PyTorch)** | **Custom CUDA** |
| :---------: | :-----------------: | :----------------: | :-------------: |
|     512     |      \~0.10 ms      |   \~0.22–0.28 ms   |     \~3–4 ms    |
|     1024    |      \~0.20 ms      |      \~0.54 ms     |    \~11–12 ms   |
|     1536    |      \~0.30 ms      |      \~1.1 ms      |    \~23–25 ms   |
|     2048    |    \~0.43–0.46 ms   |    \~1.9–2.0 ms    |    \~36–38 ms   |

**Observation:** the custom kernel already matches FlashAttention’s memory curve (near‑linear in L), though latency is still above PyTorch/cuBLAS.

<p align="center">
  <img src="results/latency_vs_L.png" width="48%" alt="Latency vs L"/>
  <img src="results/memory_vs_L.png" width="48%" alt="Memory vs L"/>
</p>

CSV: `results/sdpa_Lsweep_B1_H1_d64_fp32.csv`

---

## ⚙️ Tuning & environment variables

```powershell
# Print launch config once
$env:FLASH_DEBUG = "1"

# Shared memory budget (KB) per block (Pascal SM ~48KB)
$env:FLASH_SMEM_KB = "48"

# Tile width (try 32 on GTX1060)
$env:FLASH_BLOCK_N = "32"

# Choose kernel source/function
$env:FLASH_KERNEL_SRC = "sdpa_tiled16.cu"
$env:FLASH_KERNEL_FN  = "sdpa_tiled16"
```

---

## 🛠️ Troubleshooting

* **CuPy multiple packages**: uninstall all, install the one matching your Torch CUDA
* **NVRTC cannot open `math.h`**: check `CUDA_PATH`
* **Duplicate `-arch` flag**: remove from `CUPY_NVRTC_COMPILER_OPTIONS`
* **Illegal address / shared‑mem overflow**: adjust `shared_mem_bytes` if memory layout changes

---

## 📝 License

MIT

---

## 👤 Author

Wang Chen Han
[hank851107@gmail.com](mailto:hank851107@gmail.com)
GitHub: [HankWang-WL](https://github.com/HankWang-WL)
