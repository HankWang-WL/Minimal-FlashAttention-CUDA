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

> Constraints: Pascal (GTX1060) has no `cp.async`; FP16/BF16 is not used. Double buffering is not expected to help here.

### ✅ O1 — Must-Do Improvements

1. **Vectorized Memory Access (`float4`)**

   * **What:** Load and store 4 floats at a time.
   * **Why:** Reduces global memory transactions → faster memory throughput.
   * **How:** Ensure `D % 4 == 0` (true for 64) and pointers are 16-byte aligned, then reinterpret pointers as `float4*`.

2. **Per-Lane Q Register Caching**

   * **What:** Each thread loads its `Q` values once and reuses them.
   * **Why:** Avoids repeatedly loading from global memory, reducing memory pressure.
   * **How:** Pre-load `q0`, `q1` before the inner loop and use them inside dot product computation.

3. **Tune `BLOCK_N` for Shared Memory Usage**

   * **What:** Choose a block size that balances occupancy and shared memory usage.
   * **Why:** Too large a tile can reduce occupancy; too small wastes bandwidth.
   * **How:** For Pascal, `BLOCK_N = 32` is a good starting point.

4. **Single Final Write to Output**

   * **What:** Keep accumulation (`acc0`, `acc1`) in registers until the end.
   * **Why:** Avoids repeated global writes → faster.

5. **Memory Alignment & Coalescing**

   * **What:** Ensure `Q/K/V` tensors are contiguous and aligned.
   * **Why:** Maximizes benefit from `float4` vectorization.

---

### 🚀 O2 — Optional Next Steps

1. **Short-Tile Score in Registers**

   * **What:** Keep scores in registers when tile size ≤ 32.
   * **Why:** Reduces shared-mem traffic and sync overhead.

2. **Unroll Inner Loops / FMA**

   * **What:** Unroll the dot-product loop to allow compiler to fuse operations.
   * **Why:** Improves instruction-level parallelism.

3. **Split D for Larger Models**

   * **What:** Process `D` in 64-wide chunks.
   * **Why:** Allows the kernel to support larger head dimensions without major redesign.


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
