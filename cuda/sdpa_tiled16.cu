// sdpa_tiled16.cu — warp-collab, no Out read/write per tile, deadlock-safe

extern "C" __global__
void sdpa_tiled16(
    const float* __restrict__ Q,   // [1,1,L,D]
    const float* __restrict__ K,   // [1,1,L,D]
    const float* __restrict__ V,   // [1,1,L,D]
    float* __restrict__ Out,       // [1,1,L,D]
    int L, int D,
    int BLOCK_M, int BLOCK_N, int BLOCK_D,
    int causal   // 1 or 0
){
    // 一個 block 有兩個 warp；各自負責一列 q_row
    const int lane    = threadIdx.x;      // 0..31
    const int warp_id = threadIdx.y;      // 0..1
    const int q_row   = blockIdx.y * 2 + warp_id;
    const bool active = (q_row < L);

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = K_smem + BLOCK_N * (BLOCK_D + 1);
    float* scores0 = V_smem + BLOCK_N * (BLOCK_D + 1);  // for warp 0
    float* scores1 = scores0 + BLOCK_N;                  // for warp 1
    float* my_scores = (warp_id == 0) ? scores0 : scores1;

    // online softmax 狀態
    float m = -1e30f;
    float l = 0.0f;

    // 每個 lane 同時維護兩個 d 的暫存器累加器（若 D<64，後面會自動跳過）
    float acc0 = 0.0f;                 // d0 = lane
    float acc1 = 0.0f;                 // d1 = lane + 32
    const int d0 = lane;
    const int d1 = lane + 32;

    const float* Qrow = active ? (Q + q_row * D) : nullptr;

    // 掃過所有 K/V tiles
    for (int k_start = 0; k_start < L; k_start += BLOCK_N) {
        const int tile_len = (BLOCK_N < (L - k_start)) ? BLOCK_N : (L - k_start);

        // 載入 K/V -> shared（兩個 warp 共 64 threads）
        const int tpb = blockDim.x * blockDim.y; // 64
        const int tid = warp_id * blockDim.x + lane;
        for (int idx = tid; idx < tile_len * D; idx += tpb) {
            const int t = idx / D;
            const int d = idx % D;
            if (d < BLOCK_D) {
                K_smem[t*(BLOCK_D+1) + d] = K[(k_start + t)*D + d];
                V_smem[t*(BLOCK_D+1) + d] = V[(k_start + t)*D + d];
            }
        }
        __syncthreads();

        // 計算我的 scores（此 warp 的那一列），並寫入 my_scores
        if (active) {
            const float inv_sqrt_d = 1.0f / sqrtf((float)D);
            for (int t = lane; t < tile_len; t += 32) {
                // 部分和（沿 d 維），這裡用 lane stride=32 來做
                float partial = 0.0f;
                for (int d = lane; d < D; d += 32) {
                    partial += Qrow[d] * K_smem[t*(BLOCK_D+1) + d];
                }
                // warp reduce
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    partial += __shfl_down_sync(0xffffffff, partial, off);

                float dot = __shfl_sync(0xffffffff, partial, 0);
                if (lane == 0) {
                    dot *= inv_sqrt_d;
                    if (causal && (k_start + t) > q_row) dot = -1e30f;
                    my_scores[t] = dot;
                }
            }
        }
        __syncthreads();

        // 這個 tile 的 max
        float tile_max = -1e30f;
        if (active) {
            for (int t = lane; t < tile_len; t += 32)
                tile_max = fmaxf(tile_max, my_scores[t]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffff, tile_max, off));
        }
        float m_tile = active ? __shfl_sync(0xffffffff, tile_max, 0) : -1e30f;

        // 分母更新
        float sum_tile = 0.0f;
        if (active) {
            float local_sum = 0.0f;
            for (int t = lane; t < tile_len; t += 32)
                local_sum += expf(my_scores[t] - m_tile);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
            sum_tile = __shfl_sync(0xffffffff, local_sum, 0);
        }

        float m_new = active ? fmaxf(m, m_tile) : m;
        float l_rescaled = active ? (l * expf(m - m_new)) : l;
        float l_new = active ? (l_rescaled + sum_tile) : l;

        // 輸出累加：只在暫存器做，不讀/寫 Out（每 tile 只做一次 rescale）
        if (active && l_new > 0.f) {
            const float rescale = l_rescaled / l_new;
            acc0 *= rescale;
            acc1 *= rescale;

            // 加上本 tile 的 PV / l_new
            for (int t = 0; t < tile_len; ++t) {
                const float w = expf(my_scores[t] - m_new) / l_new;
                if (d0 < D) acc0 += w * V_smem[t*(BLOCK_D+1) + d0];
                if (d1 < D) acc1 += w * V_smem[t*(BLOCK_D+1) + d1];
            }
        }

        m = active ? m_new : m;
        l = active ? l_new : l;

        __syncthreads();
    } // end tiles

    // 整輪 tiles 結束後，才一次性寫回 Out
    if (active) {
        if (d0 < D) Out[q_row*D + d0] = acc0;
        if (d1 < D) Out[q_row*D + d1] = acc1;
    }
}
