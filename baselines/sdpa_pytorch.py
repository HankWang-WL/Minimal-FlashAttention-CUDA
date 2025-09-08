import torch
import torch.nn.functional as F

def sdpa_pytorch(q, k, v, backend: str = "math", is_causal: bool = True):
    backend = backend.lower()
    if backend not in {"math", "mem_efficient", "flash"}:
        raise ValueError(f"Unknown backend: {backend}")
    with torch.backends.cuda.sdp_kernel(
        enable_flash=(backend != "math"),
        enable_mem_efficient=(backend != "math"),
        enable_math=True,
    ):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
