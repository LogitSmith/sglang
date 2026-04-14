"""
Lightning Indexer — cudeepy-generated fused GEMM + ApproxTopK for NSA.

Replaces the 12-kernel eager pipeline in nsa_indexer.py's _get_topk_ragged:
  per_token_group_quant → cuTensorMapEncodeTiled × 6 → sm100_fp8_gemm →
  LayerNormKernel → fused_rope → elementwise × 2 → fast_hadamard →
  fused_store_indexer_cache → topk_transform → cudaMemcpyAsync

With a SINGLE persistent CTA kernel:
  TMA Load → MMA GEMM → ReLU + Scale + Head-Reduce → ApproxTopK

The kernel expects Q post-RoPE+Hadamard and K post-RoPE+Hadamard+LayerNorm,
pre-scaled by gate weights. It runs in ~0.047ms vs ~0.19ms for the 12-kernel path.

Enable: SGLANG_USE_LIGHTNING_INDEXER=1

Source: gitlab-master.nvidia.com/mmaor/cudeepy
Kernel: generated/b300/cute/DSA/dsa_0_indexer_gemm_approx_topk_bf16/kernel.py
"""

import os
import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_USE_LIGHTNING = os.environ.get("SGLANG_USE_LIGHTNING_INDEXER", "0") == "1"
_lightning_available = False
_compiled_fn = None
_compile_lock = None

if _USE_LIGHTNING:
    try:
        import threading
        _compile_lock = threading.Lock()
        # Import will happen lazily on first call
        _lightning_available = True
        logger.info("[LightningIndexer] Enabled — will compile on first use")
    except Exception as e:
        logger.warning(f"[LightningIndexer] Failed to initialize: {e}")


def _get_or_compile_kernel():
    """Lazily compile the lightning indexer kernel on first use."""
    global _compiled_fn
    if _compiled_fn is not None:
        return _compiled_fn

    with _compile_lock:
        if _compiled_fn is not None:
            return _compiled_fn

        logger.info("[LightningIndexer] Compiling kernel (first use)...")

        from cutlass.cute.runtime import from_dlpack
        from cutlass.cute import experimental as cute_ext

        # Import the kernel builder
        from sglang.srt.layers.attention.nsa.lightning_indexer_kernel import (
            build_indexer_kernel,
            S_VAL, H_I_VAL, D_I_VAL, K_TOPK_VAL, NUM_CTAS_VAL, M_TILE_VAL,
        )

        # Create dummy tensors for compilation (shapes must match kernel expectations)
        device = "cuda"
        k_dummy = torch.zeros(S_VAL, D_I_VAL, dtype=torch.bfloat16, device=device)
        q_dummy = torch.zeros(H_I_VAL, D_I_VAL, dtype=torch.bfloat16, device=device)
        scratch = torch.zeros(M_TILE_VAL, H_I_VAL, dtype=torch.float32, device=device)
        scores = torch.zeros(S_VAL, dtype=torch.float32, device=device)
        topk_vals = torch.zeros(K_TOPK_VAL, dtype=torch.float32, device=device)
        topk_idxs = torch.zeros(K_TOPK_VAL, dtype=torch.int32, device=device)
        counter = torch.zeros(1 + NUM_CTAS_VAL, dtype=torch.int32, device=device)
        per_cta_idx = torch.zeros(NUM_CTAS_VAL * 128, dtype=torch.int32, device=device)

        # Convert to CuTe tensors
        k_c = from_dlpack(k_dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        q_c = from_dlpack(q_dummy, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        scratch_c = from_dlpack(scratch, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        scores_c = from_dlpack(scores, assumed_align=16).mark_layout_dynamic()
        vals_c = from_dlpack(topk_vals, assumed_align=16).mark_layout_dynamic()
        idxs_c = from_dlpack(topk_idxs, assumed_align=16).mark_layout_dynamic()
        counter_c = from_dlpack(counter, assumed_align=16).mark_layout_dynamic()
        per_cta_c = from_dlpack(per_cta_idx, assumed_align=16).mark_layout_dynamic()

        launch_fn = build_indexer_kernel()
        _compiled_fn = cute_ext.compile(
            launch_fn, k_c, q_c, scratch_c,
            scores_c, vals_c, idxs_c, counter_c, per_cta_c
        )

        logger.info("[LightningIndexer] Kernel compiled successfully")
        return _compiled_fn


# Pre-allocated scratch buffers (lazily created per device)
_buffers = {}


def _get_buffers(device: torch.device):
    """Get or create pre-allocated scratch buffers for the kernel."""
    key = str(device)
    if key not in _buffers:
        from sglang.srt.layers.attention.nsa.lightning_indexer_kernel import (
            S_VAL, H_I_VAL, D_I_VAL, K_TOPK_VAL, NUM_CTAS_VAL, M_TILE_VAL,
        )
        _buffers[key] = {
            'scratch': torch.zeros(M_TILE_VAL, H_I_VAL, dtype=torch.float32, device=device),
            'scores': torch.zeros(S_VAL, dtype=torch.float32, device=device),
            'topk_vals': torch.zeros(K_TOPK_VAL, dtype=torch.float32, device=device),
            'topk_idxs': torch.zeros(K_TOPK_VAL, dtype=torch.int32, device=device),
            'counter': torch.zeros(1 + NUM_CTAS_VAL, dtype=torch.int32, device=device),
            'per_cta_idx': torch.zeros(NUM_CTAS_VAL * 128, dtype=torch.int32, device=device),
        }
    return _buffers[key]


def lightning_topk(
    query_bf16: torch.Tensor,       # (H_I, D_I) bf16 — post-RoPE, post-Hadamard
    key_bf16: torch.Tensor,         # (S, D_I) bf16 — post-RoPE, post-Hadamard, post-LayerNorm
    weights: torch.Tensor,          # (H_I,) fp32 — gate weights from weights_proj
    softmax_scale: float,
    index_topk: int = 2048,
) -> torch.Tensor:
    """
    Run the lightning indexer kernel.

    Fuses: GEMM + ReLU + Scale + Head-Reduce + ApproxTopK
    in a single persistent CTA kernel (~0.047ms on B300).

    Args:
        query_bf16: (H_I, D_I) or (T, H_I, D_I) bf16 queries
        key_bf16: (S, D_I) bf16 keys
        weights: (H_I,) or (T, H_I) fp32 gate weights
        softmax_scale: attention softmax scale factor
        index_topk: number of top-K indices to return

    Returns:
        topk_indices: (T, index_topk) int32
    """
    from cutlass.cute.runtime import from_dlpack

    device = query_bf16.device
    compiled_fn = _get_or_compile_kernel()
    bufs = _get_buffers(device)

    H_I = query_bf16.shape[-2] if query_bf16.dim() == 3 else query_bf16.shape[0]
    D_I = query_bf16.shape[-1]

    # Handle batched (T, H_I, D_I) vs unbatched (H_I, D_I) queries
    if query_bf16.dim() == 3:
        T = query_bf16.shape[0]
        results = []
        for t in range(T):
            idx = _run_single_query(
                compiled_fn, bufs,
                query_bf16[t], key_bf16, weights[t] if weights.dim() > 1 else weights,
                softmax_scale, H_I, D_I, index_topk, device,
            )
            results.append(idx)
        return torch.stack(results, dim=0)
    else:
        idx = _run_single_query(
            compiled_fn, bufs,
            query_bf16, key_bf16, weights,
            softmax_scale, H_I, D_I, index_topk, device,
        )
        return idx.unsqueeze(0)


def _run_single_query(
    compiled_fn, bufs,
    q_bf16, k_bf16, w_fp32,
    softmax_scale, H_I, D_I, index_topk, device,
):
    """Run kernel for a single query token."""
    from cutlass.cute.runtime import from_dlpack

    # Pre-scale Q: q_scaled = q * q_s where q_s = w * H_I^(-0.5) * softmax_scale
    q_s = w_fp32 * (H_I ** -0.5) * softmax_scale
    q_scaled = (q_bf16.float() * q_s.unsqueeze(1)).to(torch.bfloat16)

    S = k_bf16.shape[0]

    # Reset scratch buffers
    bufs['scores'].zero_()
    bufs['topk_vals'].zero_()
    bufs['topk_idxs'].zero_()
    bufs['counter'].zero_()
    bufs['per_cta_idx'].zero_()

    # Pad K to S_VAL if needed (kernel expects fixed S=131072)
    from sglang.srt.layers.attention.nsa.lightning_indexer_kernel import S_VAL
    if S < S_VAL:
        k_padded = torch.zeros(S_VAL, D_I, dtype=torch.bfloat16, device=device)
        k_padded[:S] = k_bf16
    elif S == S_VAL:
        k_padded = k_bf16
    else:
        # S > S_VAL: truncate (shouldn't happen in practice for 128K context)
        k_padded = k_bf16[:S_VAL]

    # Convert to CuTe tensors
    k_c = from_dlpack(k_padded, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    q_c = from_dlpack(q_scaled, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    scratch_c = from_dlpack(bufs['scratch'], assumed_align=16).mark_layout_dynamic(leading_dim=1)
    scores_c = from_dlpack(bufs['scores'], assumed_align=16).mark_layout_dynamic()
    vals_c = from_dlpack(bufs['topk_vals'], assumed_align=16).mark_layout_dynamic()
    idxs_c = from_dlpack(bufs['topk_idxs'], assumed_align=16).mark_layout_dynamic()
    counter_c = from_dlpack(bufs['counter'], assumed_align=16).mark_layout_dynamic()
    per_cta_c = from_dlpack(bufs['per_cta_idx'], assumed_align=16).mark_layout_dynamic()

    # Launch kernel
    compiled_fn(k_c, q_c, scratch_c, scores_c, vals_c, idxs_c, counter_c, per_cta_c)

    # Extract results — filter out indices beyond actual S (from padding)
    topk_idxs = bufs['topk_idxs'][:index_topk].clone()
    if S < S_VAL:
        # Mask out padded indices
        topk_idxs[topk_idxs >= S] = -1

    return topk_idxs


def is_lightning_available() -> bool:
    """Check if lightning indexer is enabled and available."""
    return _USE_LIGHTNING and _lightning_available
