"""
Test harness for DeepSeek V3.2 Lightning Indexer kernel.
Compares GEMM + ReLU + scale + head-reduce + approxTopK against PyTorch reference.
"""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernel import build_indexer_kernel, S_VAL, H_I_VAL, D_I_VAL, K_TOPK_VAL, NUM_CTAS_VAL
from cutlass.cute.runtime import from_dlpack
from cutlass.cute import experimental as cute_ext
from cutlass import cute

# ─── Problem Configuration ───
S = S_VAL        # 131072
H_I = H_I_VAL    # 64
D_I = D_I_VAL    # 128
K_TOPK = K_TOPK_VAL  # 2048
M_TILE = 128

WARMUP_ITERS = 3
BENCH_ITERS = 10

# ─── PyTorch Reference ───
class IndexerReference(torch.nn.Module):
    def forward(self, k, q, w):
        """
        k: (S, D_I) BF16
        q: (H_I, D_I) BF16
        w: (H_I,) FP32 — per-head indexer weights
        Returns: topk_values (K,) FP32, topk_indices (K,) INT64
        """
        # GEMM: logits(S, H_I) = k @ q^T
        logits = torch.mm(k.float(), q.float().T)  # (S, H_I) FP32
        # ReLU
        logits = torch.relu(logits)
        # q_s = w * H_I^(-0.5) * D_I^(-0.5)
        q_s = w * (H_I ** -0.5) * (D_I ** -0.5)
        # Scale and reduce heads
        scores = (logits * q_s.unsqueeze(0)).sum(dim=1)  # (S,) FP32
        # Exact topK
        topk_vals, topk_idxs = torch.topk(scores, K_TOPK)
        return topk_vals, topk_idxs

# ─── Create Input Tensors ───
print(f"Problem: S={S}, H_I={H_I}, D_I={D_I}, K={K_TOPK}")
torch.manual_seed(42)

k_pt = torch.randn(S, D_I, dtype=torch.bfloat16, device="cuda")
q_pt = torch.randn(H_I, D_I, dtype=torch.bfloat16, device="cuda")
w_pt = torch.randn(H_I, dtype=torch.float32, device="cuda").abs()  # positive weights

# Precompute q_s and pre-scale Q (fuse scaling into GEMM)
q_s_pt = w_pt * (H_I ** -0.5) * (D_I ** -0.5)
q_scaled_pt = (q_pt.float() * q_s_pt.unsqueeze(1)).to(torch.bfloat16)

# Scratch buffer (for layout inference)
scratch_pt = torch.zeros(M_TILE, H_I, dtype=torch.float32, device="cuda")

# All-scores buffer (GMEM)
scores_pt = torch.zeros(S, dtype=torch.float32, device="cuda")

# Output tensors
topk_vals_pt = torch.zeros(K_TOPK, dtype=torch.float32, device="cuda")
topk_idxs_pt = torch.zeros(K_TOPK, dtype=torch.int32, device="cuda")

# Cross-CTA sync counter (element 0 = atomic counter, 1..NUM_CTAS = per-CTA flags)
counter_pt = torch.zeros(1 + NUM_CTAS_VAL, dtype=torch.int32, device="cuda")

# Per-CTA top-1 index buffer (scratch for compact Stage 2)
per_cta_idx_pt = torch.zeros(NUM_CTAS_VAL * 128, dtype=torch.int32, device="cuda")

# ─── Convert to CuTe Tensors ───
k_cute = from_dlpack(k_pt, assumed_align=16).mark_layout_dynamic(leading_dim=1)
q_cute = from_dlpack(q_scaled_pt, assumed_align=16).mark_layout_dynamic(leading_dim=1)
scratch_cute = from_dlpack(scratch_pt, assumed_align=16).mark_layout_dynamic(leading_dim=1)
scores_cute = from_dlpack(scores_pt, assumed_align=16).mark_layout_dynamic()
vals_cute = from_dlpack(topk_vals_pt, assumed_align=16).mark_layout_dynamic()
idxs_cute = from_dlpack(topk_idxs_pt, assumed_align=16).mark_layout_dynamic()
counter_cute = from_dlpack(counter_pt, assumed_align=16).mark_layout_dynamic()
per_cta_idx_cute = from_dlpack(per_cta_idx_pt, assumed_align=16).mark_layout_dynamic()

# ─── Build & Compile Kernel ───
print("Building kernel...")
launch_fn = build_indexer_kernel()

print("Compiling kernel...")
compiled_fn = cute_ext.compile(
    launch_fn, k_cute, q_cute, scratch_cute,
    scores_cute, vals_cute, idxs_cute, counter_cute, per_cta_idx_cute
)

# ─── Run Kernel ───
print("Launching kernel...")
compiled_fn(k_cute, q_cute, scratch_cute,
            scores_cute, vals_cute, idxs_cute, counter_cute, per_cta_idx_cute)
torch.cuda.synchronize()
print("Kernel completed.")

# ─── Reference ───
print("Computing reference...")
ref_model = IndexerReference()
ref_compiled = torch.compile(ref_model, mode="max-autotune")

# Warmup
for _ in range(3):
    _ = ref_compiled(k_pt, q_pt, w_pt)
torch.cuda.synchronize()

ref_vals, ref_idxs = ref_compiled(k_pt, q_pt, w_pt)
torch.cuda.synchronize()

# ─── Correctness: Recall ───
kernel_idx_set = set(topk_idxs_pt.cpu().numpy().tolist())
ref_idx_set = set(ref_idxs.cpu().numpy().tolist())
overlap = len(kernel_idx_set & ref_idx_set)
recall = overlap / K_TOPK if K_TOPK > 0 else 0.0

print(f"\n=== Correctness ===")
print(f"Recall: {overlap}/{K_TOPK} = {recall:.4f}")
print(f"Recall target: >= 0.70")
recall_pass = recall >= 0.70
print(f"RECALL: {'PASS' if recall_pass else 'FAIL'}")
assert recall_pass, f"Recall {recall:.4f} below threshold 0.70"

# Check value accuracy (top values should roughly match)
kernel_vals_sorted = topk_vals_pt.sort(descending=True).values
ref_vals_sorted = ref_vals.sort(descending=True).values[:K_TOPK]
val_diff = (kernel_vals_sorted[:100].cpu() - ref_vals_sorted[:100].cpu()).abs()
print(f"Top-100 values max diff: {val_diff.max().item():.4f}")
print(f"Top-100 values mean diff: {val_diff.mean().item():.4f}")

# ─── Benchmark ───
print(f"\n=== Benchmark ({BENCH_ITERS} iters) ===")

# Helper to reset state between iterations
def reset_state():
    scores_pt.zero_()
    topk_vals_pt.zero_()
    topk_idxs_pt.zero_()
    counter_pt.zero_()
    per_cta_idx_pt.zero_()

# Warmup
for _ in range(WARMUP_ITERS):
    reset_state()
    compiled_fn(k_cute, q_cute, scratch_cute, scores_cute, vals_cute, idxs_cute, counter_cute, per_cta_idx_cute)
torch.cuda.synchronize()

# Benchmark kernel
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)
start_evt.record()
for _ in range(BENCH_ITERS):
    reset_state()
    compiled_fn(k_cute, q_cute, scratch_cute, scores_cute, vals_cute, idxs_cute, counter_cute, per_cta_idx_cute)
end_evt.record()
torch.cuda.synchronize()
kernel_time_ms = start_evt.elapsed_time(end_evt) / BENCH_ITERS

k_data_bytes = S * D_I * 2  # BF16
effective_bw = k_data_bytes / (kernel_time_ms * 1e-3) / 1e12

print(f"Kernel: {kernel_time_ms:.3f} ms")
print(f"Effective BW: {effective_bw:.2f} TB/s (k data only)")

# Benchmark reference
start_evt.record()
for _ in range(BENCH_ITERS):
    _ = ref_compiled(k_pt, q_pt, w_pt)
end_evt.record()
torch.cuda.synchronize()
ref_time_ms = start_evt.elapsed_time(end_evt) / BENCH_ITERS

print(f"Reference: {ref_time_ms:.3f} ms")
print(f"Speedup: {ref_time_ms / kernel_time_ms:.2f}x")
