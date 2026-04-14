"""
DeepSeek V3.2 Lightning Indexer: Fused GEMM + ReLU + Scale + HeadReduce + ApproxTopK
Single persistent CTA kernel on Blackwell B300 using cute_ext pattern.

Stage 1 (all 160 CTAs): GEMM → ReLU → scale → head-reduce → per-thread top-1 tracking
Stage 2 (last CTA only): top-2048 selection from 20480 per-CTA top-1 entries
"""
import math
import cutlass
from cutlass import cute
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils

# ── Constants ──
S_VAL = 131072
H_I_VAL = 64
D_I_VAL = 128
K_TOPK_VAL = 2048
M_TILE_VAL = 128
NUM_TILES_VAL = S_VAL // M_TILE_VAL  # 1024

MAINLOOP_STAGES = 4
EPILOGUE_STAGES = 2
MMA_INST_TILE_K = 8
NUM_WARPS = 6
EPI_SMEM_STAGES = 1  # Only need 1 stage since we read from SMEM (no TMA store)
CLUSTER_SHAPE = (1, 1, 1)
NUM_CTAS_VAL = 32


def build_indexer_kernel():
    """Build persistent single-CTA indexer: GEMM + ReLU + scale + headReduce + approxTopK."""
    use_2cta = False
    acc_dtype = cutlass.Float32
    mma_inst_shape_mnk = (128, 64, 16)

    @cute_ext.kernel
    def kernel(
        mA: cute.Tensor,           # k: (S, D_I) BF16
        mB: cute.Tensor,           # q_scaled: (H_I, D_I) BF16 — pre-scaled by q_s
        mScratch: cute.Tensor,     # (M_TILE, H_I) FP32 — for layout inference
        mScores: cute.Tensor,      # (S,) FP32 — reused: first 20480 as per-CTA top-1 vals
        mTopkVals: cute.Tensor,    # output: (K_TOPK,) FP32
        mTopkIdxs: cute.Tensor,    # output: (K_TOPK,) INT32
        mCounter: cute.Tensor,     # sync: (1 + NUM_CTAS,) INT32
        mPerCtaIdx: cute.Tensor,   # scratch: (NUM_CTAS*128,) INT32 — per-CTA top-1 indices
    ):
        ab_dtype = mA.element_type
        d_layout = utils.LayoutEnum.from_tensor(mScratch)
        d_dtype = mScratch.element_type

        mma_inst_m, mma_inst_n, mma_inst_k = mma_inst_shape_mnk
        cta_group = cute.nvgpu.tcgen05.CtaGroup.ONE

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            acc_dtype,
            cta_group,
            (mma_inst_m, mma_inst_n),
        )

        bM, bN = mma_inst_m, mma_inst_n  # 128, 64
        bK = mma_inst_k * MMA_INST_TILE_K  # 64
        mnk_tiler = (bM, bN, bK)

        cta_m, cta_n, _ = cute.arch.block_idx()
        tid_x, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Global tensor partitioning
        gA_tma = cute.zipped_divide(mA, (bM, bK))
        gB_tma = cute.zipped_divide(mB, (bN, bK))
        tBgB = gB_tma[(None, None), (0, None)]

        # SMEM layouts
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mnk_tiler, ab_dtype, MAINLOOP_STAGES,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mnk_tiler, ab_dtype, MAINLOOP_STAGES,
        )

        cta_tile_shape_mnk = (bM, bN, bK)
        epi_tile = sm100_utils.compute_epilogue_tile_shape(
            cta_tile_shape_mnk, use_2cta, d_layout, d_dtype,
        )
        sc_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            d_dtype, d_layout, epi_tile, EPI_SMEM_STAGES,
        )

        acc_shape = tiled_mma.partition_shape_C(mnk_tiler[:2])
        tmem_layout = tiled_mma.make_fragment_C(
            cute.append(acc_shape, EPILOGUE_STAGES)
        ).layout

        # Buffer allocations
        bufferA = cute_ext.allocate(
            ab_dtype, cute.AddressSpace.smem, a_smem_layout, alignment=1024,
        )
        bufferB = cute_ext.allocate(
            ab_dtype, cute.AddressSpace.smem, b_smem_layout, alignment=1024,
        )
        bufferAcc = cute_ext.allocate(
            acc_dtype, cute.AddressSpace.tmem, tmem_layout, alignment=16,
            is2cta=use_2cta,
        )
        bufferC = cute_ext.allocate(
            d_dtype, cute.AddressSpace.smem, sc_smem_layout_staged, alignment=1024,
        )

        # Bank-conflict-free SMEM for Stage 2 top-K: stride-17 for 128 threads
        # gcd(17, 32) = 1 → no bank conflicts within a warp
        topk_smem_layout = cute.make_layout(
            (128, 17), stride=(17, 1),
        )
        bufferTopK = cute_ext.allocate(
            acc_dtype, cute.AddressSpace.smem, topk_smem_layout, alignment=128,
        )
        # Separate SMEM buffer for indices (INT32, same layout for bank-conflict-free)
        topk_idx_smem_layout = cute.make_layout(
            (128, 17), stride=(17, 1),
        )
        bufferTopKIdx = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, topk_idx_smem_layout, alignment=128,
        )

        # T2R copy setup
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk, d_layout, d_dtype, acc_dtype, epi_tile, use_2cta,
        )
        accumulators = cute.zipped_divide(bufferAcc, ((epi_tile), 1))
        acc_epi_div = accumulators[((None, None), 0), 0]
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_epi_div)

        thr_copy_t2r = tiled_copy_t2r.get_slice(tid_x)
        # Use bufferC for RMEM layout inference (need to partition D from SMEM)
        sC_for_layout = cute.flat_divide(bufferC[None, None, 0], epi_tile)
        tTR_sC = thr_copy_t2r.partition_D(sC_for_layout)
        acc_d_rmem_layout = cute.make_fragment_like(
            tTR_sC[(None, None, None, 0, 0)].layout
        )

        bufferRAcc = cute_ext.allocate(
            acc_dtype, cute.AddressSpace.rmem, acc_d_rmem_layout, alignment=32,
        )

        # Pipelines for 1-CTA
        mma_operation_type = cute_ext.OperationTypeEnum.SM100_MMA_1SM_SS

        acc_pipe = cute_ext.UMMAtoAsyncPipeline.create(
            num_stages=EPILOGUE_STAGES,
            mma_operation_type=mma_operation_type,
            consumer=cute_ext.OperationTypeEnum.SM100_COPY_T2R,
            consumer_arv_count=128,
        )
        mainloop_pipe = cute_ext.TMAToUMMAPipeline.create(
            num_stages=MAINLOOP_STAGES,
            mma_operation_type=mma_operation_type,
        )

        mma_warp_id = 4
        tma_load_warp_id = 5

        k_tile_count = cute.size(gA_tma, mode=[1, 1])
        num_m_tiles = cute.size(gA_tma, mode=[1, 0])

        # ══════════ WARP-SPECIALIZED PERSISTENT LOOPS ══════════

        num_ctas = NUM_CTAS_VAL

        # === TMA Load Warp ===
        if warp_idx == tma_load_warp_id:
            for m_tile in cutlass.range(cta_m, num_m_tiles, num_ctas, unroll=1):
                tAgA_m = gA_tma[(None, None), (m_tile, None)]
                for k_tile in cutlass.range(0, k_tile_count, 1, unroll=1):
                    gA_k = tAgA_m[None, None, k_tile]
                    gB_k = tBgB[None, None, k_tile]

                    producer_stage_token, idx = mainloop_pipe.producer_acquire_and_get_stage()
                    mbar = cute_ext.get_mbarrier(producer_stage_token)
                    a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
                    b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

                    cute_ext.tma_load(
                        gA_k, bufferA[None, None, None, idx], mbar,
                        cta_v_map=a_cta_v_map,
                    )
                    cute_ext.tma_load(
                        gB_k, bufferB[None, None, None, idx], mbar,
                        cta_v_map=b_cta_v_map,
                    )

                    mainloop_pipe.producer_commit()
                    mainloop_pipe.producer_state = cute_ext.pipeline_advance_iterator(
                        mainloop_pipe.raw_pipeline, mainloop_pipe.producer_state
                    )

        # === MMA Warp ===
        if warp_idx == mma_warp_id:
            for m_tile in cutlass.range(cta_m, num_m_tiles, num_ctas, unroll=1):
                producer_stage_token, idx = acc_pipe.producer_acquire_and_get_stage()
                accumulators_sliced = bufferAcc[None, None, None, idx]

                mma_atom = cute.make_mma_atom(tiled_mma.op)
                mma_atom.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(0, k_tile_count, 1, unroll=1):
                    _, mainloop_idx = mainloop_pipe.consumer_wait_and_get_stage()
                    bufferA_stage = cute.core.slice_(
                        bufferA, (None, None, None, mainloop_idx)
                    )
                    bufferB_stage = cute.core.slice_(
                        bufferB, (None, None, None, mainloop_idx)
                    )

                    for k_block in cutlass.range(MMA_INST_TILE_K, unroll_full=True):
                        cute_ext.dot(
                            mma_atom,
                            cute.append_ones(
                                bufferA_stage[None, None, k_block], up_to_rank=3
                            ),
                            cute.append_ones(
                                bufferB_stage[None, None, k_block], up_to_rank=3
                            ),
                            accumulators_sliced,
                        )
                        mma_atom.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)

                    mainloop_pipe.consumer_release_and_advance()

                acc_pipe.producer_commit_and_advance()

        # === Epilogue Warps: Direct RMEM head reduce + per-thread top-1 tracking ===
        # Q is pre-scaled by q_s on host, so score = sum(relu(acc_values))
        if warp_idx < 4:
            num_rmem_elems = cute.size(acc_d_rmem_layout)

            # Per-thread max tracking across ALL tiles (register-based)
            best_score = cutlass.Float32(-1e30)
            best_pos = cutlass.Int32(0)

            for m_tile in cutlass.range(cta_m, num_m_tiles, num_ctas, unroll=1):
                _, idx = acc_pipe.consumer_wait_and_get_stage()
                accumulators_sliced = bufferAcc[(None, None), 0, 0, idx]
                acc_epi_tiled = cute.flat_divide(accumulators_sliced, epi_tile)

                # Accumulate score across subtiles via direct RMEM sum
                score = cutlass.Float32(0.0)
                subtile_cnt = cute.size(acc_epi_tiled.shape, mode=[3])
                for mn in range(subtile_cnt):
                    # TMEM -> RMEM
                    cute_ext.partition_and_copy(
                        tiled_copy_t2r.get_slice(tid_x),
                        acc_epi_tiled[None, None, 0, mn],
                        bufferRAcc,
                    )

                    # ReLU + sum directly from RMEM (no store-back needed)
                    for ri in range(num_rmem_elems):
                        val = bufferRAcc[ri]
                        if val > cutlass.Float32(0.0):
                            score = score + val

                acc_pipe.consumer_release_and_advance()

                # Track per-thread top-1 in registers
                global_pos = m_tile * M_TILE_VAL + tid_x
                if score > best_score:
                    best_score = score
                    best_pos = global_pos

            # Write per-CTA top-1 to compact GMEM buffers
            compact_idx = cta_m * 128 + tid_x
            mScores[compact_idx] = best_score
            mPerCtaIdx[compact_idx] = best_pos

        # ══════════ CROSS-CTA SYNC: Atomic counter to find last CTA ══════════
        # Ensure all score writes from this CTA are globally visible
        cute.arch.fence_acq_rel_gpu()
        cute.arch.sync_threads()

        # Thread 0 of warp 0 increments global counter, stores result per-CTA
        counter_base_ptr = mCounter.iterator  # _Pointer to mCounter[0]
        if warp_idx == 0:
            if tid_x == 0:
                old_count = cute.arch.atomic_add(
                    counter_base_ptr.llvm_ptr, cutlass.Int32(1),
                    sem="relaxed", scope="gpu",
                )
                # Store result to per-CTA slot for broadcast via GMEM
                per_cta_ptr = counter_base_ptr + (1 + cta_m)
                cute.arch.store(per_cta_ptr.llvm_ptr, old_count)

        cute.arch.sync_threads()

        # All threads read this CTA's flag from GMEM
        per_cta_read_ptr = counter_base_ptr + (1 + cta_m)
        flag_val = cute.arch.load(per_cta_read_ptr.llvm_ptr, cutlass.Int32)

        # ══════════ STAGE 2: Top-K from compact per-CTA buffer ══════════
        # Only the last CTA to finish executes Stage 2
        if flag_val == num_ctas - 1:
            cute.arch.fence_acq_rel_gpu()

            if warp_idx < 4:
                # Scan per-CTA top-2 entries to find global top-2048
                KPRIME = 16
                COMPACT_SIZE = NUM_CTAS_VAL * 128  # 1 entry per thread
                out_base = tid_x * KPRIME

                # Initialize SMEM top-K values to -inf
                for ic in range(KPRIME):
                    bufferTopK[(tid_x, ic)] = cutlass.Float32(-1e30)
                    bufferTopKIdx[(tid_x, ic)] = cutlass.Int32(0)

                min_val = cutlass.Float32(-1e30)
                min_col = cutlass.Int32(0)

                # Scan compact buffer with coalesced access pattern
                for j in cutlass.range(0, COMPACT_SIZE, 128, unroll=1):
                    buf_idx = j + tid_x
                    new_val = mScores[buf_idx]
                    if new_val > min_val:
                        bufferTopK[(tid_x, min_col)] = new_val
                        bufferTopKIdx[(tid_x, min_col)] = mPerCtaIdx[buf_idx]
                        # Rescan SMEM for new minimum
                        min_val = bufferTopK[(tid_x, 0)]
                        min_col = cutlass.Int32(0)
                        for rc in range(1, KPRIME):
                            rc_val = bufferTopK[(tid_x, rc)]
                            if rc_val < min_val:
                                min_val = rc_val
                                min_col = cutlass.Int32(rc)

                # Copy final values and indices from SMEM to GMEM output
                for oc in range(KPRIME):
                    mTopkVals[out_base + oc] = bufferTopK[(tid_x, oc)]
                    mTopkIdxs[out_base + oc] = bufferTopKIdx[(tid_x, oc)]

    @cute_ext.jit
    def launch_kernel(
        mA: cute.Tensor,
        mB: cute.Tensor,
        mScratch: cute.Tensor,
        mScores: cute.Tensor,
        mTopkVals: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mCounter: cute.Tensor,
        mPerCtaIdx: cute.Tensor,
    ):
        kernel(mA, mB, mScratch, mScores,
               mTopkVals, mTopkIdxs, mCounter, mPerCtaIdx).launch(
            grid=(NUM_CTAS_VAL, 1, 1),
            block=(32 * NUM_WARPS, 1, 1),
            cluster=CLUSTER_SHAPE,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
        )

    return launch_kernel
