/*
 * YOLO int8-TFMA convolution kernel (the DnCNN int8 engine brought onto the YOLO microbench).
 *
 * M-i0 exercises ONE int8 1x1 conv layer (the microbench conv1x1 shape: IC=OC=CH=16, ReLU6),
 * computed on the tensor engine and gated bit-exact (max_abs=0) vs a numpy oracle at 1 hart.
 * The compute primitives (pack_b_group / fma_group / requant_u8 / store_u8_tile / conv_tile) are
 * written K-parameterized -- at K=1 (this build) the halo/tap-fold collapse; M-i1 rebuilds the
 * SAME file with -DYOLO_K=3 (n_taps>1) with no forked kernel (PROMPT_0 s12).
 *
 * Reads a file-loaded uint8 activation @ACT_OFFSET (the B operand, NHWC) and a file-loaded int8
 * weight A-blob @WEIGHTS_OFFSET (group-major, gen_yolo_int8.py's layout), writes uint8 @OUTPUT_OFFSET.
 * Scales come from the generated yolo_int8_scales.h.
 *
 * Quant scheme (see gen_yolo_int8.py / optimizations.md): activations are ReLU6 outputs in [0,6]
 * quantized with S=6/255, so the tensor engine's SATUINT8 [0,255] clamp IS the ReLU6 ceiling
 * (quantized value of 6 == 255). Activations unsigned (B-gate), weights signed (A-gate).
 */

#include <stdint.h>

#define ERBIUM_TENSOR_ASSERT(cond) ((void)(cond))
#include "erbium/isa/atomic.h"
#include "erbium/isa/barriers.h"
#include "erbium/isa/cacheops-umode.h"
#include "erbium/isa/hart.h"
#include "erbium/isa/tensors.h"
#include "yolo_int8_scales.h"   /* generated: YOLO_QUANT0, YOLO_REQUANT[], YOLO_DEQUANT, YOLO_RELU6_HI */
#ifdef DNCNN_PMC
#include "pmc_probe.h"          /* -DDNCNN_PMC (+ -I ported_models/dncnn/src): HW perf-counter probe; else absent */
#endif

extern char heap0_end[];

#define YOLO_MAGIC 0x10510008u   /* int8 attestation magic (distinct from the FP32 bench's 0x10500001) */

#ifndef YOLO_INT8
#error "yolo_int8_argbuf.c is the int8 engine; build with -DYOLO_INT8=1 (the FP32 path is yolo_vpu_argbuf.c)"
#endif

/* ---- multi-hart config (M-i0 verifies at ACTIVE_HARTS=1; board 8-hart is M-i3) ---- */
#ifndef ACTIVE_HARTS
#define ACTIVE_HARTS 1u
#endif
#define BENCH_FLB 2u
#define BENCH_FCC FCC_0

/* ================= compile-time shape (mirror the microbench; hot dims stay constant) ========= */
#define IMG_W    80u
#define IMG_H    80u
#define CH       16u                /* IC = OC = CH (single OC tile, no IC tiling)          */
#ifndef YOLO_K
#define YOLO_K   1u                 /* op-class: 1 for M-i0 1x1; M-i1 rebuilds with 3        */
#endif
#define K        (YOLO_K)
/* YOLO_REQUANT is a per-op-class scale; guard against pairing a K kernel with another K's generated
 * header (e.g. the K=3 header + a K=1 build => a silent 2x requant error). Regen gen_yolo_int8.py <K>
 * to match -DYOLO_K if this fails. YOLO_SCALES_K is stamped into the generated header. */
_Static_assert(YOLO_SCALES_K == K,
	"yolo_int8_scales.h was generated for a different K -- rerun gen_yolo_int8.py <K> to match -DYOLO_K");
#define HALO     ((K - 1u) / 2u)    /* replicate-pad ring width (0 for 1x1)                  */
#define TAPS     (K * K)
#define P        16u                /* spatial tile width (= FMA bcols; HW 2-bit => <=16)    */
#define OC_TILE  16u                /* output channels per pass (HW a_num_rows => <=16)      */
#define QUARTET  4u                 /* int8 IC packing unit (4 IC per FMA step)              */
#define OPCODE_INT8 3u
#define ROW_STRIDE_BYTES 64u        /* one L1 scratchpad cache line                          */

/* Padded row stride: PADW*CH must be a whole number of 64B lines so no cache line spans two
 * harts' bands on write-back (the int8 seam-race invariant, re-derived per (CH,W,K)). PADW is a
 * multiple of PADW_MULT = 64/CH covering IMG_W + 2*HALO. At CH=16,HALO=0: PADW=80 (80*16=1280=20
 * lines, already clean -- the 1x1 needs no width padding, PROMPT_0 s0). */
#define PADW_MULT (64u / CH)
#define PADW     (((IMG_W + 2u * HALO + PADW_MULT - 1u) / PADW_MULT) * PADW_MULT)
#define PADH     (IMG_H + 2u * HALO)
_Static_assert((PADW * CH) % 64u == 0u,
	"row stride must be a whole number of 64B lines so no line spans two bands (seam invariant)");
_Static_assert(PADW >= IMG_W + 2u * HALO, "PADW must cover the halo-padded width");

/* ---- adaptive tap-folding (generalizes DnCNN B8; collapses at K=1) ---- */
#define FOLD_BUDGET    (16u / (CH / QUARTET))                 /* =4 at CH=16 (acols <=16 quartets) */
#define FOLD_TAPS      ((K < FOLD_BUDGET) ? K : FOLD_BUDGET)  /* =1 at K=1, =3 at K=3/CH=16        */
#define CHUNKS_PER_ROW ((K + FOLD_TAPS - 1u) / FOLD_TAPS)     /* =1 at K=1                          */
#define NGRP           (K * CHUNKS_PER_ROW)                   /* FMA dispatches per tile: 1 at K=1  */
#define NT_AT(c)       ((K - (c) < FOLD_TAPS) ? (K - (c)) : FOLD_TAPS)
_Static_assert(FOLD_TAPS >= 1u, "must fold at least one tap");
_Static_assert(FOLD_TAPS * (CH / QUARTET) <= 16u, "folded contraction must fit acols (<=16 quartets)");

/* ---- L1 scratchpad line allocation for one int8 tile (per hart) ---- */
#define SCP_A_LINE     0u                                        /* weights A: OC_TILE lines        */
#define SCP_B_LINE     OC_TILE                                   /* activations B: FOLD_TAPS*(CH/Q)  */
#define SCP_SCALE_LINE (SCP_B_LINE + FOLD_TAPS * (CH / QUARTET)) /* per-OC requant scale line        */
_Static_assert(OC_TILE <= 16u, "OC_TILE must fit a_num_rows (4-bit) and 32 FREGs (16 OC x 16 spatial)");
_Static_assert(P <= 16u, "P must fit b_num_col (2-bit) => <=16 spatial");
_Static_assert(SCP_SCALE_LINE < 48u, "SCP scale line exceeds the 48-line L1 scratchpad");

/* Baked-in shape assumptions; a wrong constant fails the build, not the board. */
_Static_assert(CH == 16u, "M-i0 shape is CH=16 (single OC tile, whole IC-quartets, 64B-clean rows)");
_Static_assert(CH % QUARTET == 0u, "CH must pack into whole IC-quartets");
_Static_assert(64u % CH == 0u, "CH must divide 64 so PADW_MULT is exact");
_Static_assert(P == 16u, "P must be 16 (FMA bcols / one 16B store block)");
_Static_assert(IMG_W % P == 0u, "IMG_W must be a multiple of P");
_Static_assert(ACTIVE_HARTS >= 1u && ACTIVE_HARTS <= 32u, "ACTIVE_HARTS in [1,32]");
_Static_assert(ACTIVE_HARTS <= IMG_H, "ACTIVE_HARTS must not exceed image rows");

/* Per-hart working-buffer strides (whole 64B lines so per-hart slots never share a line). */
#define BPK_STRIDE  (TAPS * (CH / QUARTET) * ROW_STRIDE_BYTES)  /* per-hart B pack: 256B at K=1 */
#define TEMP_STRIDE (OC_TILE * P)                               /* per-hart quant out [OC][P]: 256B */
_Static_assert(BPK_STRIDE  % 64u == 0u, "BPK_STRIDE must be a whole number of 64B lines");
_Static_assert(TEMP_STRIDE % 64u == 0u, "TEMP_STRIDE must be a whole number of 64B lines");

/* ================= memory map (byte offsets from base; match the run script's --file_load) =====
 * zero2m.bin zeroes [0,2MB); then the runner --file_load's the uint8 activation @ACT_OFFSET and the
 * int8 A-blob @WEIGHTS_OFFSET; the dump + checker read uint8 @OUTPUT_OFFSET. MVEC/BPACK/TEMP are
 * device-built scratch. All regions 64B-aligned, asserted non-overlapping and within the 16MB window. */
#ifndef YOLO_TRUNK
#define SLOTS_OFFSET    0x00000u    /* per-hart attestation slots (ACTIVE_HARTS x 64B) */
#define SLOT_BYTES      64u
#define SUMMARY_OFFSET  0x01000u
#define BARRIER_OFFSET  0x01800u
#define ACT_OFFSET      0x02000u    /* uint8 [IMG_H][IMG_W][CH] NHWC activation (B operand)   */
#define WEIGHTS_OFFSET  0x20000u    /* int8 A-blob, group-major NGRP*CH lines                 */
#define MVEC_OFFSET     0x28000u    /* per-OC requant scale, one 64B line (device-built)      */
#define BPACK_OFFSET    0x2A000u    /* per-hart folded B pack                                 */
#define TEMP_OFFSET     0x30000u    /* per-hart quant output [OC][P]                          */
#define OUTPUT_OFFSET   0x40000u    /* uint8 [IMG_H][IMG_W][CH] NHWC result                   */

#define ACT_BYTES     (IMG_H * IMG_W * CH)   /* PADW==IMG_W at K=1 => activation is plain NHWC */
#define PAD_BYTES     (PADH * PADW * CH)
#define AW_BYTES      (NGRP * CH * ROW_STRIDE_BYTES)
#define OUT_BYTES     (IMG_H * IMG_W * CH)
#define MVEC_BYTES    (((CH * 4u + 63u) / 64u) * 64u)

_Static_assert(ACT_OFFSET     % 64u == 0u, "act misaligned");
_Static_assert(WEIGHTS_OFFSET % 64u == 0u, "weights misaligned");
_Static_assert(MVEC_OFFSET    % 64u == 0u, "mvec misaligned");
_Static_assert(BPACK_OFFSET   % 64u == 0u, "bpk misaligned");
_Static_assert(TEMP_OFFSET    % 64u == 0u, "temp misaligned");
_Static_assert(OUTPUT_OFFSET  % 64u == 0u, "output misaligned");
_Static_assert(ACTIVE_HARTS * SLOT_BYTES <= SUMMARY_OFFSET,        "slots overflow into summary");
_Static_assert(ACT_OFFSET     + PAD_BYTES  <= WEIGHTS_OFFSET,      "activation overlaps weights");
_Static_assert(WEIGHTS_OFFSET + AW_BYTES   <= MVEC_OFFSET,         "weights overlap mvec");
_Static_assert(MVEC_OFFSET    + MVEC_BYTES <= BPACK_OFFSET,        "mvec overlaps bpk");
_Static_assert(ACTIVE_HARTS * BPK_STRIDE   <= (TEMP_OFFSET - BPACK_OFFSET),   "per-hart bpk overflow");
_Static_assert(ACTIVE_HARTS * TEMP_STRIDE  <= (OUTPUT_OFFSET - TEMP_OFFSET),  "per-hart temp overflow");
_Static_assert(OUTPUT_OFFSET  + OUT_BYTES  <= 16u * 1024u * 1024u, "buffers exceed the 16MB window");

#else  /* ================= YOLO_TRUNK: full-network mixed-precision memory map ============== */
/* M-i2: FP32 first conv (block-0 conv3) + FP32 head1x1 built ON DEVICE from init_model's formula;
 * 7 hidden convs int8 from a file-loaded concatenated A-blob. Two padded uint8 ping-pong buffers.
 * Output + probes sit low (below the big input/act buffers) so a small dump window captures them. */
#define SLOTS_OFFSET    0x00000u
#define SLOT_BYTES      64u
#define SUMMARY_OFFSET  0x01000u
#define BARRIER_OFFSET  0x01800u
#define WEIGHTS_OFFSET  0x02000u    /* int8 A-blob (L1..L7 concat, group-major), file-loaded     */
#define MVEC_OFFSET     0x06000u    /* per-OC requant scale (one 64B line, rebuilt per int8 layer)*/
#define FPW_OFFSET      0x07000u    /* FP32 first-conv W0 (2304 f) + head W (256 f), device-built */
#define OUTPUT_OFFSET   0x10000u    /* uint8 [IMG_H][IMG_W][CH] NHWC final head output            */
#define PROBE1_OFFSET   0x30000u    /* uint8 L1 (block0 conv1) int8 activation probe             */
#define PROBE2_OFFSET   0x50000u    /* uint8 L2 (block1 conv3) int8 activation probe             */
#define INPUT_OFFSET    0x80000u    /* FP32 [IMG_H][IMG_W][CH] NHWC input (device-built)          */
#define ACTA_OFFSET     0xF0000u    /* padded uint8 [PADH][PADW][CH] activation (ping)           */
#define ACTB_OFFSET     0x120000u   /* padded uint8 [PADH][PADW][CH] activation (pong)           */
#define BPACK_OFFSET    0x140000u   /* per-hart folded B pack                                     */
#define TEMP_OFFSET     0x148000u   /* per-hart quant output [OC][P]                              */

#define ACT_NHWC_BYTES  (IMG_H * IMG_W * CH)                 /* 102400: plain NHWC uint8          */
#define PAD_BYTES       (PADH * PADW * CH)                   /* 110208: padded uint8              */
#define FPW_FLOATS      (CH * K * K * CH + CH * CH)          /* 2304 (L0 conv3) + 256 (head)      */
#define FPW_BYTES       (FPW_FLOATS * 4u)                    /* 10240                             */
#define INPUT_BYTES     (IMG_H * IMG_W * CH * 4u)            /* 409600                            */
#define OUT_BYTES       (IMG_H * IMG_W * CH)
#define MVEC_BYTES      (((CH * 4u + 63u) / 64u) * 64u)
#define HEAD_FLAT_BASE  (4u * (CH * K * K * CH + CH * CH))   /* init_model flat index of head W   */
/* int8 A-blob: L1,L3,L5,L7 are conv1 (NGRP=1 -> CH lines); L2,L4,L6 conv3 (NGRP=3 -> 3*CH lines).*/
#define AW_CONV1_BYTES  (CH * ROW_STRIDE_BYTES)              /* 1024 */
#define AW_CONV3_BYTES  (3u * CH * ROW_STRIDE_BYTES)         /* 3072 */
#define AW_BYTES        (4u * AW_CONV1_BYTES + 3u * AW_CONV3_BYTES)   /* 13312 */

_Static_assert(YOLO_SCALES_K == 3u, "trunk uses K=3 padded geometry; regen gen_yolo_int8.py trunk");
_Static_assert(YOLO_TRUNK_LAYERS == 7, "trunk has 7 int8 hidden convs (block0 conv1 .. block3 conv1)");
_Static_assert(K == 3u && HALO == 1u && PADW == 84u && PADH == 82u, "trunk padded geometry");
_Static_assert(WEIGHTS_OFFSET % 64u == 0u && MVEC_OFFSET % 64u == 0u && FPW_OFFSET % 64u == 0u, "align");
_Static_assert(OUTPUT_OFFSET % 64u == 0u && PROBE1_OFFSET % 64u == 0u && PROBE2_OFFSET % 64u == 0u, "align");
_Static_assert(INPUT_OFFSET % 64u == 0u && ACTA_OFFSET % 64u == 0u && ACTB_OFFSET % 64u == 0u, "align");
_Static_assert(BPACK_OFFSET % 64u == 0u && TEMP_OFFSET % 64u == 0u, "align");
_Static_assert((PADW * CH) % 64u == 0u, "padded row stride must be whole 64B lines (seam invariant)");
_Static_assert((IMG_W * CH) % 64u == 0u, "NHWC output/probe row stride must be whole 64B lines");
_Static_assert(WEIGHTS_OFFSET + AW_BYTES   <= MVEC_OFFSET,   "A-blob overlaps mvec");
_Static_assert(MVEC_OFFSET    + MVEC_BYTES <= FPW_OFFSET,    "mvec overlaps fpw");
_Static_assert(FPW_OFFSET     + FPW_BYTES  <= OUTPUT_OFFSET, "fpw overlaps output");
_Static_assert(OUTPUT_OFFSET  + OUT_BYTES  <= PROBE1_OFFSET, "output overlaps probe1");
_Static_assert(PROBE1_OFFSET  + ACT_NHWC_BYTES <= PROBE2_OFFSET, "probe1 overlaps probe2");
_Static_assert(PROBE2_OFFSET  + ACT_NHWC_BYTES <= INPUT_OFFSET,  "probe2 overlaps input");
_Static_assert(INPUT_OFFSET   + INPUT_BYTES <= ACTA_OFFSET,  "input overlaps actA");
_Static_assert(ACTA_OFFSET    + PAD_BYTES  <= ACTB_OFFSET,   "actA overlaps actB");
_Static_assert(ACTB_OFFSET    + PAD_BYTES  <= BPACK_OFFSET,  "actB overlaps bpk");
_Static_assert(ACTIVE_HARTS * BPK_STRIDE  <= (TEMP_OFFSET - BPACK_OFFSET),  "per-hart bpk overflow");
_Static_assert(TEMP_OFFSET + ACTIVE_HARTS * TEMP_STRIDE <= 16u * 1024u * 1024u, "buffers exceed 16MB");
#ifdef DNCNN_PMC
/* PMC dump region in the free FPW..OUTPUT gap (FPW ends 0x9800); captured by the 0x70000 dump window.
 * Off by default -> the seam-clean production build is byte-unchanged. decode_pmc.py --base 0xA000. */
#define PMC_OFFSET 0xA000u
_Static_assert(PMC_OFFSET >= FPW_OFFSET + FPW_BYTES,                 "PMC region collides with fpw");
_Static_assert(PMC_OFFSET + sizeof(struct pmc_region) <= OUTPUT_OFFSET, "PMC region overruns into output");
#endif
#endif  /* YOLO_TRUNK memory map */

/* pointer to padded pixel (y,x)'s CH channels; y,x are REAL coords (-HALO..IMG-1+HALO ok) */
#define PAD_AT(pad, y, x) ((pad) + ((uint32_t)((y) + (int)HALO) * PADW + (uint32_t)((x) + (int)HALO)) * CH)

/* tensor_fma(tenc_loc=1) overwrites f0..f31 invisibly to GCC -> force a clobber to spill them. */
#define FREG_CLOBBER_BARRIER() __asm__ __volatile__("" ::: "memory", \
	"f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15", \
	"f16","f17","f18","f19","f20","f21","f22","f23","f24","f25","f26","f27","f28","f29","f30","f31")

/* ================= attestation structs (CI scorer reads these; keep in sync with FP32 bench) ==== */
struct yolo_slot {
	uint32_t magic;
	uint32_t hart_id;
	uint32_t minion_id;
	uint32_t thread_id;
	uint32_t row0;
	uint32_t row1;
	uint32_t active_harts;
	uint32_t checksum;
	uint32_t done;
	uint32_t reserved[7];
};
_Static_assert(sizeof(struct yolo_slot) == SLOT_BYTES, "slot must be exactly one cache line");

struct yolo_summary {
	uint32_t magic;
	uint32_t active_harts;
	uint32_t passes;
	uint32_t width;
	uint32_t height;
	uint32_t channels;
	uint32_t blocks;
	uint32_t active_mask;
	uint32_t done_count;
	uint32_t output_sum;
	uint32_t slot_checksum_sum;
	uint32_t ops_lo;
	uint32_t ops_hi;
	uint32_t head_channels;
	uint32_t reserved[2];
};
_Static_assert(sizeof(struct yolo_summary) == 16u * 4u, "summary must be 16 x uint32 for the scorer");

static uintptr_t buffer_base_from_args(uintptr_t arg_area)
{
	if (arg_area == 0u || arg_area == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}
	const uintptr_t ptr = *(volatile uintptr_t *)arg_area;
	if (ptr == 0u || ptr == ~(uintptr_t)0u) {
		return (uintptr_t)heap0_end - (16u * 1024u * 1024u);
	}
	return ptr;
}

/* ================= multi-hart barrier + hart-id (from the FP32 bench kernel) ================== */
struct bench_barrier_state {
	uint32_t count;
	uint32_t epoch;
	uint32_t reserved[14];
};
static volatile struct bench_barrier_state *g_barrier;

static inline uint32_t active_mask_t0(void)
{
#ifdef BENCH_THREAD0_ONLY
	if (ACTIVE_HARTS >= 32u) return 0xffffffffu;
	return (1u << ACTIVE_HARTS) - 1u;
#else
	uint32_t mask = 0;
	for (uint32_t h = 0; h < ACTIVE_HARTS; h += 2u) mask |= 1u << (h >> 1);
	return mask;
#endif
}
static inline uint32_t active_mask_t1(void)
{
#ifdef BENCH_THREAD0_ONLY
	return 0u;
#else
	uint32_t mask = 0;
	for (uint32_t h = 1; h < ACTIVE_HARTS; h += 2u) mask |= 1u << (h >> 1);
	return mask;
#endif
}
static inline uint32_t bench_hart_id(void)
{
#ifdef BENCH_THREAD0_ONLY
	return get_minion_id();
#else
	return get_hart_id() & 0x3fu;
#endif
}
static inline int bench_hart_enabled(uint32_t hart_id)
{
#ifdef BENCH_THREAD0_ONLY
	return get_thread_id() == 0u && hart_id < ACTIVE_HARTS && hart_id < 32u;
#else
	return hart_id < ACTIVE_HARTS && hart_id < 16u;
#endif
}
static inline void bench_barrier(void)
{
	if (ACTIVE_HARTS > 1u) {
#ifdef BENCH_THREAD0_ONLY
		volatile struct bench_barrier_state *const barrier = g_barrier;
		const uint32_t epoch = atomic_load_local_32(&barrier->epoch);
		const uint32_t prior = atomic_add_local_32(&barrier->count, 1u);
		if (prior + 1u == ACTIVE_HARTS) {
			atomic_store_local_32(&barrier->count, 0u);
			FENCE;
			atomic_add_local_32(&barrier->epoch, 1u);
		} else {
			while (atomic_load_local_32(&barrier->epoch) == epoch) FENCE;
		}
		FENCE;
#else
		shire_barrier(BENCH_FLB, BENCH_FCC, ACTIVE_HARTS,
			      active_mask_t0(), active_mask_t1());
#endif
	}
}

/* ================= reusable K-parameterized tile primitives (each CSR encoding lives here once) ===
 * At K=1 (M-i0) every fold/halo term collapses: NGRP=1, n_taps=1, dy=0, kx0=0. Hot dims are the
 * compile-time config (CH,K,P) so the pack word-copy stays unrolled and constant-strided. */

/* An IC-quartet (4 int8) is 4 contiguous 4-aligned bytes in both the activation and the packed B
 * line, so copy it as one aligned word (P1 word-copy). may_alias keeps it strict-aliasing safe. */
typedef uint32_t quartet_word __attribute__((__may_alias__));

/* Gather one group's activation windows (kernel row dy, taps kx0..kx0+n_taps-1) into the
 * quartet-interleaved B layout: line ti*(CH/QUARTET)+q holds [tap ti][quartet q]; byte j*4+x =
 * ic(4q+x) of spatial column j. Matches the A byte order (tap ti at A bytes ti*CH..). */
static inline void pack_b_group(int8_t *restrict bpk, const uint8_t *restrict pad,
				uint32_t y, uint32_t x0, int dy, uint32_t kx0, uint32_t n_taps)
{
	for (uint32_t ti = 0; ti < n_taps; ti++) {
		const uint8_t *restrict row =
			PAD_AT(pad, (int)y + dy, (int)x0 + ((int)(kx0 + ti) - (int)(K / 2u)));
		for (uint32_t q = 0; q < CH / QUARTET; q++) {
			int8_t *restrict dstl = bpk + (ti * (CH / QUARTET) + q) * ROW_STRIDE_BYTES;
			const uint8_t *restrict srcq = row + q * QUARTET;
			for (uint32_t j = 0; j < P; j++)
				*(quartet_word *)(dstl + j * QUARTET) =
					*(const quartet_word *)(srcq + j * CH);
		}
	}
}

/* One group's int8 MAC (oc OC x n_taps*CH IC x P spatial) in a single dispatch. A (weights) signed,
 * B (activations) unsigned -- mind the tensors.h wrapper name swap: arg tena_unsigned -> bit22 gates
 * B (=true, unsigned); arg tenb_unsigned -> bit21 gates A (=false, signed). Verified in tensors.cpp
 * (bit22=ub gates B @1432/1507, bit21=ua gates A @1433/1499). first resets TENC; last copies TENC->FREGs. */
static inline void fma_group(int first, int last, uint32_t n_taps, uint32_t oc)
{
	tensor_fma(false, (P / QUARTET) - 1u, oc - 1u, n_taps * (CH / QUARTET) - 1u, 0,
		   (bool)last, false, true, false, SCP_B_LINE, SCP_A_LINE, OPCODE_INT8, (bool)first);
}

/* Fused int32-accumulator -> uint8 requant in FREGs: INT32_TO_FP32 -> MUL_COL(per-OC scale) ->
 * FP32_TO_INT32(rne) -> SATUINT8 -> PACK_128B. SATUINT8 clamps [0,255]; the ReLU6 ceiling is folded
 * because S_out=6/255 makes the quantized value of 6 == 255 (see gen_yolo_int8.py). ReLU6 floor = the
 * [0] clamp. transf0 executes first (transform args are transf9-first, PROMPT_0 s9). */
static inline void requant_u8(uint32_t scp_scale, uint32_t oc)
{
	tensor_quant(0, (P / QUARTET) - 1u, oc - 1u, scp_scale,
		     QUANT_LAST_TRANS, QUANT_LAST_TRANS, QUANT_LAST_TRANS,
		     QUANT_LAST_TRANS, QUANT_LAST_TRANS,
		     QUANT_PACK_128B, QUANT_SATUINT8, QUANT_FP32_TO_INT32,
		     QUANT_FP32_MUL_COL, QUANT_INT32_TO_FP32);
}

/* Store packed uint8 [OC][P] (rows in FREG 0,2,4,...; one 16B block/row; P byte stride). */
static inline void store_u8_tile(uint8_t *dst, uint32_t oc)
{
	tensor_store(1, 0, 0, oc - 1u, (uint64_t)dst, 0, P);
}

#ifndef YOLO_TRUNK
/* One output tile at spatial (y,x0): fold TAPS into NGRP dispatches -> requant -> uint8 [OC][P],
 * scattered into the (unpadded) output NHWC. B1: pack all groups' B once, one evict per tile.
 * Every dim is a compile-time config constant so the loops stay unrolled and the pack word-copy
 * keeps constant strides. At K=1 the (ky,c) loops run once (gidx=0, nt=1). */
static void conv_tile(const uint8_t *restrict pad, const int8_t *restrict aw,
		      int8_t *restrict bpk, const float *restrict mvec,
		      uint8_t *restrict temp, uint8_t *restrict out,
		      uint32_t y, uint32_t x0)
{
	uint32_t off = 0;
	for (uint32_t ky = 0; ky < K; ky++)
		for (uint32_t c = 0; c < K; c += FOLD_TAPS) {
			const uint32_t nt = NT_AT(c);
			pack_b_group(bpk + off, pad, y, x0, (int)ky - (int)(K / 2u), c, nt);
			off += nt * (CH / QUARTET) * ROW_STRIDE_BYTES;
		}
	FENCE; evict(bpk, off); WAIT_CACHEOPS;

	for (uint32_t oc0 = 0; oc0 < CH; oc0 += OC_TILE) {
		const uint32_t ntile = (CH - oc0 < OC_TILE) ? (CH - oc0) : OC_TILE;
		uint32_t boff = 0, gidx = 0;
		for (uint32_t ky = 0; ky < K; ky++)
			for (uint32_t c = 0; c < K; c += FOLD_TAPS) {
				const uint32_t nt = NT_AT(c);
				const uint32_t aq = nt * (CH / QUARTET);
				tensor_load(false, false, SCP_A_LINE, 0, 0,
					    (uint64_t)(aw + ((uint64_t)gidx * CH + oc0) * ROW_STRIDE_BYTES), 0,
					    ntile - 1u, ROW_STRIDE_BYTES, 0);
				tensor_load(false, false, SCP_B_LINE, 0, 0,
					    (uint64_t)(bpk + boff), 0,
					    aq - 1u, ROW_STRIDE_BYTES, 0);
				tensor_wait(TENSOR_LOAD_WAIT_0);

				fma_group(gidx == 0u, gidx == NGRP - 1u, nt, ntile);
				tensor_wait(TENSOR_FMA_WAIT);
				boff += aq * ROW_STRIDE_BYTES;
				gidx++;
			}

		tensor_load(false, false, SCP_SCALE_LINE, 0, 0, (uint64_t)(mvec + oc0), 0,
			    0u, ROW_STRIDE_BYTES, 0);
		tensor_wait(TENSOR_LOAD_WAIT_0);
		requant_u8(SCP_SCALE_LINE, ntile);  tensor_wait(TENSOR_QUANT_WAIT);
		store_u8_tile(temp, ntile);         tensor_wait(TENSOR_STORE_WAIT);
		FREG_CLOBBER_BARRIER();
		FENCE; evict(temp, ntile * P); WAIT_CACHEOPS;

		for (uint32_t oc = 0; oc < ntile; oc++)
			for (uint32_t j = 0; j < P; j++)
				out[((y * IMG_W) + (x0 + j)) * CH + oc0 + oc] = temp[oc * P + j];
	}
}

#endif  /* !YOLO_TRUNK (single-conv conv_tile) */

/* Replicate-fill the HALO ring (no-op at K=1). Matches the microbench conv3's clamp_coord (edge). */
static void fill_halo_band(uint8_t *pad, uint32_t row0, uint32_t row1)
{
	if (HALO == 0u) { (void)pad; (void)row0; (void)row1; return; }
	for (uint32_t y = row0; y < row1; y++)
		for (uint32_t ic = 0; ic < CH; ic++) {
			PAD_AT(pad, (int)y, -1)[ic]         = PAD_AT(pad, (int)y, 0)[ic];
			PAD_AT(pad, (int)y, (int)IMG_W)[ic] = PAD_AT(pad, (int)y, (int)IMG_W - 1)[ic];
		}
	if (row0 == 0u)
		for (int x = -1; x <= (int)IMG_W; x++)
			for (uint32_t ic = 0; ic < CH; ic++)
				PAD_AT(pad, -1, x)[ic] = PAD_AT(pad, 0, x < 0 ? 0 : (x >= (int)IMG_W ? (int)IMG_W - 1 : x))[ic];
	if (row1 == IMG_H)
		for (int x = -1; x <= (int)IMG_W; x++)
			for (uint32_t ic = 0; ic < CH; ic++)
				PAD_AT(pad, (int)IMG_H, x)[ic] = PAD_AT(pad, (int)IMG_H - 1, x < 0 ? 0 : (x >= (int)IMG_W ? (int)IMG_W - 1 : x))[ic];
}

static uint32_t stripe_checksum(const uint8_t *out, uint32_t row0, uint32_t row1)
{
	uint32_t sum = 0;
	for (uint32_t y = row0; y < row1; y++)
		for (uint32_t x = 0; x < IMG_W; x++)
			for (uint32_t c = 0; c < CH; c++)
				sum += out[(y * IMG_W + x) * CH + c];
	return sum;
}

#ifdef YOLO_TRUNK
/* ================= M-i2 full-network trunk (config-driven mixed precision) =================== */
/* NET[] mirrors yolo_vpu_argbuf.c main: 4 blocks of {conv3x3 -> conv1x1} then head1x1. Precision
 * split (mirror DnCNN): FP32 first conv (block-0 conv3, input stem) + FP32 head; int8 the 7 hidden
 * convs between them. Hot dims (K,P,CH,PADW) stay compile-time literals per tile (no runtime dim). */
enum layer_kind { FP32_FIRST, CONV1_INT8, CONV3_INT8, FP32_HEAD };
struct layer_cfg { uint8_t k; uint8_t kind; };
static const struct layer_cfg NET[] = {
	{ 3u, FP32_FIRST },   /* L0 block0 conv3 (FP32 input stem)   */
	{ 1u, CONV1_INT8 },   /* L1 block0 conv1                     */
	{ 3u, CONV3_INT8 },   /* L2 block1 conv3                     */
	{ 1u, CONV1_INT8 },   /* L3 block1 conv1                     */
	{ 3u, CONV3_INT8 },   /* L4 block2 conv3                     */
	{ 1u, CONV1_INT8 },   /* L5 block2 conv1                     */
	{ 3u, CONV3_INT8 },   /* L6 block3 conv3                     */
	{ 1u, CONV1_INT8 },   /* L7 block3 conv1                     */
	{ 1u, FP32_HEAD  },   /* L8 head1x1 (+128)                   */
};
#define NLAYERS (sizeof(NET) / sizeof(NET[0]))
_Static_assert(NLAYERS == 9u, "trunk = 4 blocks x {conv3,conv1} + head");

/* microbench FP32 layer scales (compile-time; the int8 LAYER_SCALEs are folded into YOLO_REQUANT[]) */
#define CONV3_SCALE (1.0f / 256.0f)
#define CONV1_SCALE (1.0f / 128.0f)   /* (unused in trunk FP32 path; folded into requant) */
#define HEAD_SCALE  (1.0f /  64.0f)

static inline int clamp_coord(int v, unsigned limit)
{
	if (v < 0) return 0;
	if (v >= (int)limit) return (int)limit - 1;
	return v;
}
/* rne == np.rint (first-conv quantize); round-half-up == floor(v+0.5) (head, mirrors microbench). */
static inline int32_t rint_rne(float v)
{
	int32_t r; __asm__("fcvt.w.s %0, %1, rne" : "=r"(r) : "f"(v)); return r;
}
static inline int32_t round_half_up(float v)
{
	int32_t r; const float vv = v + 0.5f;
	__asm__("fcvt.w.s %0, %1, rdn" : "=r"(r) : "f"(vv)); return r;
}
static inline float relu6f(float v)
{
	if (v < 0.0f) return 0.0f;
	if (v > 6.0f) return 6.0f;
	return v;
}

/* FP32 first conv (block-0 conv3): scalar accumulation in EXACT (ky,kx,ic) order so the numpy oracle
 * reproduces it bit-for-bit (needs -ffp-contract=off so mul+add is not fused into fmadd). Replicate
 * (clamp_coord) edges, ReLU6, then quantize to uint8 by *YOLO_QUANT0 (=42.5, no fdiv) -> out_pad. */
static void conv_first_fp32(const float *restrict input, const float *restrict w0,
			    uint8_t *restrict out_pad, uint32_t row0, uint32_t row1)
{
	for (uint32_t oc = 0; oc < CH; oc++) {
		const float *const w_oc = w0 + oc * (K * K * CH);
		for (uint32_t y = row0; y < row1; y++)
			for (uint32_t x = 0; x < IMG_W; x++) {
				float acc = 0.0f;
				for (int ky = -1; ky <= 1; ky++) {
					const int yy = clamp_coord((int)y + ky, IMG_H);
					for (int kx = -1; kx <= 1; kx++) {
						const int xx = clamp_coord((int)x + kx, IMG_W);
						const float *const pix = input + ((uint32_t)yy * IMG_W + (uint32_t)xx) * CH;
						const float *const w = w_oc + (uint32_t)((ky + 1) * 3 + (kx + 1)) * CH;
						for (uint32_t ic = 0; ic < CH; ic++)
							acc += pix[ic] * w[ic];
					}
				}
				const float o = relu6f(acc * CONV3_SCALE);
				int32_t q = rint_rne(o * YOLO_QUANT0);
				if (q < 0) q = 0; if (q > 255) q = 255;
				PAD_AT(out_pad, (int)y, (int)x)[oc] = (uint8_t)q;
			}
	}
}

/* FP32 head1x1: dequant the last int8 activation (uint8 * YOLO_DEQUANT), scalar dot over ic, +128,
 * *HEAD_SCALE, round-half-up, clamp -> uint8 NHWC output. Same (ic) order as the numpy oracle. */
static void head_fp32(const uint8_t *restrict src_pad, const float *restrict wh,
		      uint8_t *restrict out, uint32_t row0, uint32_t row1)
{
	for (uint32_t oc = 0; oc < CH; oc++) {
		const float *const w = wh + oc * CH;
		for (uint32_t y = row0; y < row1; y++)
			for (uint32_t x = 0; x < IMG_W; x++) {
				const uint8_t *const pix = PAD_AT(src_pad, (int)y, (int)x);
				float acc = 0.0f;
				for (uint32_t ic = 0; ic < CH; ic++) {
					const float xv = (float)pix[ic] * YOLO_DEQUANT;
					acc += xv * w[ic];
				}
				const float v = 128.0f + acc * HEAD_SCALE;
				int32_t q = round_half_up(v);
				if (q < 0) q = 0; if (q > 255) q = 255;
				out[(y * IMG_W + x) * CH + oc] = (uint8_t)q;
			}
	}
}

/* int8 conv1x1 tile (K=1): center tap (kx0=1 -> col x0 since K/2=1), NGRP=1. Reuses the shared
 * pack_b_group / fma_group / requant_u8 / store_u8_tile primitives; writes uint8 into dst interior. */
static void conv1_int8_tile(const uint8_t *restrict pad, const int8_t *restrict aw,
			    int8_t *restrict bpk, const float *restrict mvec,
			    uint8_t *restrict temp, uint8_t *restrict dst, uint32_t y, uint32_t x0)
{
	pack_b_group(bpk, pad, y, x0, 0, 1u, 1u);
	FENCE; evict(bpk, (CH / QUARTET) * ROW_STRIDE_BYTES); WAIT_CACHEOPS;

	tensor_load(false, false, SCP_A_LINE, 0, 0, (uint64_t)aw, 0, CH - 1u, ROW_STRIDE_BYTES, 0);
	tensor_load(false, false, SCP_B_LINE, 0, 0, (uint64_t)bpk, 0, (CH / QUARTET) - 1u, ROW_STRIDE_BYTES, 0);
	tensor_wait(TENSOR_LOAD_WAIT_0);
	fma_group(1, 1, 1u, CH);
	tensor_wait(TENSOR_FMA_WAIT);

	tensor_load(false, false, SCP_SCALE_LINE, 0, 0, (uint64_t)mvec, 0, 0u, ROW_STRIDE_BYTES, 0);
	tensor_wait(TENSOR_LOAD_WAIT_0);
	requant_u8(SCP_SCALE_LINE, CH);  tensor_wait(TENSOR_QUANT_WAIT);
	store_u8_tile(temp, CH);         tensor_wait(TENSOR_STORE_WAIT);
	FREG_CLOBBER_BARRIER();
	FENCE; evict(temp, CH * P); WAIT_CACHEOPS;

	for (uint32_t oc = 0; oc < CH; oc++)
		for (uint32_t j = 0; j < P; j++)
			PAD_AT(dst, (int)y, (int)(x0 + j))[oc] = temp[oc * P + j];
}

/* int8 conv3x3 tile (K=3, replicate-pad): 3 groups (one kernel row each, 3 folded column taps).
 * Identical dispatch to the M-i1 conv_tile but scatters into the padded dst interior. */
static void conv3_int8_tile(const uint8_t *restrict pad, const int8_t *restrict aw,
			    int8_t *restrict bpk, const float *restrict mvec,
			    uint8_t *restrict temp, uint8_t *restrict dst, uint32_t y, uint32_t x0)
{
	uint32_t off = 0;
	for (uint32_t ky = 0; ky < 3u; ky++) {
		pack_b_group(bpk + off, pad, y, x0, (int)ky - 1, 0u, 3u);   /* dy=ky-1, kx0=0, n_taps=3 */
		off += 3u * (CH / QUARTET) * ROW_STRIDE_BYTES;
	}
	FENCE; evict(bpk, off); WAIT_CACHEOPS;

	uint32_t boff = 0;
	for (uint32_t gidx = 0; gidx < 3u; gidx++) {
		tensor_load(false, false, SCP_A_LINE, 0, 0,
			    (uint64_t)(aw + (uint64_t)gidx * CH * ROW_STRIDE_BYTES), 0,
			    CH - 1u, ROW_STRIDE_BYTES, 0);
		tensor_load(false, false, SCP_B_LINE, 0, 0, (uint64_t)(bpk + boff), 0,
			    3u * (CH / QUARTET) - 1u, ROW_STRIDE_BYTES, 0);
		tensor_wait(TENSOR_LOAD_WAIT_0);
		fma_group(gidx == 0u, gidx == 2u, 3u, CH);
		tensor_wait(TENSOR_FMA_WAIT);
		boff += 3u * (CH / QUARTET) * ROW_STRIDE_BYTES;
	}

	tensor_load(false, false, SCP_SCALE_LINE, 0, 0, (uint64_t)mvec, 0, 0u, ROW_STRIDE_BYTES, 0);
	tensor_wait(TENSOR_LOAD_WAIT_0);
	requant_u8(SCP_SCALE_LINE, CH);  tensor_wait(TENSOR_QUANT_WAIT);
	store_u8_tile(temp, CH);         tensor_wait(TENSOR_STORE_WAIT);
	FREG_CLOBBER_BARRIER();
	FENCE; evict(temp, CH * P); WAIT_CACHEOPS;

	for (uint32_t oc = 0; oc < CH; oc++)
		for (uint32_t j = 0; j < P; j++)
			PAD_AT(dst, (int)y, (int)(x0 + j))[oc] = temp[oc * P + j];
}

#ifdef YOLO_TRUNK_PROBE
/* Copy a band's padded-interior activation into a plain NHWC probe buffer (a discriminating gate:
 * L1/L2 vary and diverge under a wrong scale/weight/pack, unlike the all-128 collapsed final out). */
static void probe_copy(const uint8_t *restrict src_pad, uint8_t *restrict probe,
		       uint32_t row0, uint32_t row1)
{
	for (uint32_t y = row0; y < row1; y++)
		for (uint32_t x = 0; x < IMG_W; x++) {
			const uint8_t *const pix = PAD_AT(src_pad, (int)y, (int)x);
			for (uint32_t c = 0; c < CH; c++)
				probe[(y * IMG_W + x) * CH + c] = pix[c];
		}
}
#endif

/* On-device int8 weight quant + pack (self-contained: no file-loaded blob, mirror the FP32 boundary
 * weights). Each hidden conv's FP32 weights come from init_model's flat formula; quantize by MULTIPLY
 * (qi=clip(rint(w*INV_SW),-127,127), INV_SW=127/max|w| baked -- no fdiv) and pack into the group-major
 * A-blob layout the int8 tiles load. Bit-identical to gen_yolo_int8.py's file-loaded blob (asserted in
 * the generator: div-quant == mul-quant; the packed blob matched byte-for-byte). */
static const struct { uint16_t k; uint16_t flat_base; uint16_t aw_off; } INT8W[YOLO_TRUNK_LAYERS] = {
	{ 1u, 2304u,     0u },   /* L1 block0 conv1 */
	{ 3u, 2560u,  1024u },   /* L2 block1 conv3 */
	{ 1u, 4864u,  4096u },   /* L3 block1 conv1 */
	{ 3u, 5120u,  5120u },   /* L4 block2 conv3 */
	{ 1u, 7424u,  8192u },   /* L5 block2 conv1 */
	{ 3u, 7680u,  9216u },   /* L6 block3 conv3 */
	{ 1u, 9984u, 12288u },   /* L7 block3 conv1 */
};
static inline int8_t quant_w_dev(float w, float inv_sw)
{
	int32_t q = rint_rne(w * inv_sw);
	if (q < -127) q = -127; if (q > 127) q = 127;
	return (int8_t)q;
}
static void build_int8_weights(int8_t *restrict aw)
{
	for (uint32_t li = 0; li < YOLO_TRUNK_LAYERS; li++) {
		const uint32_t k  = INT8W[li].k;
		const uint32_t fb = INT8W[li].flat_base;
		const float    iv = YOLO_INV_SW[li];
		int8_t *const blk = aw + INT8W[li].aw_off;
		if (k == 1u) {                                   /* one group: line=oc, byte=ic */
			for (uint32_t oc = 0; oc < CH; oc++)
				for (uint32_t ic = 0; ic < CH; ic++) {
					const uint32_t fi = fb + oc * CH + ic;
					const float w = (float)((int32_t)((fi * 29u + 7u) % 31u) - 15);
					blk[oc * ROW_STRIDE_BYTES + ic] = quant_w_dev(w, iv);
				}
		} else {                                         /* 3 groups (ky), fold 3 col taps: byte ti*CH+ic */
			for (uint32_t oc = 0; oc < CH; oc++)
				for (uint32_t ky = 0; ky < 3u; ky++)
					for (uint32_t ti = 0; ti < 3u; ti++) {
						const uint32_t tap = ky * 3u + ti;
						for (uint32_t ic = 0; ic < CH; ic++) {
							const uint32_t fi = fb + oc * (3u * 3u * CH) + tap * CH + ic;
							const float w = (float)((int32_t)((fi * 29u + 7u) % 31u) - 15);
							blk[(ky * CH + oc) * ROW_STRIDE_BYTES + ti * CH + ic] =
								quant_w_dev(w, iv);
						}
					}
		}
	}
}

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = bench_hart_id();
	if (!bench_hart_enabled(hart_id)) {
		return 0;
	}

	uint8_t *const base    = (uint8_t *)buffer_base_from_args(arg_area);
	float   *const input   = (float *)(base + INPUT_OFFSET);            /* device-built FP32 NHWC   */
	int8_t  *const aw      = (int8_t *)(base + WEIGHTS_OFFSET);         /* device-built int8 A-blob */
	float   *const fpw     = (float *)(base + FPW_OFFSET);              /* device-built FP32 weights*/
	float   *const fpw_l0  = fpw;                                       /* L0 conv3 W0 (2304 f)     */
	float   *const fpw_hd  = fpw + CH * K * K * CH;                     /* head W (256 f)           */
	float   *const mvec    = (float *)(base + MVEC_OFFSET);             /* per-OC requant scale     */
	int8_t  *const bpk     = (int8_t *)(base + BPACK_OFFSET + hart_id * BPK_STRIDE);
	uint8_t *const temp    = base + TEMP_OFFSET + hart_id * TEMP_STRIDE;
	uint8_t *const actA    = base + ACTA_OFFSET;
	uint8_t *const actB    = base + ACTB_OFFSET;
	uint8_t *const out     = base + OUTPUT_OFFSET;
	volatile struct yolo_slot *const slots =
		(volatile struct yolo_slot *)(base + SLOTS_OFFSET);
	volatile struct yolo_summary *const summary =
		(volatile struct yolo_summary *)(base + SUMMARY_OFFSET);
	g_barrier = (volatile struct bench_barrier_state *)(base + BARRIER_OFFSET);

	const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

	if (hart_id == 0u) {
		for (uint32_t i = 0; i < IMG_H * IMG_W * CH; i++)     /* init_model input formula  */
			input[i] = (float)((i * 17u + 23u) & 0xffu) * (1.0f / 255.0f);
		for (uint32_t j = 0; j < CH * K * K * CH; j++)        /* init_model block0 conv3 (flat 0..) */
			fpw_l0[j] = (float)((int32_t)((j * 29u + 7u) % 31u) - 15);
		for (uint32_t m = 0; m < CH * CH; m++)                /* init_model head (flat HEAD_FLAT_BASE..) */
			fpw_hd[m] = (float)((int32_t)(((HEAD_FLAT_BASE + m) * 29u + 7u) % 31u) - 15);
		build_int8_weights(aw);                              /* 7 hidden convs: init_model -> int8 A-blob */
		FREG_CLOBBER_BARRIER();                              /* rint_rne uses an FP reg; spill before int8 */
		FENCE;
		evict(input, INPUT_BYTES);
		evict(fpw, FPW_BYTES);
		evict(aw, AW_BYTES);
		WAIT_CACHEOPS;
	}
	bench_barrier();

#ifdef DNCNN_PMC
	pmc_probe_begin(base + PMC_OFFSET, hart_id, ACTIVE_HARTS);   /* bracket the network compute (excl. 1-time weight build) */
#endif

	/* ---- L0: FP32 first conv (input stem) -> ReLU6 -> quantize -> actA interior ---- */
	FREG_CLOBBER_BARRIER();
	conv_first_fp32(input, fpw_l0, actA, row0, row1);
	FREG_CLOBBER_BARRIER();
	FENCE;
	/* interior rows [row0,row1) live at padded rows [row0+HALO,row1+HALO); every padded row is a
	 * whole # of 64B lines (PADW*CH=1344=21 lines) so band boundaries are line-aligned (seam-clean). */
	evict(actA + (uint64_t)(row0 + HALO) * PADW * CH, (uint64_t)(row1 - row0) * PADW * CH);
	WAIT_CACHEOPS;
	bench_barrier();

	uint8_t *src = actA;
	uint8_t *dst = actB;
	uint32_t li = 0;              /* int8 hidden-layer index -> YOLO_REQUANT[li] */
	uint32_t aw_off = 0;         /* byte offset into the concatenated A-blob     */

	for (uint32_t layer = 1u; layer < NLAYERS - 1u; layer++) {   /* L1 .. L7 (int8 hidden) */
		const uint32_t k = NET[layer].k;

		if (k == 3u) {   /* replicate-pad halo needs the src interior complete + neighbour rows fresh */
			fill_halo_band(src, row0, row1);
			FENCE;
			evict(src, PAD_BYTES);   /* writeback+invalidate whole src (1-hart; M-i3 refines seams) */
			WAIT_CACHEOPS;
			bench_barrier();
		}

		if (hart_id == 0u) {
			for (uint32_t oc = 0; oc < CH; oc++)
				mvec[oc] = YOLO_REQUANT[li];
			FENCE; evict(mvec, MVEC_BYTES); WAIT_CACHEOPS;
		}
		bench_barrier();

		for (uint32_t ty = row0; ty < row1; ty++)
			for (uint32_t tx = 0; tx < IMG_W / P; tx++) {
				if (k == 1u)
					conv1_int8_tile(src, aw + aw_off, bpk, mvec, temp, dst, ty, tx * P);
				else
					conv3_int8_tile(src, aw + aw_off, bpk, mvec, temp, dst, ty, tx * P);
			}

		FENCE;
		evict(dst + (uint64_t)(row0 + HALO) * PADW * CH, (uint64_t)(row1 - row0) * PADW * CH);
		WAIT_CACHEOPS;
		bench_barrier();

#ifdef YOLO_TRUNK_PROBE
		if (li == 0u) { probe_copy(dst, base + PROBE1_OFFSET, row0, row1);
			FENCE; evict(base + PROBE1_OFFSET + (uint64_t)row0 * IMG_W * CH,
				     (uint64_t)(row1 - row0) * IMG_W * CH); WAIT_CACHEOPS; }
		if (li == 1u) { probe_copy(dst, base + PROBE2_OFFSET, row0, row1);
			FENCE; evict(base + PROBE2_OFFSET + (uint64_t)row0 * IMG_W * CH,
				     (uint64_t)(row1 - row0) * IMG_W * CH); WAIT_CACHEOPS; }
#endif

		aw_off += (k == 1u) ? AW_CONV1_BYTES : AW_CONV3_BYTES;
		li++;
		{ uint8_t *t = src; src = dst; dst = t; }   /* ping-pong */
	}

	/* ---- L8: FP32 head1x1 over the last int8 activation (dequant + +128) -> uint8 NHWC out ---- */
	FREG_CLOBBER_BARRIER();
	head_fp32(src, fpw_hd, out, row0, row1);
	FREG_CLOBBER_BARRIER();
	FENCE;
	evict(out + (uint64_t)row0 * IMG_W * CH, (uint64_t)(row1 - row0) * IMG_W * CH);
	WAIT_CACHEOPS;
	bench_barrier();

#ifdef DNCNN_PMC
	pmc_probe_end(base + PMC_OFFSET, hart_id);   /* close the network-compute bracket before attestation */
#endif

	volatile struct yolo_slot *const my_slot = &slots[hart_id];
	my_slot->magic        = YOLO_MAGIC;
	my_slot->hart_id      = hart_id;
	my_slot->minion_id    = get_minion_id();
	my_slot->thread_id    = get_thread_id();
	my_slot->row0         = row0;
	my_slot->row1         = row1;
	my_slot->active_harts = ACTIVE_HARTS;
	my_slot->checksum     = stripe_checksum(out, row0, row1);
	my_slot->done         = 1u;
	FENCE;
	evict((const void *)my_slot, sizeof(*my_slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0u, done_count = 0u, slot_checksum_sum = 0u, output_sum = 0u;
		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == YOLO_MAGIC && slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}
		for (uint32_t i = 0; i < OUT_BYTES; i++)
			output_sum += out[i];

		summary->magic             = YOLO_MAGIC;
		summary->active_harts      = ACTIVE_HARTS;
		summary->passes            = 1u;
		summary->width             = IMG_W;
		summary->height            = IMG_H;
		summary->channels          = CH;
		summary->blocks            = 4u;
		summary->active_mask       = active_mask;
		summary->done_count        = done_count;
		summary->output_sum        = output_sum;
		summary->slot_checksum_sum = slot_checksum_sum;
		summary->ops_lo            = 0u;
		summary->ops_hi            = 0u;
		summary->head_channels     = CH;
		FENCE;
		evict((const void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}
	return 0;
}

#else  /* !YOLO_TRUNK : the M-i0/M-i1 single int8 conv gate (unchanged) */

int main(uintptr_t arg_area)
{
	const uint32_t hart_id = bench_hart_id();
	if (!bench_hart_enabled(hart_id)) {
		return 0;
	}

	uint8_t *const base   = (uint8_t *)buffer_base_from_args(arg_area);
	uint8_t *const act    = base + ACT_OFFSET;                 /* file-loaded uint8 NHWC activation  */
	const int8_t *const aw = (const int8_t *)(base + WEIGHTS_OFFSET);  /* file-loaded int8 A-blob     */
	float   *const mvec   = (float *)(base + MVEC_OFFSET);     /* shared per-OC requant scale         */
	int8_t  *const bpk    = (int8_t *)(base + BPACK_OFFSET + hart_id * BPK_STRIDE);   /* per-hart      */
	uint8_t *const temp   = base + TEMP_OFFSET + hart_id * TEMP_STRIDE;               /* per-hart      */
	uint8_t *const out    = base + OUTPUT_OFFSET;
	volatile struct yolo_slot *const slots =
		(volatile struct yolo_slot *)(base + SLOTS_OFFSET);
	volatile struct yolo_summary *const summary =
		(volatile struct yolo_summary *)(base + SUMMARY_OFFSET);
	g_barrier = (volatile struct bench_barrier_state *)(base + BARRIER_OFFSET);

	const uint32_t row0 = (IMG_H * hart_id) / ACTIVE_HARTS;
	const uint32_t row1 = (IMG_H * (hart_id + 1u)) / ACTIVE_HARTS;

	if (hart_id == 0u) {
		for (uint32_t oc = 0; oc < CH; oc++)   /* per-OC requant scale (per-tensor here => uniform) */
			mvec[oc] = YOLO_REQUANT[0];
		fill_halo_band(act, 0u, IMG_H);        /* no-op at K=1 (HALO=0)                              */
		FENCE;
		evict(mvec, MVEC_BYTES);
		evict(act, PAD_BYTES);
		WAIT_CACHEOPS;
	}
	bench_barrier();

	/* Invalidate this hart's read window so it reads fresh neighbours from DRAM, then tile the band. */
	FENCE;
	evict(act + (uint64_t)row0 * PADW * CH, (uint64_t)(row1 - row0) * PADW * CH);
	WAIT_CACHEOPS;
	for (uint32_t ty = row0; ty < row1; ty++)
		for (uint32_t tx = 0; tx < IMG_W / P; tx++)
			conv_tile(act, aw, bpk, mvec, temp, out, ty, tx * P);

	FENCE;
	evict(out + (uint64_t)row0 * IMG_W * CH, (uint64_t)(row1 - row0) * IMG_W * CH);
	WAIT_CACHEOPS;
	bench_barrier();

	volatile struct yolo_slot *const my_slot = &slots[hart_id];
	my_slot->magic        = YOLO_MAGIC;
	my_slot->hart_id      = hart_id;
	my_slot->minion_id    = get_minion_id();
	my_slot->thread_id    = get_thread_id();
	my_slot->row0         = row0;
	my_slot->row1         = row1;
	my_slot->active_harts = ACTIVE_HARTS;
	my_slot->checksum     = stripe_checksum(out, row0, row1);
	my_slot->done         = 1u;
	FENCE;
	evict((const void *)my_slot, sizeof(*my_slot));
	WAIT_CACHEOPS;
	bench_barrier();

	if (hart_id == 0u) {
		uint32_t active_mask = 0u, done_count = 0u, slot_checksum_sum = 0u, output_sum = 0u;
		for (uint32_t h = 0; h < ACTIVE_HARTS; h++) {
			if (slots[h].magic == YOLO_MAGIC && slots[h].done == 1u) {
				done_count++;
				active_mask |= 1u << slots[h].hart_id;
				slot_checksum_sum += slots[h].checksum;
			}
		}
		for (uint32_t i = 0; i < OUT_BYTES; i++)
			output_sum += out[i];

		const uint64_t macs = (uint64_t)IMG_W * IMG_H * CH * CH * TAPS;
		const uint64_t ops = macs * 2u;
		summary->magic             = YOLO_MAGIC;
		summary->active_harts      = ACTIVE_HARTS;
		summary->passes            = 1u;
		summary->width             = IMG_W;
		summary->height            = IMG_H;
		summary->channels          = CH;
		summary->blocks            = 1u;
		summary->active_mask       = active_mask;
		summary->done_count        = done_count;
		summary->output_sum        = output_sum;
		summary->slot_checksum_sum = slot_checksum_sum;
		summary->ops_lo            = (uint32_t)ops;
		summary->ops_hi            = (uint32_t)(ops >> 32);
		summary->head_channels     = CH;
		FENCE;
		evict((const void *)summary, sizeof(*summary));
		WAIT_CACHEOPS;
	}

	return 0;
}

#endif  /* YOLO_TRUNK vs single-conv main */
