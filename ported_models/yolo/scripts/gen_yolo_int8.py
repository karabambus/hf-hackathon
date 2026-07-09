"""Generate the INT8 conv reference + scales + blobs for the YOLO int8-TFMA port (M-i0 1x1, M-i1 3x3).

The YOLO microbench (ported_models/yolo/src/yolo_vpu_argbuf.c) has deterministic weights and
input (init_model): input[i] = ((i*17+23)&0xff)/255, weights[i] = ((i*29+7)%31)-15. This script
reproduces those EXACTLY and quantizes ONE hidden conv layer into the int8 TFMA path the kernel runs:

    uint8 activation (B, unsigned) x int8 weight (A, signed) -> int32 acc -> requant -> uint8

Op-class is chosen by the CLI arg K (default 1):
  * K=1 -> the microbench conv1x1 shape (IC=OC=CH=16, ReLU6) with scale CONV1_SCALE  [M-i0]
  * K=3 -> the microbench conv3x3 stride-1 pad-1 REPLICATE-pad shape with scale CONV3_SCALE  [M-i1]

Quant scheme (all scales are compile-time constants baked into the header; NO runtime fdiv):
  * Activations are ReLU6 outputs in [0,6], so BOTH the input and output activation use the ReLU6
    scale S = 6/255 (uint8 0..255 spans exactly [0,6]). With S_out = 6/255 the "quantized value of
    6" is exactly 255, so the tensor engine's SATUINT8 [0,255] clamp IS the ReLU6 ceiling -- the
    ReLU6 [0,6] clamp is folded into the requant for free (see optimizations.md).
  * Weights: per-tensor symmetric int8, S_w = max|w| / 127.
  * Folded requant multiplier M = S_in * S_w * LAYER_SCALE / S_out = S_w * LAYER_SCALE (S_in=S_out).
    The kernel's requant is  clip(rint(acc_i32.f32 * M), 0, 255)  -- identical math here.

Replicate-pad (K=3): the microbench conv3 pads by EDGE replication (clamp_coord). The oracle uses
np.pad(mode='edge') -- mathematically identical to the kernel's fill_halo_band replicate ring -- so
the edge/seam rows match max_abs=0. The activation blob is emitted PRE-padded to the kernel's padded
[PADH][PADW][CH] layout with ONLY the interior filled; the halo ring is left zero and the kernel's
fill_halo_band re-derives it on device (so the gate actually exercises the replicate-pad).

Emits (into ported_models/yolo/src/ for the committed header, and the bench dir for the blobs):
  - ported_models/yolo/src/yolo_int8_scales.h : YOLO_QUANT0 / YOLO_REQUANT[] / YOLO_DEQUANT /
                                                YOLO_RELU6_HI  (mirror of DNCNN_*)
  - <bench>/yolo_int8_input.bin   : uint8 [PADH][PADW][CH] NHWC  (the B activation, file-loaded @0x2000)
  - <bench>/yolo_int8_weights.bin : int8 A-blob, group-major 64B lines (file-loaded @0x20000)
  - <bench>/yolo_reference_int8.npy: uint8 [IMG_H][IMG_W][CH] the kernel must reproduce (max_abs=0)

Also prints the int8-vs-FP32 quality delta (informational; not a max_abs gate).

Run: local-artifacts/torch-venv/bin/python ported_models/yolo/scripts/gen_yolo_int8.py [K]
"""
import os
import sys
import numpy as np

# ---- microbench shape (mirror yolo_vpu_argbuf.c) ----
IMG_W, IMG_H, CH = 80, 80, 16
ARG1 = sys.argv[1] if len(sys.argv) > 1 else "1"
TRUNK = (ARG1 == "trunk")                          # 'trunk' => M-i2 full-network mixed-precision
K = 3 if TRUNK else int(ARG1)                      # 1 => M-i0 1x1 ; 3 => M-i1 3x3 ; trunk uses K=3 geometry
assert K in (1, 3), "K must be 1 (conv1x1), 3 (conv3x3), or 'trunk' (full network)"
QUARTET = 4
ROW_STRIDE = 64                        # one L1 scratchpad line
CONV3_SCALE = np.float32(1.0 / 256.0)  # microbench CONV3_SCALE (block-0 conv3 shape)
CONV1_SCALE = np.float32(1.0 / 128.0)  # microbench CONV1_SCALE (block conv1 shape)
HEAD_SCALE  = np.float32(1.0 / 64.0)   # microbench HEAD_SCALE (head1x1, +128)
RELU6 = np.float32(6.0)
# trunk quant constants: QUANT0=255/6=42.5 EXACT (multiply, no fdiv -> ceiling 6*42.5==255 exact);
# DEQUANT = 1/42.5 (float32) for the FP32 head. S_in=S_out=1/42.5 cancel in every int8 requant M.
QUANT0 = np.float32(255.0 / 6.0)       # == 42.5 exactly
DEQUANT_T = np.float32(1.0 / 42.5)     # head uint8 -> FP32
YOLO_BLOCKS = 4
BLOCK_WEIGHTS = CH * 3 * 3 * CH + CH * CH   # 2560: conv3 tile (2304) + conv1 tile (256)

# ---- adaptive tap-fold (mirror the kernel's FOLD_* macros exactly) ----
HALO = (K - 1) // 2
PADW_MULT = 64 // CH                              # =4 at CH=16
PADW = ((IMG_W + 2 * HALO + PADW_MULT - 1) // PADW_MULT) * PADW_MULT
PADH = IMG_H + 2 * HALO
FOLD_BUDGET = 16 // (CH // QUARTET)               # =4 at CH=16 (acols <=16 quartets)
FOLD_TAPS = min(K, FOLD_BUDGET)                   # =1 at K=1, =3 at K=3/CH=16
CHUNKS_PER_ROW = (K + FOLD_TAPS - 1) // FOLD_TAPS  # =1 for K in {1,3}
NGRP = K * CHUNKS_PER_ROW                          # FMA dispatches/tile: 1 at K=1, 3 at K=3
LAYER_SCALE = CONV1_SCALE if K == 1 else CONV3_SCALE

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(HERE, "..", "src")
BENCH = os.path.join(REPO, "local-artifacts", "erbium_amp_probe", "yolo-bench")

# ReLU6 activation scale: uint8 0..255 spans [0,6] exactly => S_out=6/255 makes quantized-6 == 255.
S_ACT = np.float32(RELU6 / np.float32(255.0))


def init_model_input():
    """Exactly yolo_vpu_argbuf.c init_model's input: input[i]=((i*17+23)&0xff)/255, NHWC [H,W,CH]."""
    i = np.arange(IMG_H * IMG_W * CH, dtype=np.uint64)
    v = ((i * np.uint64(17) + np.uint64(23)) & np.uint64(0xFF)).astype(np.float32) * np.float32(1.0 / 255.0)
    return v.reshape(IMG_H, IMG_W, CH)


def init_model_weights_flat():
    """Exactly init_model's flat weight array: weights[i]=((i*29+7)%31)-15 (FP32), enough for block 0."""
    n = CH * 3 * 3 * CH + CH * CH  # block-0 conv3 tile + conv1 tile
    idx = np.arange(n, dtype=np.uint64)
    return (((idx * np.uint64(29) + np.uint64(7)) % np.uint64(31)).astype(np.int64) - 15).astype(np.float32)


def conv3_fp(inp_f, w3, scale):
    """Microbench conv3x3 (stride-1, pad-1 EDGE/replicate, ReLU6): out=relu6(acc*scale). w3 is
    W3[oc][k][ic], k=(ky+1)*3+(kx+1). Matches conv3_scalar + clamp_coord + relu6_f32 exactly."""
    xp = np.pad(inp_f.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode='edge')  # replicate halo
    out = np.zeros((IMG_H, IMG_W, CH), dtype=np.float32)
    for oc in range(CH):
        acc = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        for ky in (-1, 0, 1):
            for kx in (-1, 0, 1):
                k = (ky + 1) * 3 + (kx + 1)
                win = xp[1 + ky:1 + ky + IMG_H, 1 + kx:1 + kx + IMG_W, :]
                acc += (win * w3[oc, k, :]).sum(axis=2)
        out[:, :, oc] = np.clip(acc * np.float32(scale), 0.0, float(RELU6))  # ReLU6
    return out


def slice_block0_weights(wflat):
    """Split init_model's flat weights into block-0 conv3 W3[oc][k][ic] and conv1 W1[oc][ic]."""
    n3 = CH * 3 * 3 * CH
    w3 = wflat[:n3].reshape(CH, 3 * 3, CH)       # [oc][k][ic]  (microbench: oc*9*CH + k*CH + ic)
    w1 = wflat[n3:n3 + CH * CH].reshape(CH, CH)  # [oc][ic]     (microbench: oc*CH + ic)
    return w3, w1


def quant_act_u8(act_f):
    """ReLU6 activation -> uint8 with scale S_ACT (round-nearest-even, clamp [0,255])."""
    return np.clip(np.rint(act_f.astype(np.float32) / S_ACT), 0, 255).astype(np.uint8)


def quant_w_i8(w_f):
    """Per-tensor symmetric int8: S_w = max|w|/127, round-nearest-even, clamp [-127,127]."""
    amax = float(np.abs(w_f).max())
    S_w = np.float32(amax / 127.0)
    qi = np.clip(np.rint(w_f.astype(np.float32) / S_w), -127, 127).astype(np.int8)
    return qi, S_w


def conv_acc_i32(qact_u8, qw_taps):
    """int32 accumulator of the int8 conv (K-parameterized, REPLICATE pad for K>1).
    qw_taps is int8 [OC][K*K][IC] with tap index k=(ky+1)*K+(kx+1) (microbench order, ky,kx=-r..r).
    acc[y,x,oc] = sum_{ky,kx,ic} qact_padded[y+ky][x+kx][ic] * qw_taps[oc][(ky+1)*K+(kx+1)][ic]."""
    r = (K - 1) // 2
    a = qact_u8.astype(np.int64)
    ap = np.pad(a, ((r, r), (r, r), (0, 0)), mode='edge')     # replicate halo (edge == clamp_coord)
    acc = np.zeros((IMG_H, IMG_W, CH), dtype=np.int64)
    for oc in range(CH):
        s = np.zeros((IMG_H, IMG_W), dtype=np.int64)
        for ky in range(-r, r + 1):
            for kx in range(-r, r + 1):
                k = (ky + r) * K + (kx + r)
                win = ap[r + ky:r + ky + IMG_H, r + kx:r + kx + IMG_W, :]
                s += (win * qw_taps[oc, k, :].astype(np.int64)).sum(axis=2)
        acc[:, :, oc] = s
    assert np.abs(acc).max() < 2 ** 31, "int32 overflow in conv acc"
    return acc.astype(np.int32)


def requant_u8(acc_i32, M):
    """int32 acc -> uint8: * per-tensor M in fp32, round-nearest-even, ReLU6+clamp [0,255].

    Bit-for-bit the kernel's requant: SATUINT8 clamps [0,255], and S_out=6/255 makes 255 == qval(6),
    so this clamp IS the ReLU6 ceiling. Matches tensor_quant INT32_TO_FP32->MUL_COL->FP32_TO_INT32(rne)
    ->SATUINT8. acc<=~4.7M at K=3 so int32->fp32 is exact (< 2^24)."""
    return np.clip(np.rint(acc_i32.astype(np.float32) * np.float32(M)), 0, 255).astype(np.uint8)


def fp32_reference(act_f, w_taps):
    """FP32-path uint8 (quality anchor): full-precision conv of the [0,6] activation with float
    weights, relu6(out*LAYER_SCALE), quantized by S_ACT. REPLICATE pad for K>1."""
    if K == 1:
        a = act_f.astype(np.float32).reshape(IMG_H * IMG_W, CH)
        out = a @ w_taps[:, 0, :].astype(np.float32).T          # [HW,OC]
        y = np.clip(out * np.float32(LAYER_SCALE), 0.0, float(RELU6)).reshape(IMG_H, IMG_W, CH)
    else:
        y = conv3_fp(act_f, w_taps, LAYER_SCALE)                # relu6 already applied
    return np.clip(np.rint(y / S_ACT), 0, 255).astype(np.uint8)


def pack_a_blob(qw_taps):
    """Group-major folded A layout the kernel loads (matches conv_tile's group iteration + the
    tensor_ima8a32 contraction verified in tensors.cpp): line (gidx*CH + oc) holds group gidx's
    weights for output channel oc; within the line byte (ti*CH + ic) = W[oc][tap][ic] where the
    folded tap = ky*K + (c+ti). Group gidx enumerates (ky, chunk c) exactly as the kernel:
        for ky in 0..K:  for c in 0..K step FOLD_TAPS:  nt=min(K-c,FOLD_TAPS)
    For K=1 there is one group (line oc = W[oc][:] in bytes 0..CH-1); for K=3 three groups (one
    per kernel row, nt=3 column taps folded into bytes 0..3*CH-1). Rest zero-padded to 64B."""
    blob = np.zeros((NGRP * CH, ROW_STRIDE), dtype=np.int8)
    gidx = 0
    for ky in range(K):
        c = 0
        while c < K:
            nt = min(K - c, FOLD_TAPS)
            for oc in range(CH):
                for ti in range(nt):
                    k = ky * K + (c + ti)                        # microbench tap index
                    blob[gidx * CH + oc, ti * CH:ti * CH + CH] = qw_taps[oc, k, :]
            c += nt
            gidx += 1
    assert gidx == NGRP, "group count must equal NGRP"
    return blob.tobytes()


def pack_act_blob(qact_u8):
    """Emit the activation PRE-padded to the kernel's [PADH][PADW][CH] buffer: interior filled,
    halo ring left ZERO (the kernel's fill_halo_band replicate-fills it on device, so the gate
    exercises the replicate-pad). At K=1 PADH=IMG_H, PADW=IMG_W, HALO=0 => plain NHWC (unchanged)."""
    padded = np.zeros((PADH, PADW, CH), dtype=np.uint8)
    padded[HALO:HALO + IMG_H, HALO:HALO + IMG_W, :] = qact_u8
    return padded.tobytes()


def write_scales_h(S_w, M, path):
    with open(path, "w") as f:
        f.write("/* auto-generated by gen_yolo_int8.py -- YOLO int8-TFMA scales (mirror of DNCNN_*).\n")
        f.write(" * ReLU6 activations quantize with S=6/255, so SATUINT8's 255 clamp == quantized 6. */\n")
        f.write("#ifndef YOLO_INT8_SCALES_H\n#define YOLO_INT8_SCALES_H\n")
        f.write(f"#define YOLO_SCALES_K {K}  /* op-class this header was generated for; kernel asserts == YOLO_K */\n")
        f.write(f"static const float YOLO_QUANT0    = {1.0/float(S_ACT):.9g}f;  /* ReLU6 act FP32 -> uint8 (=255/6) */\n")
        f.write(f"static const float YOLO_REQUANT[1] = {{ {float(M):.9g}f }};  /* per-layer int32->uint8 requant */\n")
        f.write(f"static const float YOLO_DEQUANT   = {float(S_ACT):.9g}f;  /* uint8 -> FP32 (=6/255) */\n")
        f.write(f"static const int   YOLO_RELU6_HI  = 255;  /* quantized value of 6 (== SATUINT8 upper clamp) */\n")
        f.write("#endif\n")


# ===================== M-i2 full-trunk (mixed-precision) generation =====================
# Network mirrors yolo_vpu_argbuf.c main: 4 blocks of {conv3x3 -> conv1x1} then head1x1.
# Precision split (mirror DnCNN): FP32 first conv (block-0 conv3, input stem) + FP32 head1x1;
# int8 the 7 hidden convs between them (block-0 conv1 ... block-3 conv1). All intermediate
# activations are ReLU6 [0,6] quantized with S=1/42.5 (uint8 0..255 == [0,6]); the int8->int8
# hand-off is plain uint8 (no FP32 round-trip). Requant M_L = S_w_L * LAYER_SCALE_L (S_in=S_out).

def init_model_weights_flat_full():
    """Exactly init_model's full flat weight array: weights[i]=((i*29+7)%31)-15, all 10496 floats
    (YOLO_BLOCKS*BLOCK_WEIGHTS + head 256)."""
    n = YOLO_BLOCKS * BLOCK_WEIGHTS + CH * CH
    idx = np.arange(n, dtype=np.uint64)
    return (((idx * np.uint64(29) + np.uint64(7)) % np.uint64(31)).astype(np.int64) - 15).astype(np.float32)


def blk_conv3_w(wflat, b):
    return wflat[b * BLOCK_WEIGHTS: b * BLOCK_WEIGHTS + CH * 9 * CH].reshape(CH, 9, CH)   # [oc][k][ic]


def blk_conv1_w(wflat, b):
    off = b * BLOCK_WEIGHTS + CH * 9 * CH
    return wflat[off: off + CH * CH].reshape(CH, CH)                                       # [oc][ic]


def head_w_of(wflat):
    off = YOLO_BLOCKS * BLOCK_WEIGHTS
    return wflat[off: off + CH * CH].reshape(CH, CH)                                        # [oc][ic]


def conv_scalar_f32(inp_f, w_taps, k, scale):
    """FP32 conv with the EXACT kernel scalar accumulation order (ky,kx,ic), one float32 rounding
    per multiply and per add (matches -ffp-contract=off), ReLU6, replicate/edge pad. w_taps is
    [OC][k*k][IC] (tap = (ky+r)*k+(kx+r)). Bit-exact vs the on-device conv_first_fp32/scalar."""
    r = (k - 1) // 2
    xp = np.pad(inp_f.astype(np.float32), ((r, r), (r, r), (0, 0)), mode='edge').astype(np.float32)
    out = np.zeros((IMG_H, IMG_W, CH), np.float32)
    for oc in range(CH):
        acc = np.zeros((IMG_H, IMG_W), np.float32)
        for ky in range(-r, r + 1):
            for kx in range(-r, r + 1):
                kk = (ky + r) * k + (kx + r)
                win = xp[r + ky:r + ky + IMG_H, r + kx:r + kx + IMG_W, :]
                for ic in range(CH):
                    acc = (acc + (win[:, :, ic] * np.float32(w_taps[oc, kk, ic]))).astype(np.float32)
        out[:, :, oc] = np.clip((acc * np.float32(scale)).astype(np.float32), 0.0, float(RELU6))
    return out


def head_scalar_u8(deq_f, wh):
    """FP32 head1x1: acc = sum_ic deq[ic]*wh[oc][ic] (sequential f32), 128+acc*HEAD_SCALE,
    round-half-up (floor(v+0.5)), clamp [0,255]. Matches head_fp32 on device."""
    out = np.zeros((IMG_H, IMG_W, CH), np.uint8)
    for oc in range(CH):
        acc = np.zeros((IMG_H, IMG_W), np.float32)
        for ic in range(CH):
            acc = (acc + (deq_f[:, :, ic] * np.float32(wh[oc, ic]))).astype(np.float32)
        v = (np.float32(128.0) + (acc * np.float32(HEAD_SCALE)).astype(np.float32)).astype(np.float32)
        out[:, :, oc] = np.clip(np.floor(v + np.float32(0.5)), 0, 255).astype(np.uint8)
    return out


def conv_acc_i32_k(qact_u8, qw_taps, k):
    """int32 accumulator of an int8 conv (replicate/edge pad), qw_taps int8 [OC][k*k][IC]."""
    r = (k - 1) // 2
    a = qact_u8.astype(np.int64)
    ap = np.pad(a, ((r, r), (r, r), (0, 0)), mode='edge')
    acc = np.zeros((IMG_H, IMG_W, CH), np.int64)
    for oc in range(CH):
        s = np.zeros((IMG_H, IMG_W), np.int64)
        for ky in range(-r, r + 1):
            for kx in range(-r, r + 1):
                kk = (ky + r) * k + (kx + r)
                win = ap[r + ky:r + ky + IMG_H, r + kx:r + kx + IMG_W, :]
                s += (win * qw_taps[oc, kk, :].astype(np.int64)).sum(axis=2)
        acc[:, :, oc] = s
    assert np.abs(acc).max() < 2 ** 31, "int32 overflow in trunk conv acc"
    return acc.astype(np.int32)


def pack_a_blob_k(qw_taps, k):
    """Group-major folded int8 A layout for op-class k (matches the kernel's conv{1,3}_int8_tile group
    iteration + tensor_ima8a32 contraction). Line gidx*CH+oc, byte ti*CH+ic = W[oc][ky*k+(c+ti)][ic]."""
    fold = min(k, 64 // CH)                         # FOLD_BUDGET at CH=16 is 4 -> fold=min(k,4)
    ngrp = k * ((k + fold - 1) // fold)             # 1 at k=1, 3 at k=3
    blob = np.zeros((ngrp * CH, ROW_STRIDE), np.int8)
    gidx = 0
    for ky in range(k):
        c = 0
        while c < k:
            nt = min(k - c, fold)
            for oc in range(CH):
                for ti in range(nt):
                    kk = ky * k + (c + ti)
                    blob[gidx * CH + oc, ti * CH:ti * CH + CH] = qw_taps[oc, kk, :]
            c += nt
            gidx += 1
    assert gidx == ngrp
    return blob.tobytes(), ngrp


def write_trunk_scales_h(quant0, dequant, Ms, InvSw, path):
    with open(path, "w") as f:
        f.write("/* auto-generated by gen_yolo_int8.py trunk -- YOLO int8-TFMA full-network scales.\n")
        f.write(" * FP32 first conv + FP32 head; 7 int8 hidden convs. Per-layer requant M (S_w*LAYER_SCALE);\n")
        f.write(" * S_in=S_out=1/42.5 cancel. QUANT0=42.5 exact (255==qval(6)), DEQUANT=1/42.5 for the head. */\n")
        f.write("#ifndef YOLO_INT8_SCALES_H\n#define YOLO_INT8_SCALES_H\n")
        f.write("#define YOLO_SCALES_K 3  /* trunk uses the K=3 padded geometry */\n")
        f.write(f"#define YOLO_TRUNK_LAYERS {len(Ms)}\n")
        f.write(f"static const float YOLO_QUANT0     = {float(quant0):.9g}f;  /* FP32 first-conv ReLU6 -> uint8 (=42.5) */\n")
        f.write(f"static const float YOLO_DEQUANT    = {float(dequant):.9g}f;  /* uint8 -> FP32 for the head (=1/42.5) */\n")
        req = ", ".join(f"{m:.9g}f" for m in Ms)
        f.write(f"static const float YOLO_REQUANT[{len(Ms)}] = {{ {req} }};  /* per-hidden-layer int32->uint8 requant */\n")
        inv = ", ".join(f"{s:.9g}f" for s in InvSw)
        f.write(f"static const float YOLO_INV_SW[{len(InvSw)}] = {{ {inv} }};  /* per-layer 127/max|w|: on-device int8 weight quant (multiply, no fdiv) */\n")
        f.write("static const int   YOLO_RELU6_HI  = 255;  /* quantized value of 6 (== SATUINT8 upper clamp) */\n")
        f.write("#endif\n")


def gen_trunk():
    os.makedirs(BENCH, exist_ok=True)
    inp_f = init_model_input()
    wflat = init_model_weights_flat_full()

    # ---- L0 FP32 first conv (block-0 conv3, input stem) -> ReLU6 -> uint8 (trunk input) ----
    a0 = conv_scalar_f32(inp_f, blk_conv3_w(wflat, 0), 3, CONV3_SCALE)     # float32 [0,6]
    cur = np.clip(np.rint(a0 * QUANT0), 0, 255).astype(np.uint8)          # multiply by 42.5 (no fdiv)

    # ---- 7 int8 hidden convs: block0 conv1 ... block3 conv1 ----
    seq = [(1, blk_conv1_w(wflat, 0)[:, None, :], CONV1_SCALE),           # L1
           (3, blk_conv3_w(wflat, 1),             CONV3_SCALE),           # L2
           (1, blk_conv1_w(wflat, 1)[:, None, :], CONV1_SCALE),           # L3
           (3, blk_conv3_w(wflat, 2),             CONV3_SCALE),           # L4
           (1, blk_conv1_w(wflat, 2)[:, None, :], CONV1_SCALE),           # L5
           (3, blk_conv3_w(wflat, 3),             CONV3_SCALE),           # L6
           (1, blk_conv1_w(wflat, 3)[:, None, :], CONV1_SCALE)]           # L7
    blobs, Ms, Sws, ngrps, hidden, InvSw = [], [], [], [], [], []
    for (k, w, lscale) in seq:
        amax = float(np.abs(w).max())
        inv_sw = np.float32(127.0 / amax)                                # baked reciprocal (no fdiv on device)
        qw, Sw = quant_w_i8(w)                                            # per-tensor symmetric int8 (divide)
        # on-device quantizes by MULTIPLY (qi = clip(rint(w*inv_sw),-127,127)); assert bit-identical
        qw_mul = np.clip(np.rint(w.astype(np.float32) * inv_sw), -127, 127).astype(np.int8)
        assert np.array_equal(qw_mul, qw), "on-device multiply quant must equal generator divide quant"
        M = np.float32(Sw) * np.float32(lscale)
        acc = conv_acc_i32_k(cur, qw, k)
        cur = requant_u8(acc, M)                                          # -> uint8, ReLU6 ceiling folded
        blob, ngrp = pack_a_blob_k(qw, k)
        blobs.append(blob); Ms.append(float(M)); Sws.append(float(Sw)); ngrps.append(ngrp)
        InvSw.append(float(inv_sw))
        hidden.append(cur.copy())                                         # L1..L7 activations (probes)

    # ---- L8 FP32 head1x1 (+128) over the dequantized last hidden activation ----
    deq = cur.astype(np.float32) * DEQUANT_T
    ref_i8 = head_scalar_u8(deq, head_w_of(wflat))                        # uint8 output (max_abs=0 gate)

    # ---- FP32-path anchor (all layers FP32, no intermediate quant) for the quality delta ----
    fa = conv_scalar_f32(inp_f, blk_conv3_w(wflat, 0), 3, CONV3_SCALE)   # block0 conv3
    for b in range(YOLO_BLOCKS):
        if b > 0:
            fa = conv_scalar_f32(fa, blk_conv3_w(wflat, b), 3, CONV3_SCALE)
        fa = conv_scalar_f32(fa, blk_conv1_w(wflat, b)[:, None, :], 1, CONV1_SCALE)
    ref_fp = head_scalar_u8(fa, head_w_of(wflat))

    # ---- emit: concatenated int8 A-blob, FP32-quality npy skip, trunk scales header, reference npy ----
    with open(os.path.join(BENCH, "yolo_trunk_int8_weights.bin"), "wb") as f:
        for blob in blobs:
            f.write(blob)
    np.save(os.path.join(BENCH, "yolo_trunk_reference_int8.npy"), ref_i8)
    np.save(os.path.join(BENCH, "yolo_trunk_probe1_L1.npy"), hidden[0])   # L1 (block0 conv1) probe
    np.save(os.path.join(BENCH, "yolo_trunk_probe2_L2.npy"), hidden[1])   # L2 (block1 conv3) probe
    write_trunk_scales_h(QUANT0, DEQUANT_T, Ms, InvSw, os.path.join(SRC, "yolo_int8_scales.h"))
    # committed CI reference (repo path the benchmark_config points at)
    refdir = os.path.join(REPO, "ported_models", "yolo", "refs")
    os.makedirs(refdir, exist_ok=True)
    np.save(os.path.join(refdir, "yolo_trunk_int8_reference.npy"), ref_i8)          # CI gate (final)
    np.save(os.path.join(refdir, "yolo_trunk_probe1_L1.npy"), hidden[0])            # local M-i2 gate
    np.save(os.path.join(refdir, "yolo_trunk_probe2_L2.npy"), hidden[1])            # local M-i2 gate

    ablob_bytes = sum(len(b) for b in blobs)
    d = np.abs(ref_i8.astype(int) - ref_fp.astype(int))
    print(f"TRUNK  layers int8={len(Ms)}  ngrps={ngrps}  a_blob={ablob_bytes}B")
    print(f"QUANT0={float(QUANT0):.6g} DEQUANT={float(DEQUANT_T):.8g}")
    print("Ms =", [f"{m:.4e}" for m in Ms])
    print("Sws=", [f"{s:.5f}" for s in Sws])
    print(f"int8 ref : shape {ref_i8.shape} min {int(ref_i8.min())} max {int(ref_i8.max())} "
          f"sat255={int((ref_i8==255).sum())} zeros={int((ref_i8==0).sum())} sum={int(ref_i8.sum())}")
    print(f"ACCURACY int8-vs-FP32(head): max_abs={int(d.max())} mean_abs={d.mean():.4f} "
          f"frac>1={100*(d>1).mean():.3f}%  (informational)")


def main():
    if TRUNK:
        gen_trunk()
        return
    os.makedirs(BENCH, exist_ok=True)
    inp_f = init_model_input()
    w3, w1 = slice_block0_weights(init_model_weights_flat())
    act_f = conv3_fp(inp_f, w3, CONV3_SCALE)    # genuine [0,6] ReLU6 activation into the layer

    if K == 1:
        w_taps = w1[:, None, :]                 # [OC][1][IC]
    else:
        w_taps = w3                             # [OC][9][IC], microbench tap order

    qact = quant_act_u8(act_f)
    qw_flat, S_w = quant_w_i8(w_taps)           # symmetric int8 over the whole tile
    qw_taps = qw_flat                           # same shape as w_taps
    M = np.float32(S_w) * np.float32(LAYER_SCALE)  # S_in=S_out cancel; M = S_w * LAYER_SCALE

    acc = conv_acc_i32(qact, qw_taps)
    ref_i8 = requant_u8(acc, M)                 # what the kernel must reproduce (max_abs=0 gate)
    ref_fp = fp32_reference(act_f, w_taps)      # quality anchor

    # emit blobs + header
    with open(os.path.join(BENCH, "yolo_int8_input.bin"), "wb") as f:
        f.write(pack_act_blob(qact))
    with open(os.path.join(BENCH, "yolo_int8_weights.bin"), "wb") as f:
        f.write(pack_a_blob(qw_taps))
    np.save(os.path.join(BENCH, "yolo_reference_int8.npy"), ref_i8)
    write_scales_h(S_w, M, os.path.join(SRC, "yolo_int8_scales.h"))

    d = np.abs(ref_i8.astype(int) - ref_fp.astype(int))
    print(f"K={K}  HALO={HALO} PADW={PADW} PADH={PADH} FOLD_TAPS={FOLD_TAPS} NGRP={NGRP}  "
          f"LAYER_SCALE={float(LAYER_SCALE):.6g}")
    print(f"S_act={float(S_ACT):.6g} (=6/255)  S_w={float(S_w):.6g}  M={float(M):.6g}")
    print(f"qact  uint8: min {int(qact.min())} max {int(qact.max())}   act_blob={PADH*PADW*CH}B  a_blob={NGRP*CH*ROW_STRIDE}B")
    print(f"qw    int8 : min {int(qw_flat.min())} max {int(qw_flat.max())}  (S_w=max|w|/127, max|w|={float(np.abs(w_taps).max())})")
    print(f"int8 ref   : shape {ref_i8.shape} min {int(ref_i8.min())} max {int(ref_i8.max())}  "
          f"sat255={int((ref_i8==255).sum())} zeros={int((ref_i8==0).sum())}")
    edge = np.concatenate([d[0].ravel(), d[-1].ravel(), d[:, 0].ravel(), d[:, -1].ravel()])
    print(f"ACCURACY int8-vs-FP32: max_abs={int(d.max())} mean_abs={d.mean():.3f} "
          f"frac>1={100*(d>1).mean():.2f}%  edge_max_abs={int(edge.max())}  (quality gate, informational)")


if __name__ == "__main__":
    main()
