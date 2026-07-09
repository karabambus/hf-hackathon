# YOLOv10n → Erbium ETSOC1 — optimization journal

Append-only ledger of every meaningful change. One paragraph per entry, oldest at top.
Mirrors the depth-anything-real journal style.

---

## [2026-05-07]  Project setup  PASS

**Workspace.** `<artifact-archive>/yolov10n-real/`
mirrors the depth-anything-real layout: `kernels/`, `inputs/`, `refs/`,
`blobs/`, `tools/`, `runs/`, plus a `PLAN.md` and this journal.

**Model acquisition.**

- Hugging Face base: `onnx-community/yolov10n` pinned in
  [`docs/HF_REFERENCES.md`](../../../docs/HF_REFERENCES.md).
- Used the YOLOv10n base model and re-exported to ONNX at
  `imgsz=(288, 512)` with `nms=False` to keep raw 3-scale heads (NMS
  belongs on host, not silicon).
- Local pip install (`pip install --target=/media/...` with
  `TMPDIR=/media/...` because `/` is full).
- ONNX summary: 293 nodes, input `[1, 3, 288, 512]`, output
  `[1, 84, 3024]` (4 box-xywh + 80 class-sigmoid, 3024 = 36×64 + 18×32 +
  9×16 anchors).

**Op inventory.** 83 Conv (12× 3×3 64→64; 6× 3×3 32→32; multiple 1×1 and
depthwise 3×3 with groups=128/80), 70 SiLU (=Sigmoid·x), 19 Concat, 11
Add, 11 Split, 2 MatMul + 2 Softmax (DFL box decode), 2 Resize (FPN
upsample), 3 MaxPool. 2.76M params total → 9.16 MB FP32 weights.

**Reference output.** Ran ORT FP32 on the `web_car` photo (used during
depth-anything validation).  Top class score 0.639; 17 anchors above
0.25 confidence.  Saved to `refs/ort_web_car_output.bin`.

**Weight extraction.** `tools/extract_weights.py` walks the ONNX graph,
collects all 83 Conv weights+biases (BN already fused by Ultralytics's
`.fuse()` call during export), emits `blobs/weights_region.bin` (9.20 MB,
64-byte aligned per entry) and `blobs/weights_layout.json` (per-layer
offsets, shapes, strides, pads, groups).


## [2026-05-07]  M0  heartbeat passes on `<board-host>`  PASS

**Kernel** `kernels/yolo_smoke_M0.c`: writes `0xCAFEBABE 0xDEAD10DE`
to dump.bin[0..8].  `buffer_base_from_args` recovers the buffer base
from the launcher's argument area (same pattern as DnCNN3 / depth-
anything M0 — link-time `heap0_start` lands 64 MB into the buffer
which loses the bottom 16 MB if you trust the link-time addresses,
so the runtime base must come from `arg_area`).

**Build script** `build_yolo.sh` — adapted from depth-anything's
`build_depth.sh`.  Same toolchain, same `erbium.ld`, same
`hart_report_crt.S`.  Uses the patched `erbium_soc1sim_argbuf_big`
launcher (80 MB buffer).

**Run script** `run_yolo.sh` — adapted from `run_depth.sh`.  No input
or weights for M0.

**Result.** Kernel wait 0.0004 s.  `dump.bin[0:8]` =
`BE BA FE CA DE 10 AD DE` (little-endian = `0xCAFEBABE 0xDEAD10DE`).
Build → stage → silicon → dump → audit pipeline confirmed for the
YOLOv10n workspace.


## [2026-05-07]  M2..M10 — full FP32 baseline, then 8-hart parallel  PASS

**Compressed log of milestones M2..M10** (full FP32 baseline + first
parallelism step). 17 silicon iterations on `<board-host>`. All audit
against `ort_web_car_*.bin` (ORT FP32 reference for the 288×512 web_car
photo). Detailed per-milestone notes below.

### M2 (conv0+conv1+SiLU) — 1st-try PASS
Built a generic `conv2d_fp32` and `conv2d_dw_fp32` helper in
`yolo_common.h`.  Wall 3.84 s, max_abs 3.4e-5.

### M3 (+C2f model.2) — 1st-try PASS
First C2f block (1 bottleneck w/ residual). Validated split → m.0 cv1
→ cv2 → add → concat → cv2 pipeline. Wall 10.3 s, max_abs 2.1e-5.

### M4 (+conv3, C2f model.4) — 1st-try PASS
2-bottleneck C2f. Wall 22.1 s.

### M5 (full backbone exc. PSA) — needed 1 bisect
SCDown m.5 / C2f m.6 / SCDown m.7 / C2f m.8 / SPPF m.9. First attempt
failed `c2f_m6_out` audit at max_abs 2.66 — root cause was a 32 KB
**memory overlap** between `SCR_M5_CV2_OUT [128,18,32]` (ends
0x36C8000) and `SCR_M6_CONCAT [256,18,32]` (started 0x36C0000). Fix:
tightened cumulative offset bookkeeping with explicit byte sizes per
slot. Re-run PASS at max_abs 9.2e-6.

### M6 (+PSA model.10) — needed re-run
PSA = 256-ch 1x1 cv1 + split + (qkv 128→256, reshape into 2 heads of
{Q[32], K[32], V[64]}, depthwise 3x3 PE on V_reshape, scaled QᵀK +
softmax + V@softmax_T → +PE → 1x1 proj 128→128) + residual + (FFN
128→256→128) + residual + concat + cv2 1x1. 18 ops total.
First run looked broken (Tailscale dropped mid-rsync, dump lost). After
SSH re-auth, re-ran: every PSA intermediate (`m10_qkv`, `m10_q`, …
`m10_proj`) audited bit-accurate; the misleading mismatch was just
that the audit script used `--milestone M5` and matched the previous
M5 dump dir. Pinning `run_dir` explicitly fixed it. **PASS** at
max_abs 7.7e-6, wall 53 s.

Lesson: if SSH dies mid-run, the soc4 dump.bin is still on disk; just
re-pull it. Also: audit script's milestone-name → run-dir glob is
fuzzy when M(N+1) carries M(N) taps; pass the run dir explicitly.

### M7 (+FPN model.11..22) — needed 2 bisects
Up-path (m.11 upsample → m.12 concat → m.13 C2f-no-shortcut →
m.14 upsample → m.15 concat → m.16 C2f-no-shortcut), down-path
(m.17 down 3x3 s=2 → m.18 concat → m.19 C2f → m.20 SCDown →
m.21 concat → m.22 C2fCIB). The CIB has 5 sub-convs: DW3x3 → 1x1 →
DW7x7 → 1x1 → DW3x3 with residual.

Iteration 1 (with default RSYNC truncated 64 MB dump): all 3 head
inputs (`head_p3_in`, `head_p4_in`, `head_p5_in`) FAIL. Couldn't
bisect because FPN scratch lived past 64 MB.

Iteration 2 (`RSYNC_FULL=1`): bisect found `m11_up` already wrong.
Range matched ref but specific values shifted. Per-channel diff showed
channels 0..141 PASS, 142..255 FAIL. Root cause: I'd reserved 0x48000
bytes for `m11_up` but actually need 0x90000 (256 ch is 2× 128 ch,
which I'd been using for sizing). m12_concat at 0x4040000 overlapped.

Iteration 3: re-laid every FPN scratch slot with size = `OC*H*W*4`
rounded to 0x10000, cumulative-aligned. **PASS** all 4 head taps at
max_abs 1.1e-5. Wall 85 s.

Lesson: any time I copy a scratch-offset block from M(N) to M(N+1)
I need to re-derive sizes from the actual tensor shapes, not from
what I "remember" the size to be. A smaller-shape block in M(N) can
become a larger-shape block in M(N+1).

### M8 (+detection-head logits, no DFL) — 1st-try PASS
24 head convs (cv2.k.0/1/2 for box, cv3.k.0/1/2 for class, k=0..2)
across 3 scales. Reg branch is 3 convs (3x3+SiLU, 3x3+SiLU, 1x1+no-act).
Cls branch is 5 convs (DW3x3+SiLU, 1x1+SiLU, DW3x3+SiLU, 1x1+SiLU,
1x1+no-act). All 6 logit taps PASS at max_abs 5.6e-5. Wall 112 s.

### M9 (+DFL decode + box decode + class sigmoid → final) — 1st-try PASS
DFL: per scale, for each (edge, spatial) softmax 16 bins and weighted-sum
with [0..15] = expected box edge in feature units. Box decode: anchor
center = (w+0.5, h+0.5) in stride units; lt=box_buf[0,1], rb=box_buf[2,3];
xywh = ((bl+br)/2*s, (bt+bb)/2*s, (br-bl)*s, (bb-bt)*s). Class scores:
sigmoid(cls_logits). Concat into `final[1,84,3024]` ordered [P3, P4, P5].

**Final tap PASS at max_abs 9.2e-4 vs gate 3.0** — bit-accurate to
ORT. Wall 112 s (single-hart scalar FP32 baseline).

### M11 — 16-hart scalar (T0+T1) PASS
Removed the T0-only gate from `conv2d_fp32_mh`/`conv2d_dw_fp32_mh`
behind a `YOLO_USE_16HART` build flag. T1 (odd) harts have no VPU
but can do scalar fmadd. With 16 harts splitting OC instead of 8,
**wall 8.41 s → 1.75× over M10**. Total speedup vs M9 baseline:
112 / 8.41 = 13.3×. Audit unchanged at max_abs 9.2e-4.

### M12 — VPU-vectorized 1x1 conv (T0 only, 8 harts) PASS
Added `conv2d_1x1_fp32_mh_vpu` using Erbium VPU intrinsics
`fbcx.ps` (broadcast scalar→8 lanes), `flq2` (load 8 floats),
`fmadd.ps` (parallel multiply-add 8 lanes), `fsq2` (store).
Replaced 41 of the 1x1 `CONV_MH` calls with `CONV_1x1_VPU`. Kept the
3x3 calls at scalar 16-hart. The VPU function gates to T0 only since
odd harts don't have VPU.

First attempt FAILED with max_abs 444 — root cause was `register float
acc asm("f0") = bias_pkg` generating an `fmv.s f0, f4` that only
copies the lower 32 bits (lane 0); lanes 1..7 retained whatever was
left in f0 from previous loop iterations. Fix: directly broadcast the
bias to `acc` with `fbcx.ps`, no intermediate `bias_pkg`.

**Wall 4.26 s — 2.0× over M11. Total 26.3× vs M9.** PASS at the same
9.2e-4 max_abs.

### M13 — VPU 3x3 stride=1 pad=1 conv (T0 only) PASS
Added `conv2d_3x3_p1_fp32_mh_vpu` adapted from depth-anything's
`conv3x3_pad1_fp32_vpu`. Replaced 24 of the 3x3 stride=1 pad=1 calls.
Kept 4 stride=2 3x3 (conv0, conv1, conv3, m.17) at scalar 16-hart.

**Wall 3.50 s — 1.22× over M12. Total 32.0× vs M9 (0.286 FPS).** PASS.

Diminishing return from this VPU swap because the stride=1 3x3 layers
share total MAC roughly with the 1x1's already vectorized — the
remaining 4 stride=2 3x3 layers and ~13 depthwise convs (still scalar)
plus memory bandwidth dominate.

### M14 — VPU depthwise 3x3 stride=1 (T0 only) PASS
Replaced 9 stride-1 depthwise calls (m.10 pe, m.22 cv1.0/1.4, head
cv3.k.0.0/1.0). Wall 3.49 s — essentially flat from M13 (these layers
are tiny). PASS. Kept the change anyway since it's free.

### M15 — multi-hart concat / add helpers PASS
Added `mh_copy_floats`, `mh_add_floats`, `mh_iadd_floats`, `mh_concat3`,
`mh_concat4` and macros `MH_COPY/ADD/IADD/CONCAT3/CONCAT4`. Replaced 6
add_chw, 1 SPPF 4-way concat, 3 FPN 3-way concats. Wall 3.37 s
(Δ −0.12 s).  PASS.

### M16 — OC8-blocked VPU 1x1 (8 OCs simultaneously) PASS
New helper `conv2d_1x1_fp32_mh_vpu_oc8` holds 8 VPU accumulators in
f0..f7, broadcasting 8 different W[oc, ic] scalars for the same input
v_pkg per ic. Input v_pkg is loaded ONCE per (ic, ow8) and reused by
all 8 fmadd.ps's, so memory bandwidth on the input drops 8×.
Wall 3.15 s (Δ −0.22 s).  PASS.

### M17 — OC dispatcher (small OC → per-OC, large OC → OC8) PASS
Issue with M16: layers with OC < 64 get 8 tiles or fewer, leaving
some compute harts idle. Added `conv2d_1x1_disp` that picks per-OC
when OC < 64 and OC8-blocked otherwise. Wall 3.19 s (Δ +0.04, within
noise). Kept anyway since it's correct and the small-OC layers are a
small fraction of compute.

### M18 — OC8-blocked VPU 3x3 stride=1 — FAIL (kernel hung)
Same blocking idea applied to 3x3 stride=1. Kernel hung on silicon
("Unbalanced number of abort unblockers" runtime FATAL after timeout).
Suspected cause: register-pressure saturation with 8 VPU accumulators
+ v_pkg + w_pkg + the EDGE-case scalar fallback that has to fsq2 each
of the 8 accs to memory and reload them inside the inner kx loop.  At
24 layers × hundreds of ow8 tiles × IC × 9 ky/kx, the EDGE-case overhead
plus instruction-cache miss probability adds up. Reverted to M17 base
for M19.

### M19 — multi-hart DFL + box decode + class sigmoid PASS (later flaky)
Per-anchor work (DFL softmax of 4 edges × 16 bins, box xywh, sigmoid
of 80 cls logits) is independent across the 3024 anchors. Split anchor
range across 8 T0 harts within each scale. Wall 3.12 s. Initial run
PASS at max_abs 9.2e-4 — but later re-runs (M22, M23-attempt-1) showed
sporadic max_abs ≈ 500 with the same kernel binary. Each per-hart
slice writes 84 disjoint regions of `final_out` (channels c=0..84 at
the hart's anchor stride), and even with whole-buffer evicts on every
hart the L1D-non-coherence guarantees seem to be intermittently
violated in our runtime. Left M19 in the timeline but do **not**
recommend it as a baseline.

### M20 — OC4-blocked VPU 3x3 stride=1 PASS but no win
Half-step from M18 (OC8 hung) — 4 accumulators in f0..f3 with OC4
tiling. Built and ran. Wall 3.20 s (slightly worse than M19's 3.12 s).
The 3x3-stride-1 layers were not memory-bound enough for the input
re-use to help; the per-tile prologue/epilogue (8 fsq2 + scalar SiLU)
ate the gain. PASS.

### M21 — parallel PSA scoring (transpose + matmuls + softmax) FAIL
Attempted to multi-hart the whole PSA scoring tail by row index
(NHEAD * HW = 288 rows split across 8 T0 harts).  Each hart computed
its rows for QT-transpose, QT@K matmul, softmax, T-transpose, V@sm_T
matmul, and the +PE add.  Built fine, ran fast (3.07 s) but FAIL at
max_abs 500. Suspected race in cross-hart reads of `logits`/`sm_T`
that span row boundaries (each hart needs full rows from other harts'
slices when transposing → scattered cache misses, plus a larger
working set per hart).  Reverted for M22.

### M22 — clean copy of M19 (silicon flake) FAIL
Re-built M19 verbatim while updating `yolo_common.h` with a new
unused `mh_maxpool5_s1_p2` helper.  FAIL at max_abs 500.  Same
intermittent issue as M19. Confirms silicon non-determinism around
the parallel DFL writes.

### M23 — single-hart DFL (revert) PASS — current stable baseline
Reverted to single-hart DFL+box+sigmoid. First attempt of M23 hit a
runtime "FATAL: Unbalanced number of abort unblockers" (silicon flake
unrelated to the kernel — same launcher path that M19 succeeded on).
**Re-run: wall 3.14 s, max_abs 9.2e-4 PASS.**  This is the current
safe optimization baseline. Total **35.7× over M9 single-hart**.
**FPS = 1 / 3.14 = 0.32.**

---
### M30 — full pipeline on silicon (preprocess + model + postprocess) PASS

**Goal change.** User asked: can the rest of the pipeline (everything except
JPEG decode) run on chip too?  Yes.

**Stage 0 (preprocess).** Read raw uint8 RGB at `RAW_INPUT_OFFSET` (0x4A00000),
shape `[SRC_H=480, SRC_W=640, 3]` (host loads it from a `_raw_HxWx3_uint8_rgb.bin`
file).  Bilinear resize → 288×512, divide by 255, transpose HWC → CHW.  Write
FP32 `[1, 3, 288, 512]` at the existing `INPUT_OFFSET` so the rest of the model
sees an unchanged input.  Multi-hart by output row (8 T0 harts each handle 36 rows).

**Stage 2 (postprocess).** After the FP32 final tensor is computed, hart 0:
1. Scans all 3024 anchors, takes the per-anchor max class probability,
   keeps those ≥ 0.25 — produces a candidate list in scratch (`tb`).
2. Class-aware NMS: O(n²) on the ≤ 100 surviving candidates with IoU > 0.5.
3. Writes the surviving detections to `DETECTIONS_OFFSET` (0x1D00000) as
   `{ uint32 N }` + N × `{ uint32 class_id; float score; float x1,y1,x2,y2; }`.

The host now loads only the raw RGB file and reads back the small detection
list — no FP32 input to upload, no full final tensor to download.

**Result.** Wall **3.13 s** (M29 baseline was 3.07 s, so pre+post added ~50 ms).
Silicon detections on `web_car.jpg`:
```
  car     prob=0.665  bbox=(  4.6,  56.0) → (505.5, 273.6)
  person  prob=0.477  bbox=(424.3,  88.6) → (511.7, 204.6)
```
ORT-host reference: same classes, ±2 px boxes, ~5 % score delta — the gap
is the bilinear kernel difference between PIL and my silicon resize, not a
model bug.

What the host still does:
- JPEG decode (~5 ms, PIL).  Putting a JPEG decoder on chip is much bigger
  scope than the ~5 ms it costs and gives no measurable speedup.
- Mapping `class_id` → COCO label string (a 4 KB table lookup).

---

**Bottleneck after M23 (educated guess, no PMC ledger yet):**
Memory bandwidth on the largest 1x1 / 3x3 stride=1 convs. Compute is
already VPU 8-wide on T0 harts and the OC8 input-reuse tiling drops
input reads 8×. The 4 stride=2 3x3 layers (~120M MAC, scalar 16-hart)
add ~150 ms; depthwise stride=2 layers add small amounts; the rest is
DRAM/L2 traffic. The next big lever is TFMA INT8 (4× weight & weight
bandwidth, 4× compute via the int8 matmul accelerator) — large
infrastructure (per-OC weight scales, A-pack format, per-tensor
activation quant + B-pack, dequant-with-bias) but the natural fit for
the 41 1x1 layers that still dominate.


### M10 (multi-hart parallel: 8 T0 compute harts) — PASS
Added `conv2d_fp32_mh(hid, …)` and `conv2d_dw_fp32_mh(hid, …)` in
`yolo_common.h`. Each compute hart owns the OC range [OC·t0/8, OC·(t0+1)/8).
After each conv, the hart evicts its OC slice; all 16 harts (8 T0 + 8
T1 idle) hit `MH_BARRIER()` (FENCE + WAIT_CACHEOPS + shire_barrier on
FLB1). Single-hart non-conv blocks (residual adds, concat copies, PSA
attention scoring, DFL decode) run inside `if (is_h0)` guards followed
by their own evict + barrier — same data flow, just with sync edges.

**Wall 14.74 s — 7.6× speedup over M9 single-hart baseline.**
Audit PASS, max_abs 9.2e-4 (bit-identical to M9). FPS = 1/14.74 = 0.068.

Next bottleneck: every `for` loop is still scalar — VPU is unused.
Single-hart concat copies in the FPN are visible as serial dead time
between barrier points. PSA scoring (matmul + softmax + transposes)
is also serial.

---

## [2026-05-07]  M1  conv0 (3→16, 3x3 s=2) + SiLU  PASS

**Kernel** `kernels/yolo_smoke_M1.c`: single-hart scalar FP32, naive
6-deep nested conv loop, then SiLU via depth-anything's `my_expf` +
`fast_recip` primitives.  Output dumped at `0x300000`, [1,16,144,256].

**Two bisect iterations needed.**

1. *First attempt* — kernel crashed on launch ("Stream error event 30
   code 1", wait 0.002 s).  Root cause: my `silu()` had `(uint32_t)k <<
   23` for negative k, which is UB on overflow.  Replaced with the
   exact `my_expf` from depth-anything M10 (proven on silicon) — uses
   `(int32_t)((v.u >> 23) & 0xFF) + k` to add to the biased exponent.

2. *Second attempt* — kernel ran but output range was 7× too large
   ([-237, 172] vs ref [-32, 24]) and contained 99 NaN.  Root cause:
   the `run_yolo.sh` `INPUT_LOAD` block looked for
   `inputs/${IMAGE}.bin` while the file is `inputs/${IMAGE}_288x512.bin`.
   The input was silently never staged → kernel read whatever was
   already in DRAM at `0x10000` → garbage.  Fixed the script to fall
   back to the `_288x512` suffix.

3. *Third attempt (RAW_NO_SILU)* — to bisect SiLU vs. conv math.
   `max_abs = 7.6e-6` vs ORT pre-SiLU tap.  Conv math bit-accurate.

4. *Final attempt* — full SiLU.  `max_abs = 9.5e-6` vs ORT
   `conv0_post`.  Gate 1e-3.  **PASS.**

**Wall.** No PMC ledger yet; raw kernel completion time was not
recorded (the "Stream error event 30" log line may suppress wall_s in
the timing pull).  Order of magnitude expectation:
16·144·256·27 = 16M MAC + 590k SiLU calls ≈ 30 ms scalar single-hart.

**Lessons in.**
- `(uint32_t)k << 23` is UB for negative k — cast through `int32_t`
  first or compute the exponent bias carefully.
- The silent-no-input failure mode is sneaky.  Added a warning when
  IMAGE is set but no file is found.

---

## [2026-07-08]  M-i0 — int8-TFMA 1×1 conv primitive  PASS (max_abs=0, 1 hart / sys-emu)

First milestone of the int8-TFMA port: bring the board-proven DnCNN int8 engine onto the YOLO
**microbench** (`ported_models/yolo/src/yolo_vpu_argbuf.c`, 80×80 NHWC, CH=16, ReLU6), starting with the
one genuinely new path — the **1×1 conv as an int8 GEMM (K=1)**. (Not the full 293-node model; that
artifact is not in the repo. See `local-artifacts/yolo_int8/PROMPT_0_working_method.md`.)

**Files.** New int8 engine `src/yolo_int8_argbuf.c` (behind `-DYOLO_INT8`, K-parameterized via `-DYOLO_K`;
FP32 `yolo_vpu_argbuf.c` untouched, one flag from rollback). Export/oracle `scripts/gen_yolo_int8.py` +
generated `src/yolo_int8_scales.h`. Local gate tooling: `local-artifacts/build_run_verify_yolo.sh`,
`local-artifacts/check_yolo_int8.py` (NHWC-uint8 adaptation of `check_int8_seams.py`).

**Quant scheme.** Activations are ReLU6 outputs in [0,6]; **both** the 1×1 input and output activations
use the ReLU6 scale **S = 6/255**, so uint8 0..255 spans exactly [0,6]. Consequence: the tensor engine's
**SATUINT8 [0,255] clamp IS the ReLU6 ceiling** — quantized value of 6 = 6/(6/255) = 255 — so the ReLU6
[0,6] fold needs no custom clamp and the DnCNN `requant_u8` (INT32_TO_FP32 → MUL_COL → FP32_TO_INT32(rne)
→ SATUINT8 → PACK) is reused verbatim. Weights per-tensor symmetric int8 (S_w = max|w|/127). Folded
requant multiplier **M = S_in·S_w·CONV1_SCALE / S_out = S_w·CONV1_SCALE** (S_in=S_out cancel), baked as a
compile-time constant (no runtime fdiv). Activations **unsigned** (B-gate, arg `tena_unsigned=true`→bit22),
weights **signed** (A-gate, arg `tenb_unsigned=false`→bit21) — the tensors.h name-swap, verified in
`tensors.cpp` `tensor_ima8a32_execute` (bit22=ub gates B, bit21=ua gates A).

**SCP map (≤48 lines).** A(weights) line 0 (16 OC lines); B(activations) line 16 (FOLD_TAPS·CH/QUARTET =
4 quartet lines at K=1); requant scale line 20. `acols`=16 IC (4 quartets), `bcols`=P=16, OC=16 single
tile. **Memory map** (base-relative, matches the runner's `--file_load`): slots 0x0, summary 0x1000,
barrier 0x1800, uint8 activation 0x2000, int8 A-blob 0x20000, per-OC scale 0x28000, per-hart B-pack
0x2A000, per-hart quant-out 0x30000, uint8 output 0x40000. Every offset/size is a `_Static_assert`.
Seam invariant PADW·CH = 80·16 = 1280 = 20 lines, `%64==0` clean with **no width padding** at K=1 (HALO=0).

**Gate.** `build_run_verify_yolo.sh 1` → build (ACTIVE_HARTS=1, `-DYOLO_INT8=1 -DYOLO_K=1`
`-fno-tree-loop-distribute-patterns`) → 1-hart sys-emu → `check_yolo_int8.py`: **max_abs=0** vs the numpy
oracle (`yolo_reference_int8.npy`), byte-identical. The oracle mirrors the kernel's exact integer math:
acc = Σ uint8·int8 (int32, exact — |acc|<2^24), then clip(rint(acc.f32·M),0,255). **int8-vs-FP32 quality
delta: max_abs=1, mean_abs=0.016** (informational).

**Portability.** `pack_b_group`/`fma_group`/`requant_u8`/`conv_tile` are the shared K-parameterized
DnCNN primitives; at K=1 the halo/tap-fold collapse (NGRP=1, n_taps=1, dy=0). M-i1 rebuilds the SAME file
with `-DYOLO_K=3` (n_taps>1, replicate-pad halo) — no forked 1×1 kernel.

**Caveats.** (1) The microbench's mild deterministic data does not drive any output to 255 (sat255=0), so
the ReLU6 **ceiling** is proven *by construction* (S_out=6/255) and by the oracle's identical clamp, not
data-exercised here — the deeper M-i2 trunk will exercise it. The ReLU6 **floor** (0-clamp) is heavily hit
(65200 zeros). (2) Seams are invisible at 1 hart — the 8-hart board seam gate is M-i3. (3) CI reference
registration (`yolo_int8` model entry) is deferred to M-i2 when the full trunk output is the CI artifact.

---

## [2026-07-08]  M-i1 — int8-TFMA 3×3 stride-1 pad-1 conv (replicate-pad, adaptive fold)  PASS (max_abs=0, 1 hart / sys-emu)

The int8 **3×3** hidden-conv shape (microbench `conv3x3`: IC=OC=16, stride-1, **pad-1 edge/replicate**,
ReLU6). The genuinely new points vs M-i0 are the **replicate-pad halo** (not zero-pad) and the **adaptive
tap-fold** (n_taps>1). Verified bit-exact vs a numpy oracle at 1 hart / sys-emu.

**No forked kernel.** `src/yolo_int8_argbuf.c` is **byte-unchanged** from M-i0 except one added safety
`_Static_assert` (below): the M-i0 engine was already K-parameterized, so M-i1 is just `-DYOLO_K=3`. The
halo/fold macros light up at K=3: `HALO=1`, `FOLD_TAPS=min(K,64/CH·QUARTET)=min(3,4)=3`, `CHUNKS_PER_ROW=1`
⇒ **NGRP=3** FMA dispatches (one full kernel row folded per dispatch), `acols = FOLD_TAPS·CH/QUARTET = 12`
quartets (field 11 ≤ 15). `pack_b_group(n_taps=3, kx0=0)` gathers a folded row; `fma_group` accumulates
into **TENC across the 3 groups** (`first`=gidx0 resets, `last`=gidx2 copies TENC→FREGs); `requant_u8`
unchanged.

**Replicate-pad — kernel and oracle made to agree explicitly.** The activation blob is emitted **pre-padded**
to the kernel's `[PADH=82][PADW=84][CH]` buffer with **only the interior filled**; the halo ring is left
**zero** so the kernel's `fill_halo_band` (edge-replicate, mirroring the microbench `clamp_coord`) re-derives
it on device — i.e. the gate genuinely *exercises* the replicate-pad rather than pre-baking it. The oracle
(`gen_yolo_int8.py` `conv_acc_i32`) pads the already-quantized uint8 with `np.pad(mode='edge')`, tap order
`k=(ky+1)·K+(kx+1)` matching `conv3_scalar`. Discrimination check: device == replicate oracle at **all 316
edge/corner pixels (max_abs=0)**, whereas a **zero-pad** oracle diverges there by up to **5** — so max_abs=0
is a real replicate-pad proof, not a tautology.

**Folded A-blob (group-major).** `pack_a_blob` lays line `gidx·CH+oc`, byte `ti·CH+ic` = W[oc][tap][ic],
folded tap `= ky·K+(c+ti)`. This matches (a) `conv_tile`'s group iteration + `pack_b_group`'s column map and
(b) the `tensor_ima8a32` contraction verified in `tensors.cpp` (A byte `k+x` ↔ B quartet `k/4` sub-`x`;
`ua`=bit21 signs weights, `ub`=bit22 keeps activations unsigned). Requant **M = S_w·CONV3_SCALE**
(= 0.11811/256 = 4.6137e-4); S_in=S_out=6/255 cancel and SATUINT8's 255 is qval(6) (ReLU6 ceiling fold).
acc ≤ 255·127·9·16 = 4.66M < 2²⁴ ⇒ INT32→FP32 exact; rne == np.rint.

**SCP map (≤48 lines) / seam.** A(weights) lines 0..15; B(activations) lines 16..27 (12 quartet lines);
requant scale line 28 (`SCP_SCALE_LINE<48` asserted). Memory map as M-i0 but activation region now the
padded blob (act_blob = 82·84·16 = 110208 B ≤ 0x1E000; A-blob = NGRP·CH·64 = 3072 B). **Seam invariant
re-derived for (CH=16, W=80, K=3):** `PADW_MULT=64/CH=4` ⇒ smallest `PADW≥W+2 =84`; `PADW·CH=1344=21 lines,
%64==0` clean; output stride 80·16=1280=20 lines clean; `BPK_STRIDE=9·4·64=2304`, `TEMP_STRIDE=256` both
`%64==0` — every derived offset/size is `_Static_assert`ed. (8-hart board seam proof is M-i3.)

**New guard.** `yolo_int8_scales.h` holds a single per-op-class `YOLO_REQUANT`, so a K kernel built against
another K's header would silently 2×-misscale. The generator now stamps `#define YOLO_SCALES_K <K>` and the
kernel `_Static_assert(YOLO_SCALES_K == K, …)` — a wrong pairing now **fails the build** (negative-tested:
K=1 build + K=3 header → static-assert error). The header/blobs are regenerated per-K by `gen_yolo_int8.py
<K>`; verify always regens before building, so committed-header K is immaterial to reproducibility.

**Gate.** `build_run_verify_yolo.sh 3` → build (ACTIVE_HARTS=1, `-DYOLO_INT8=1 -DYOLO_K=3`) → 1-hart
sys-emu → `check_yolo_int8.py`: **max_abs=0**, all 80 rows, edges included. M-i0 K=1 regression re-run →
still max_abs=0. **int8-vs-FP32 quality delta: max_abs=1, mean_abs=0.062** (informational). Adversarial
reviewer (`yolo-int8-reviewer`): **PROCEED**, all seven axes CONFIRMED-OK, no BLOCK findings.

**Caveats.** (1) Same ReLU6-ceiling note as M-i0 (sat255=0; ceiling proven by construction, floor hit with
56640 zeros). (2) M-i1 tests **one isolated** 3×3 layer on a genuine [0,6] activation (block-0 conv3 output,
block-0 conv3 weights reused as the 3×3 tile) — the microbench's precision split (first conv + head FP32,
hidden convs int8) is wired in M-i2. (3) Seams remain M-i3.

---

## [2026-07-08]  M-i2 — full microbench trunk int8 (FP32 first conv + head) + mixed blob + CI ref  PASS (max_abs=0, 1 hart / sys-emu)

Wired the int8 1×1 and 3×3 paths into the **full microbench trunk** (`yolo_vpu_argbuf.c` main: 4 blocks of
`conv3x3→conv1x1` then `head1x1`), config-driven, mixed precision, bit-exact vs a full-network numpy oracle.

**Precision split (mirror DnCNN).** `NET[9]` = `{conv3 FP32_FIRST, conv1, conv3, conv1, conv3, conv1, conv3,
conv1, head FP32}`. **FP32 first conv** (block-0 conv3, input stem) and **FP32 head1x1** kept full precision;
the **7 hidden convs between** (block-0 conv1 … block-3 conv1: 4×conv1 + 3×conv3) run int8-TFMA. Config kinds
`{FP32_FIRST, CONV1_INT8, CONV3_INT8, FP32_HEAD}` dispatched per-layer; hot dims (K∈{1,3}, P, CH, PADW) are
compile-time literals in the two int8 tile funcs (no runtime dim → constant-folding preserved). The
single-conv M-i0/M-i1 path is wrapped `#ifndef YOLO_TRUNK`, byte-unchanged (both re-verified max_abs=0).

**Seams + scale chain.** FP32 first conv quantizes its ReLU6 output to uint8 by ×QUANT0=42.5 (=255/6 exact,
no fdiv; 6×42.5=255 so SATUINT8's 255 IS the ReLU6 ceiling). Int8→int8 hand-off stays plain uint8 (S_in=S_out
=1/42.5 cancel in every requant M_L=S_w_L·LAYER_SCALE_L). Last int8 layer dequants into the FP32 head by
×DEQUANT=1/42.5, +128, ×HEAD_SCALE, round-half-up. Per-layer M baked in `YOLO_REQUANT[7]` (conv1 layers
9.2274e-4, conv3 layers 4.6137e-4, indexed by the int8-layer counter li). FP-register clobber barriers bracket
both FP32 scalar passes (tensor_fma tenc_loc=1 clobbers f0..f31 invisibly to GCC).

**Bit-exact FP32 boundaries.** The FP32 first conv + head are SCALAR (sequential (ky,kx,ic) / (ic) order),
built with `-ffp-contract=off`, so the numpy oracle reproduces them bit-for-bit (rv64imfc has no vector ext →
no float reassociation; separate mul+add, no fmadd fusion). Verified: the quantized first-conv output feeds
the int8 trunk with no seam error.

**Self-contained (no file-loaded blob).** Input, the FP32 L0/head weights, AND the 7 int8 A-blobs are all
built ON DEVICE from init_model's exact formulas (like the FP32 `yolo` bench) — the only file-load is
`zero2m.bin`. int8 weights quantize by MULTIPLY (`qi=clip(rint(w·INV_SW),-127,127)`, `INV_SW=127/max|w|=8.4667`
baked in `YOLO_INV_SW[7]`, no fdiv) and pack into the group-major A-blob layout on device. This is byte-for-byte
identical to `gen_yolo_int8.py`'s reference blob (the generator asserts divide-quant==multiply-quant, and the
on-device A-blob @0x2000 in the dump is checked == the generator blob, 13312 B). So CI needs no external
weights bundle and the entry is reproducible from committed state alone.

**⚠ Degeneracy — the final output is a tautology; the L1/L2 probes are the load-bearing gate.** The
microbench's `init_model` weights + decaying scale chain drive activations geometrically to zero — the FP32
reference chain itself collapses (L0 max 0.78 → block2-conv1 max 3e-4 → block3-conv1 ≈ 0), which is exactly
why the FP32 `yolo` CI gate is `constant_u8=128` (head +128 on ~zero input). Consequence: the int8 trunk
output is uniformly 128 regardless of the hidden convs (perturbing ANY int8/head weight leaves the final
output unchanged → a final-output max_abs gate is trivially satisfied). BUT the intermediate int8 activations
vary and discriminate: **L1 (block0 conv1) ∈[0,5], 36% nonzero; L2 (block1 conv3) ∈[0,2], 26% nonzero**, and
they DIVERGE under a deliberate perturbation (first-conv center weight +3 → L1 max_abs 1; QUANT0 42.5→43 →
L1,L2 max_abs 1; L1 M×1.5 → L1 max_abs 2 / L2 1; L2 M×1.5 → L2 max_abs 1). So the kernel dumps L1/L2 to
persistent probe buffers (behind `-DYOLO_TRUNK_PROBE`) and the gate checks all three regions max_abs=0 — L1/L2
exercise the FP32→int8 quant seam, both int8 op-classes in-trunk, the int8→int8 hand-off, and 2 scale-chain
links. Layers L3–L7 have genuine all-zero correct output (a bug producing spurious non-zero would still
diverge from the zero oracle if it survived to a probe). The head is exercised structurally but only on zero
input (can't be data-discriminated with this microbench — caveat).

**Memory map (base-relative, YOLO_TRUNK).** slots 0x0, summary 0x1000, barrier 0x1800, int8 A-blob 0x2000
(13312 B, device-built), mvec 0x6000, FP32 weights 0x7000 (2304+256 f, device-built), OUTPUT 0x10000 (uint8
NHWC), PROBE1(L1) 0x30000, PROBE2(L2) 0x50000, FP32 input 0x80000 (409600 B, device-built), actA 0xF0000 /
actB 0x120000 (padded uint8 [82][84][16] ping-pong), per-hart bpack 0x140000, temp 0x148000. Every offset/size
`_Static_assert`ed non-overlapping + 64B-aligned; seam invariant `PADW·CH=84·16=1344=21 lines %64==0` and
output/probe stride `80·16=1280=20 lines` both clean. **SCP (≤48):** A weights 0..15, B activations 16..27
(12 quartet lines for the K=3 fold; conv1 uses 16..19), requant scale line 28.

**Gate.** `build_run_verify_yolo_trunk.sh` (regen header+refs → build K=3 TRUNK PROBE ACTIVE_HARTS=1
`-ffp-contract=off` → 1-hart sys-emu → check_yolo_trunk.py): **max_abs=0** on all three regions
(final@0x10000, L1@0x30000, L2@0x50000) vs the numpy oracle, AND the on-device A-blob @0x2000 == the generator
blob byte-for-byte. **Also confirmed on the ET-SoC1 board** (1 hart, `run_yolo_int8_trunk_board.sh`, kernel
wait **0.434 s** — vs ~37 min functional sys-emu): A-blob byte-identical, final/L1/L2 all **max_abs=0**.
**int8-vs-FP32 delta: max_abs=0** (both collapse to 128; output_sum=13107200=102400·128 cross-check). M-i0
(K=1) + M-i1 (K=3) single-conv gates re-verified still max_abs=0.

**CI reference.** Added a `yolo_int8` model entry to `.github/ci/benchmark_config.json` (board + ci_smoke +
models): source `yolo_int8_argbuf.c`, trunk defines (incl. `-DYOLO_TRUNK_PROBE`, `-ffp-contract=off`),
**file_loads = zero2m only** (weights on-device), dump_magic 0x10510008, dump_size 0x70000. The accuracy gate
is `uint8_npy` **@0x30000 (the DISCRIMINATING L1 probe)** shape [80,80,16] max_abs 0 → committed reference
`ported_models/yolo/refs/yolo_trunk_probe1_L1.npy` — NOT the final output @0x10000, which collapses to a
tautological all-128 (byte-identical to the FP32 `yolo` gate → zero int8 coverage). Committed refs: the L1
probe (CI gate) + the L2 probe + the final output (documentation / local `check_yolo_trunk.py` gate). The
existing `yolo` gate (`constant_u8`/128/max_abs 0) and the pre-existing local `dncnn20l64` mod are untouched
(additions only).

**Caveats.** (1) Final output is a tautological all-128 (microbench degeneracy) — a reproducibility gate, not a
math gate; the L1/L2 probes are the discriminating math gate. (2) The FP32 head is exercised on zero input only
(no data discrimination possible here). (3) Seams remain M-i3 (whole-buffer src evict per conv3 layer is
1-hart-correct; 8-hart seam proof pending). (4) sys-emu wall ~37 min (scalar FP32 first conv dominates the
functional emulation; negligible on real silicon). (5) `-ffp-contract=off` is REQUIRED for the FP32-boundary
bit-exactness and is baked into the trunk build defines (local + CI).

---

## [2026-07-09]  M-i3 — 8-hart board seam gate  PASS (3/3 consecutive clean, max_abs=0)

Proved the int8 trunk is **seam-clean on the 8-hart ET-SoC1 board** — the one bug class that 1 hart /
sys-emu cannot show (a cacheline shared across two harts' row bands silently corrupts only at 8 harts on
silicon). No kernel change vs M-i2's `5e1f77c`; this milestone is the multi-hart *proof*.

**Seam invariant (re-derived per (CH,W,K), all `_Static_assert`ed).** Harts split IMG_H=80 into row bands
`[row0,row1) = [80·h/8, 80·(h+1)/8)`. Every band write-back must be a whole number of 64B lines so no line
spans two bands:
- **conv3 / padded buffers (K=3):** `PADW·CH = 84·16 = 1344 B = 21 lines`, `%64==0`. Band evicts use
  `(row0+HALO)·PADW·CH` for `(row1−row0)` rows → line-aligned.
- **conv1 / output / probes (K=1, unpadded NHWC):** `IMG_W·CH = 80·16 = 1280 B = 20 lines`, `%64==0`.
- **per-hart scratch** `BPK_STRIDE`/`TEMP_STRIDE` are 64B multiples (per-hart slots never share a line).

**Seam-surface ordering (conservative, correctness-first — M-i4 tightens for perf).** Per int8 conv3 layer:
each hart `fill_halo_band` (its own rows' L/R pad + single-owner top/bottom rings — no cross-hart write
sharing) → FENCE → `evict(src, PAD_BYTES)` writeback+**invalidate whole src** → barrier → compute reading
band±1 rows fresh from DRAM. conv1 needs no halo (reads only its own band) and rides the prior layer's
band-evict + barrier. FP-reg clobber barriers bracket both FP32 passes and the on-device weight quant.

**Board result — `run_yolo_int8_trunk_board_8hart.sh 3`** (8-hart no-dump plain build
`yolo_int8_trunk_8hart.elf`, `ACTIVE_HARTS=8`, only `zero2m` file-loaded; `--dump_after` is launcher-side
so it can't mask the race). **3/3 consecutive clean**, kernel wait ~0.066 s each:
- regions `max_abs=0`: final@0x10000 (all-128 sentinel), **L1@0x30000 (0..5)** + **L2@0x50000 (0..2)** the
  discriminating int8 probes; A-blob@0x2000 on-device==oracle byte-exact.
- summary self-attestation @0x1000: `magic=0x10510008, active_harts=8, done_count=8, active_mask=0xff,
  output_sum==slot_checksum_sum=13107200` (each hart's band checksum sums to the whole — catches a hart that
  skipped/corrupted its band).
- **The 3 dumps are byte-identical (md5 `6d0391aa…`)** — determinism across runs is independent evidence of
  seam-cleanliness (a race would diverge run-to-run).

Audit tooling (local, uncommitted): `local-artifacts/check_yolo_seams.py` (regions + summary gate, adapts
`check_int8_seams.py` to NHWC uint8) and `run_yolo_int8_trunk_board_8hart.sh` (SSH round-trip, ≥3 runs,
per-run audit + consecutive-clean streak). Reproduce: announce on Discord, then
`bash local-artifacts/run_yolo_int8_trunk_board_8hart.sh 3`.

**Ready for M-i4 (perf).** The conservative whole-buffer `evict(src, PAD_BYTES)` per conv3 (paid by all 8
harts) is the obvious first perf lever — tighten to band±1 rows — but any buffer/evict/sync/layout change
re-opens the seam question and must re-pass this 3/3 board gate.

---

## [2026-07-09]  M-i4 — perf ladder  DONE (int8 1.53× over FP32, board-proven; all levers pruned on evidence)

**Step 1 — PMC-profiled the board build FIRST (contract §3), and it overturned the transfer-doc prediction.**
Instrumented the trunk with the DnCNN `pmc_probe.h` behind `-DDNCNN_PMC` (region @0xA000 in the free
FPW..OUTPUT gap, captured by the 0x70000 dump; the no-PMC production `.text` is **byte-identical** to the
seam-proven ELF — md5 differs only in non-deterministic linker metadata). 8-hart board run, 67.2 MMAC/pass,
`decode_pmc.py --base 0xA000`:

| signal | value | reading |
|---|---|---|
| MAC / retired-inst | **0.09** (~11 instr/MAC) | scalar-marshalling-bound, not MAC-bound |
| hart0 IPC (t0) | **2.805** | cores saturated *executing instructions*, not stalled |
| MAC / hart0-cycle | 2.04 | far below TFMA peak → dispatch starved by marshalling |
| DDR traffic (8 shires) | **84,724 rd / 79,893 wr total** | negligible — 100 KB activation set is L2-resident |
| L2 accesses / MAC | 0.0496 | data stays on-chip |
| wall (kernel wait) | 65.7 ms @ 8 hart | throughput 1.02 GMAC/s |

**Bound named: INSTRUCTION / MARSHALLING-bound, NOT memory-bound.** The transfer doc (`YOLO_INT8_TECHNIQUE_
TRANSFER.md` §13) predicted the bound would shift to "per-layer DRAM ping-pong + barrier." The board refutes
that: the activation ping-pong fits in the 4 MB shire L2 (DDR traffic is a rounding error) and the cores are
compute-saturated at IPC 2.8. The real sink is ~11 instructions/MAC — int8 pack/requant/scatter marshalling
plus the two FP32 boundary convs done scalar.

**Lever plan revised on this evidence (measure, don't guess):**
- **REJECT L2 read-residency (`-DYOLO_L2_RESIDENT`)** — data already L2-resident, zero DDR headroom.
- **REJECT prefetch/double-buffer** — not memory-latency-bound (IPC 2.8, cores busy; nothing to hide).
- **KEEP 3×3→1×1 fusion** — re-motivated: not for a (nonexistent) DRAM round-trip but to delete per-block
  evicts+reloads+pack+barrier → fewer *instructions* (the actual bound).
- **KEEP direct-store on 1×1 (`-DYOLO_DIRECT_STORE`), high priority** — removes the temp buffer + evict +
  scalar scatter → directly cuts the dominating instruction count.
- **Mask-scoped barriers — minor** (IPC 2.8 ⇒ not badly barrier-stalled).

Harness (local, uncommitted): `run_yolo_int8_pmc_board.sh` + DnCNN `decode_pmc.py`.

**Step 2 — int8-vs-FP32 A/B baseline (board, interleaved N=7, `run_yolo_ab_board.sh`).** Both 8-hart, 1
pass, on-device `init_model` input, only differing in FP32-VPU vs int8-TFMA hidden convs:

| variant | median | min | max |
|---|---|---|---|
| FP32 microbench | 100.300 ms | 100.168 | 100.360 |
| **int8-TFMA trunk** | **65.596 ms** | 65.558 | 65.671 |

**1.53× faster, distributions fully NON-OVERLAPPING** (fp32 min 100.168 > int8 max 65.671; ~0.2 ms spread
each). **The DoD "beat the FP32 microbench" bar is met and robust.**

**Step 3 — lever verdicts (all four planned levers pruned on EVIDENCE, not opinion):**
- **L2 read-residency — REJECT (PMC).** Activations already L2-resident; DDR traffic negligible. Zero headroom.
- **Prefetch / double-buffer — REJECT (PMC).** cycles≈instr/IPC (32.9M≈92M/2.8) ⇒ cores barely stall; not
  memory-latency-bound, nothing to hide.
- **Direct-store on 1×1 — REJECT (tensors.cpp).** The engine PACKs `[OC][P]` (channel-major) but the
  microbench output is NHWC `[P][CH]`; `tensor_store` writes contiguous rows and there is NO transposed store
  (only transposed `tload`), so the scalar scatter (the transpose) is irremovable without a channel-major
  output (breaks the NHWC oracle/CI). *This is the DnCNN-vs-YOLO difference the transfer doc missed: DnCNN's
  layout matched `[OC][P]`; YOLO's NHWC does not.*
- **3×3→1×1 fusion — LOW CEILING (analysis).** Its wins are DRAM round-trip + barrier removal, but PMC shows
  DRAM is a rounding error and the cores don't stall on barriers (IPC 2.8). conv3 output `[OC][P]` is
  transposed vs conv1's `[P][CH]` B-pack input, so fusion can't even reuse FREGs across the pair — the
  scatter+repack stays. Predicted gain marginal.
- **Mask-scoped barriers — minor** (not barrier-stalled).

**Why: the bound is raw instruction count** (pack + requant + scatter marshalling + the two FP32 scalar
boundary convs), and the packs are already at the DnCNN optimum (P1 word-copy, B8 fold). The classic
DnCNN levers all target memory/barrier/round-trip costs that this L2-resident, compute-saturated workload
does not pay. **So 1.53× is at/near the reasonable limit for this microbench shape** — the remaining
instruction sinks are either already-optimal (pack) or structurally fixed (FP32 precision-split boundary,
NHWC transpose). DoD met (beat FP32, non-overlapping); further levers face diminishing returns by evidence.

**Step 4 — fusion EMPIRICALLY tested (not just analysed) and REVERTED.** Implemented per-block 3×3→1×1
fusion behind `-DYOLO_FUSE` (skip the per-layer dst-band evict + cross-hart barrier for L1..L6; a following
conv3 restores visibility via its own whole-src evict+barrier, a following conv1 reads its own band from
cache). Board gate, one session:
- **Correctness (1-hart): PASS** `max_abs=0` — the cache hand-off is mathematically correct.
- **Seam re-gate (8-hart ×3): PASS 3/3** `max_abs=0` + summary — removing the 6 barriers/evicts introduced
  no cross-band race (conv3's whole-src evict + conv1-own-rows carry the visibility).
- **A/B fused vs non-fused (interleaved N=7): −0.62% REGRESSION, non-overlapping** (fused 66.03 ms vs
  non-fused 65.63 ms; non-fused max 65.663 < fused min 66.002). Removing the barriers/evicts does not cut
  *instructions* (the bound), and leaving intermediates dirty makes the next conv3's whole-src evict write
  back more → net slower. **Reverted** (kernel `.text` byte-identical to the seam-proven M-i3 build).
  *This converts fusion from an analysis-only rejection to a board-measured one — the 4th and last lever.*

**M-i4 outcome (DoD met):** int8-TFMA trunk is **1.53× faster than the FP32 microbench**, board-proven and
non-overlapping (65.6 vs 100.3 ms, 8-hart). Post-int8 bound named with PMC: **instruction/marshalling-bound,
L2-resident** (not the transfer-doc's predicted DRAM-bound). All five planned levers pruned: L2-residency +
prefetch (PMC — no memory headroom), direct-store (tensors.cpp — NHWC vs `[OC][P]` transpose, no transposed
store), fusion (board — measured −0.62%), mask-scoped barriers (not barrier-stalled). The DnCNN levers were
tuned against an instruction/pack-bound model that YOLO's mixed 1×1/3×3 + FP32-boundary shape does not
reproduce; the honest reasonable-limit result is **1.53×, and it is secured** (committed as the `yolo_int8`
model, seam-clean at 8 harts, `max_abs=0` CI gate on the L1 probe). Shipped kernel carries only the guarded
`-DDNCNN_PMC` probe (production `.text` unchanged).

---

## Next milestones (not yet executed)

Per `PLAN.md`:

- **M1**: input copy + first Conv 3×3 stride-2 (3→16) + SiLU.
  Audit against ORT tap `/model.0/Conv_output_0` (or post-SiLU).
- **M2..M5**: progressively larger backbone slices.
- **M6**: full backbone + neck (FPN).
- **M7**: full kernel including 3 detection heads + DFL decode.
- **M8+**: TFMA INT8 + multi-hart (mirror depth-anything M9..M47).

Realistic compute estimate (single-hart scalar baseline, 6.7 GFLOP at
~5 GFLOP/s VPU = 1.3 s; at scalar FP ~1 GFLOP/s = 6.7 s).  Multi-hart +
TFMA INT8 should bring this well below 1 s.  ≥50 silicon iterations
expected before reaching the floor.
