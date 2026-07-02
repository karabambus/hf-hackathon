# ET-SoC-1 Hardware Reference

Background on the silicon this hackathon targets. The operational guides
([`ET_SOC1_QUICKSTART.md`](ET_SOC1_QUICKSTART.md)) and the porting field guide
([`opinionated_porting_options/martin.md`](opinionated_porting_options/martin.md))
tell you *how* to build and run kernels. This document explains *what the chip
is*, with public references, so the mental model behind those guides has
citations you can follow.

Everything here is sourced from public material. Where the repo's own toolchain
naming (`erbium`, `soc1sim`) maps onto public terminology, that mapping is
called out. When public numbers and this repo's approximate mental model
disagree (e.g. "~32 shires" vs. the exact 34), trust the exact figures below and
treat the guides' round numbers as intentional simplification.

## What CORE-ET / ET-SoC-1 Is

**ET-SoC-1** ("Esperanto Technologies System-on-Chip 1") is a massively-manycore
RISC-V AI inference accelerator, originally designed by Esperanto Technologies
and first presented publicly by founder Dave Ditzel at **Hot Chips 33 (August
2021)**. It was pitched as an energy-efficient "kilocore" accelerator for ML
recommendation workloads in data centers.[^hc33][^servethehome][^wikichip]

The lineage that matters for this hackathon:

- **Esperanto Technologies** designed and brought up ET-SoC-1 (silicon
  bring-up/characterization in H2 2021), then wound down operations in **July
  2025**.[^corsix]
- **AINekko / AIFoundry** acquired the ET-SoC-1 IP and, in **October 2025**,
  released it as an open-source initiative: a simulator, kernel driver, and
  firmware via the `et-platform` repository under **Apache-2.0**, plus a
  programmer's reference (`et-man`). Full RTL open-sourcing is planned, though
  some third-party-licensed IP blocks may stay proprietary.[^corsix][^etplatform]
- **OpenHW Group `core-et`** hosts the RTL side as the *CORE-ET Agentic Silicon
  Platform (ETASP)*. The original RTL lives in an **Erbium** branch (with full
  microarchitecture documentation and a DV infrastructure); the main branch is
  an ongoing refactor of those Erbium modules into clean, technology-abstracted
  SystemVerilog for FPGA prototyping and ASIC implementation.[^coreet] This is
  why this repo's linker scripts and launcher are named `erbium*`
  (`erbium.ld`, `erbium_soc1sim_argbuf`): **Erbium is the CORE-ET RTL design
  name for the ET-SoC-1 generation.**

So "porting to CORE-ET" = writing kernels for the ET-SoC-1 (Erbium)
microarchitecture, validated in the ET-SoC-1 system emulator (`soc1sim`) and on
real ET-SoC-1 boards.

## Top-Line Specifications

| Property | Value | Source |
|---|---|---|
| Process node | TSMC 7 nm | [^wikichip][^servethehome] |
| Transistors | ~24 billion | [^wikichip][^servethehome] |
| Die area | ~570 mm² | [^wikichip][^servethehome] |
| ET-Minion cores | 1,088 × 64-bit in-order RISC-V, each with a vector/tensor unit | [^hc33][^wikichip] |
| ET-Maxion cores | 4 × 64-bit out-of-order RISC-V (OS-class, can run Linux) | [^hc33][^corsix] |
| Service core | 1 × "minion-lite" management core | [^corsix][^hackster] |
| On-chip SRAM | >160 MB (>160 million bytes) | [^hc33][^pcper] |
| External DRAM | LPDDR4x (≈4 GiB across 8 controllers) | [^hc33][^corsix] |
| Host / IO | PCIe Gen4 x8, eMMC flash | [^hc33][^wikichip] |
| Peak throughput | 100–200 TOPS | [^hc33][^wikichip] |
| Typical power | <20 W for ML recommendation | [^hc33][^servethehome] |
| ET-Minion clock | ~0.5–1.5 GHz | [^servethehome] |
| ET-Maxion clock | ~0.5–2 GHz | [^servethehome] |

## Core Hierarchy (the part that shapes your kernels)

Two public descriptions of the same chip — line up with the mental model in
[`martin.md`](opinionated_porting_options/martin.md) §1:

**Bottom-up naming (WikiChip / Hot Chips):**[^wikichip][^hc33]

```text
ET-Minion            1 in-order RV64 core + vector/tensor unit, 2 hardware threads (harts)
  ×8  -> Neighborhood   8 minions that share physical proximity; enables "cooperative loads"
  ×4  -> Shire          32 minions + 4 banks of L2 SRAM via a crossbar (a "Minion Shire")
  ×34 -> Chip           34 minion shires × 32 = 1,088 ET-Minion cores, on a 2D mesh NoC
```

**Tile view (AINekko open-source description):**[^corsix]

- **44 tiles total** on a bidirectional mesh network-on-chip.
- **Minion tile (a shire):** 32 in-order RISC-V cores, **2 threads each**, + 4 MiB
  SRAM. 33–34 are usable for kernels.
- **Maxion tile:** 4 out-of-order cores + 1 minion-lite management core + 5 MiB
  SRAM (this is the host/OS side).
- **DRAM tiles:** 8 LPDDR4x controllers, ~4 GiB total.
- **PCIe tile:** Gen4 x8 host interface.

Both views agree on the facts the porting guides rely on:
**a shire ≈ 32 minions, each minion has 2 harts, and the shire is the natural
unit of cooperation.** In this repo's runtime, the `--shire N` launcher flag
selects which shire a kernel runs on.

### The vector/tensor unit

Each ET-Minion carries Esperanto's AI-optimized vector/tensor unit. The tensor
path is driven by a custom **multi-cycle tensor instruction**: a separate
controller takes over and runs the operation across the full **512-bit** datapath
for up to **512 cycles**, doing a large matrix multiply while the rest of the
core is free.[^wikichip] This is the hardware behind martin.md's rule *"hart 0
has the useful tensor path; feeding it reliably matters more than adding scalar
work."*

## Memory Model

- **Single, unified address space** spanning the whole ASIC, with a hardware
  **L1 / L2 / L3 cache hierarchy**.[^corsix]
- **L2 is per-shire and reconfigurable:** each shire's L2 is **four 1 MiB SRAM
  banks**, software-configurable as **cache or scratchpad**. Each 8-core
  neighborhood reaches those four banks through a crossbar.[^wikichip] This is
  the "shire-local scratchpad / cooperation zone" the guides tell you to tile
  into.
- **Cooperative loads:** cores within a neighborhood can hand data directly to
  each other without a round trip through the cache/DRAM.[^wikichip]
- **No CPU-style coherence to lean on across minions.** The porting guides'
  hardest footguns — silent corruption from missing barriers, stale
  `dump.bin` from a skipped `evict + WAIT_CACHEOPS + FENCE`, L1D not coherent
  across minions — all follow from treating data movement and synchronization as
  explicit parts of the algorithm rather than something the hardware hides.

The one-sentence version, repeated from martin.md because the silicon earns it:
**slow shared DRAM, fast shire-local scratchpad, explicit movement, explicit
sync, asymmetric harts.**

## Instruction Set

ET-Minion and ET-Maxion are **64-bit RISC-V (RV64)** cores.[^corsix][^wikichip]
This repo's kernels are compiled for `rv64imfc` with the `lp64f` ABI (see
[`ET_SOC1_QUICKSTART.md`](ET_SOC1_QUICKSTART.md)), plus Esperanto's custom
vector/tensor instructions for the matrix engine. Note the practical limit from
martin.md §5.5: some FP operations (division, trig, sqrt, long-float casts) are
not implemented in the current hardware/firmware path — prefer the ET backend's
known workarounds.

## Primary Sources

For deeper reading, in rough order of authority for *this* hackathon:

1. **AINekko `et-platform`** — the open-source simulator, driver, firmware, and
   the `et-man` programmer's reference manual. This is the tooling you actually
   build against.[^etplatform]
2. **OpenHW `core-et`** — the RTL (Erbium) and microarchitecture
   documentation.[^coreet]
3. **Ditzel et al., Hot Chips 33 (2021)** — the original architecture
   disclosure.[^hc33]
4. **WikiChip: "A Look At The ET-SoC-1"** — the clearest public write-up of the
   neighborhood/shire hierarchy and tensor unit.[^wikichip]

[^hc33]: Dave Ditzel et al., "Accelerating ML Recommendation with over a Thousand RISC-V/Tensor Processors on Esperanto's ET-SoC-1 Chip," Hot Chips 33 (2021). Slides: <https://www.esperanto.ai/wp-content/uploads/2021/08/HC2021.Esperanto.Ditzel.Final_.pdf> — IEEE Xplore: <https://ieeexplore.ieee.org/document/9566904/>
[^servethehome]: "Esperanto ET-SoC-1 1092 RISC-V AI Accelerator Solution at Hot Chips 33," ServeTheHome. <https://www.servethehome.com/esperanto-et-soc-1-1092-risc-v-ai-accelerator-solution-at-hot-chips-33/>
[^wikichip]: "A Look At The ET-SoC-1, Esperanto's Massively Multi-Core RISC-V Approach To AI," WikiChip Fuse. <https://fuse.wikichip.org/news/4911/a-look-at-the-et-soc-1-esperantos-massively-multi-core-risc-v-approach-to-ai/>
[^pcper]: "Speaking Esperanto, 1,088 RISC-V Cores With A Pool Of Over 152MB Of SRAM," PC Perspective. <https://pcper.com/2022/04/speaking-esperanto-1088-risc-v-cores-with-a-pool-of-over-152mb-of-sram/>
[^hackster]: "Esperanto Technologies Begins Trialling 1,093-Core ET-SoC-1 RISC-V Machine Learning Accelerator," Hackster.io. <https://www.hackster.io/news/esperanto-technologies-begins-trailing-1-093-core-et-soc-1-risc-v-machine-learning-accelerator-6c01af691701>
[^corsix]: "Esperanto lives on," corsix.org (on AINekko open-sourcing the ET-SoC-1 IP, Oct 2025). <http://www.corsix.org/content/esperanto-lives-on>
[^etplatform]: AIFoundry / AINekko `et-platform` (simulator, driver, firmware; Apache-2.0). <https://github.com/aifoundry-org/et-platform> — programmer's reference: <https://github.com/aifoundry-org/et-man>
[^coreet]: OpenHW Group `core-et` (CORE-ET Agentic Silicon Platform; Erbium RTL branch). <https://github.com/openhwgroup/core-et>
