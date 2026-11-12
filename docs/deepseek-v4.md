# DeepSeek V4 target engine (CPU + optional CUDA tier)

[简体中文](deepseek-v4.zh-CN.md) (older; this English page is the reference)

DeepSeek V4 Flash runs from the official sharded safetensors checkpoint with
no conversion: routed experts stay native fp4, the dense set stays fp8-e4m3
with UE8M0 block scales, MLA + DSA sparse attention, 43 layers, 256 routed
experts + 1 shared, top-6. The engine is CPU-first and CPU-canonical: every
piece of state lives in host memory, and an optional CUDA tier accelerates
prefill and decode stage by stage, falling back to the CPU reference for any
stage it declines. Every GPU stage was accepted only when it reproduced the
engine's CPU reference on greedy text; what "identical" can and cannot mean
across kernels and cache states is spelled out in [Validation](#validation).

## Windows release (recommended)

Windows users do not need to build or copy an engine manually. Download and
unpack the [latest Windows release](https://github.com/JustVugg/colibri/releases/latest),
then start Colibri through the release launcher:

```powershell
coli.cmd web --model F:\path\to\DeepSeek-V4-Flash
```

The launcher reads the checkpoint's `config.json`, selects `deepseek_v4` from
its `model_type`, and starts the matching prebuilt engine. The archive also
contains the CUDA backend; hardware detection and the safe initial CPU/GPU
plan are automatic. Use `coli.cmd doctor --model F:\path\to\DeepSeek-V4-Flash`
to inspect that plan before changing any of the advanced CUDA controls below.

The build instructions in this document are for contributors and custom
builds. Do not mix an engine built from current source into an older release:
the launcher, family registry, engine, and backend libraries are one versioned
unit.

## Scope

- Production code is in `c/deepseek_v4.c` (amalgamated units); the public
  engine/session API is `c/deepseek_v4.h`. The CUDA tier is
  `c/backend_cuda_dsv4.cu` (+ `backend_loader_dsv4.c` on Windows).
- Official sharded safetensors checkpoints load through shared `st.h`;
  standard MXFP4 matmul uses shared `quant.h`.
- Unified `c/coli` routes `run`, `chat`, `serve`, and `web` to V4; `serve` is
  the OpenAI/Anthropic-compatible gateway (`c/openai_server.py`): it renders
  OpenAI and Anthropic tools into V4's native prompt contract and parses DSML
  call blocks back into each protocol (see the
  [per-engine API matrix](api.md#tool-calling-support)); grammar remains
  unsupported.
- Serving is greedy, one KV slot; speculative drafting exists and is off by
  default (`V4_DRAFT`, `V4_MTP`).
- Build targets: x86-64/aarch64 Linux, Windows/MSYS2, arm64 macOS (CPU);
  CUDA tier on Windows (runtime DLL) and Linux (`CUDA=1`, direct link;
  verified under WSL2).

Destroy every session before destroying its engine.

## Memory policy

A checkpoint has 43 transformer layers, hidden size 4096, and 256 routed
experts per sparse layer with top-k 6. Dense weights occupy about 6.27 GiB and
a resident BF16 output head about 1.06 GiB. Routed-expert weights (~137 GiB
on disk, ~12.6 MB each) are streamed and cached according to the RAM budget:
the planner reserves workspace and a minimum expert working set, then enables
dense/head residency and grows the expert cache when memory permits.
`--ram GiB` is a planner budget, not an OS-enforced limit; without it the
budget is derived from available OS memory. `COLI_MODEL_MIRROR=<dir>` (a
second copy of the checkpoint on another SSD) splits expert reads across two
drives — measured 2× read bandwidth; on this workload the two NVMe are the
limit for both prefill refill and decode (see Performance).

## Download

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
  --local-dir /path/to/DeepSeek-V4-Flash
```

The REAP-pruned 150B variant loads with the same engine and needs no
conversion either. It keeps the official FP4 expert layout and the official
dense FP8, and drops 124 of the 256 routed experts per layer:

```bash
hf download puwaer/DeepSeek-V4-Flash-0731-reap-150b \
  --local-dir /path/to/DeepSeek-V4-Flash-reap-150b
```

85 GB on disk instead of 167. Its shards pack an expert's three weights and
three scales non-contiguously, sometimes across shards; the engine reads such
experts per matrix and takes the original contiguous fast path everywhere else
(#1310). The banner reports it by its measured geometry, `43L x 132E`, rather
than the 284B of the official checkpoint, because that number is not its.

Its experts take 76 GB, about half the official checkpoint's, so the RAM that suits the official
checkpoint does not carry over, and more is not always faster: on a 128 GB
Threadripper PRO 3975WX (#1906) `--ram 64` decoded 0.768 tok/s, 90 and 100
0.745, 120 0.765. Two or three runs of your own prompt at different `--ram`
values find the best one for a machine.

A download can finish with a truncated shard even when the client reports
success. If `st.h` rejects a shard as out of bounds, compare every local shard
size with the Hugging Face repository before treating it as an engine failure.

## Build

CPU engine (all platforms):

```bash
cd c
make deepseek-v4            # ARCH=native for the local CPU (default x86-64-v3)
```

The build uses link-time optimization by default. A gcc built without it (the
portable w64devkit on Windows says `LTO support has not been enabled in this
configuration`) builds with `make deepseek-v4 LTO=0`.

### Windows CUDA tier

The engine is MinGW-built (MSYS2 **UCRT64** shell: `MSYSTEM=UCRT64`; the plain
MSYS shell picks the cygwin gcc and fails on `getrusage`), while the CUDA
kernels must be built by nvcc + MSVC into a DLL that the engine loads at
runtime. Two flavours of the same source ship side by side and the loader
picks at start-up by GPU:

| DLL | build target | kernels | GPUs |
|---|---|---|---|
| `coli_cuda_dsv4_dg.dll` | `make cuda-dsv4-dg-dll` (from a vcvars64 shell; fetches the pinned DeepGEMM headers on first use) | DeepGEMM tensor-core prefill paths + generic kernels | compute capability 12.x (RTX 50-series) |
| `coli_cuda_dsv4.dll` | `make cuda-dsv4-dll CUDA_ARCH=portable` | generic CUDA kernels only (fp32 compute from fp8/fp4 decode) | any sm_80+ (RTX 30/40/50, A/H series) |

The release's Windows CUDA package (`colibri-<tag>-windows-x86_64-cuda.zip`)
carries both, built with `DSV4_TC=0`: that leaves out the opt-in cuBLASLt
MXFP8 path (`DSV4_CUDA_TC=1`), so the DLLs need only the `cudart64_12.dll`
shipped beside them and the NVIDIA driver, not cuBLASLt's hundreds of MB.
Unpacked next to the main archive's `deepseek_v4.exe`, the tier is on by
default (`DSV4_CUDA=0` turns it off). A DLL that is not in the engine's folder
and one whose own dependency is missing are told apart at start-up; Windows
reports both as error 126.

The DeepGEMM sm120 headers the DLL needs are not in the repository:
`c/tools/fetch_deepgemm.sh` checks out the community sm120 port of DeepGEMM
(MIT; upstream DeepGEMM has no sm120 kernels) plus its CUTLASS/CuTe submodule
(BSD-3) at a pinned commit into the gitignored `c/third_party/deepgemm/`,
verifies both commit ids, and applies `c/patches-deepgemm-sm120-msvc.patch`
(MSVC ignores `alignas(64)` on by-value TMA descriptors — the patch pads
them). `make cuda-dsv4-dg-dll` and `DEEPGEMM=1` run it on first build
(`make deepgemm-fetch` runs it alone; ~40 MB, git required);
`DEEPGEMM_HOME=<dir>` points at an existing tree instead and `DEEPGEMM_PIN`
bumps the commit. Attribution: `THIRD_PARTY_NOTICES.md`. Loader rules:
try `_dg` first, ask it whether device 0 fits (`dsv4_cuda_backend_arch_ok`),
otherwise load the generic DLL; `COLI_DSV4_DLL=<file name>` forces one. The
start-up banner names the choice: `[DSV4 CUDA] backend=coli_cuda_dsv4_dg.dll
(deepgemm-sm120)`. Deploy = `deepseek_v4.exe` + both DLLs next to it.

### Linux CUDA tier

```bash
cd c
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=portable   # generic kernels, any sm_80+
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 DEEPGEMM=1           # DeepGEMM flavour, sm_120a only (fetches the pinned headers)
make -f Makefile.deepseek-v4 deepseek-v4 CUDA=1 CUDA_ARCH=sm_86 NO_TC=1   # toolkit older than 12.8
```

`CUDA_HOME` defaults to `/usr/local/cuda` (`CUDA_HOME=/usr/local/cuda-13.3`
to pick a versioned install). `CUDA_ARCH` is `native` (the local card),
`portable` (sm_80/86/89/90/120 + PTX) or one `sm_XX`; `DEEPGEMM=1` ignores it
and builds sm_120a only — the DeepGEMM kernels use block-scaled MMA PTX that
exists for no other target. `NO_TC=1` (`-DCOLI_DSV4_NO_TC`) compiles out the
opt-in cuBLASLt MXFP8 path (`DSV4_CUDA_TC=1`), whose block-scaling API needs
cuBLASLt ≥ 12.8; use it for CUDA 12.6 toolkits (Jetson JetPack). `libcuda`
is linked from the toolkit's `lib64/stubs` (driver installs and WSL2 ship
only `libcuda.so.1`), and the binary carries an rpath to `$(CUDA_HOME)/lib64`.

Verified on Ubuntu 22.04 under WSL2 (gcc 11, CUDA 13.3.1 + 12.6.85, RTX
5080): the CPU-only build (6 s), the generic fat binary, the DeepGEMM
binary and the 12.6 `NO_TC=1` object all compile; without the gate the 12.6
compile fails on exactly the four cuBLASLt 12.8 symbols. At run time both
CUDA binaries bring the tier up (`[DSV4 CUDA] device 0: ... sm_120`,
`v4_gpu tier=dense-matvec`) and generate; see [Validation](#validation) for
the WSL2 numbers.

There is no loader on Linux: one flavour is linked directly. A DeepGEMM
binary on a non-sm120 card logs `v4_gpu warning=backend ... does not support
device` and runs CPU-only. Host-slab pinning for expert uploads is
Windows-only for now (`VirtualQuery`); Linux uses pageable copies (correct,
slower refill).

## Run

Canonical serve line (Windows, PowerShell; the environment used for every
number below):

```powershell
$env:COLI_MODEL_MIRROR = "D:\models\DeepSeek-V4-Flash"   # optional second copy
$env:COLI_CUDA = "1"; $env:COLI_GPU = "0"; $env:CUDA_DENSE = "1"; $env:COLI_CUDA_PIPE = "2"
$env:COLI_CUDA_ATTN_BATCH = "1"; $env:COLI_CUDA_MOE_BATCH = "1"
$env:V4_MOE_REFILL_GROUP = "12"; $env:DSV4_CUDA_EXPERT_MIRRORS = "688"
$env:V4_LOADER_LANES = "3"   # GPU tier: 9-lane default tuned for the CPU path, see the env table
python ./coli serve --model C:\models\DeepSeek-V4-Flash --ram 32 --ctx 20000
```

`coli run|chat|web` take the same environment. On Windows a prompt with text
outside ASCII (Chinese, accented letters) does not survive the command line of
the engine run by hand: give it with `--prompt-file <UTF-8 file>`. `--ngen` is a ceiling, not a
target (answers end at EOS; an oversized ceiling is clamped to the context
with a stderr note). `CTX`/`--ctx` sets the context window.

What the knobs do (full table in [Environment reference](#environment-reference-v4-engine)):
`COLI_CUDA_ATTN_BATCH=1` puts the batched prefill attention block on the GPU
(compressor/indexer projections, sparse attention on a persistent device KV
ring, wo, mHC) and enables the GPU decode attention/indexer paths;
`COLI_CUDA_MOE_BATCH=1` runs prefill MoE on a transient VRAM expert bank
(route-aware refill, expert-grouped GEMM); `DSV4_CUDA_EXPERT_MIRRORS` is the
decode-side VRAM expert cache; `V4_MOE_REFILL_GROUP` the refill lookup width.
All default to CPU-safe values when unset; the GPU tier itself is on when the
DLL loads (`DSV4_CUDA=0` disables it). A GGUF on the CUDA expert tier
(`COLI_CUDA=1`) is the exception: its dense matrices are already on the card
(`COLI_CUDA_DENSE`), so the batched attention is on there unless
`COLI_CUDA_ATTN_BATCH=0`; a layer without the FP8 `wq_a` mirror takes the
device the engine opened.

## VRAM budget and per-card settings

What the tier keeps in VRAM (measured on a 16 GB RTX 5080): dense/attention
mirrors **6.3 GB** (needed for GPU attention and the GPU dense matvecs in
decode), the transient prefill expert bank (`COLI_CUDA_MOE_BATCH`, freed when
decode starts; 12.75 MiB per slot, 13.5 MiB on DeepGEMM builds: **3.2 GiB**
for 256 experts, 3.4 GiB on DeepGEMM), attention/KV/work buffers **~0.5 GB**,
and the rest as decode expert mirrors at **12.75 MiB each** (gate, up and
down in fp4 with their scales).
`DSV4_CUDA_EXPERT_MIRRORS` is only an upper bound (default 4096): the cache
grows while at least `DSV4_CUDA_VRAM_RESERVE_MB` stays free, so free VRAM
sizes it at run time — you normally do not need to set it. Decode is
disk-bound, so more mirrors help in proportion to the hit rate they add
(~11 000 experts total; 600 mirrors ≈ 5 %); more RAM (`--ram`) buys more than
more VRAM.

The default reserve is 600 MiB without the bank. With `COLI_CUDA_MOE_BATCH=1`
it is sized from the model's bank at engine open, so the next large prefill
still finds room for it after decode has filled the card: 600 MiB + the bank
(two banks with `COLI_CUDA_MOE_DOUBLE=1`, back to one once its prefetch has
turned itself off; a second bank that did not fit is retried by the next
prefill, so its room stays reserved) + 64 mirrors, because free VRAM is
re-read only every 64 misses. That is 4681 MiB for 256 experts on generic
builds, 4873 on DeepGEMM and 3100 for a 132-expert checkpoint (7945, 8329 and
4783 with the second bank); the engine logs it as `v4_gpu mirror-reserve=`.
Below 32 live mirrors the default keeps only 600 MiB free, so a card too small
for the bank still decodes from a working mirror set (the bank then falls back
to the CPU union). An explicit `DSV4_CUDA_VRAM_RESERVE_MB` is used as given.
It and the 600 MiB default without the bank carry no probe margin, so from 8
live mirrors on the cache counts the mirrors it has added since the last read
of free VRAM against that read and stops at the reserve instead of
overshooting it by up to 64 mirrors. The first 8 may still overshoot it by
about 100 MiB, so a reserve just under the free VRAM leaves a mirror set the
cache can recycle. The bank-sized default already counts those 64 mirrors in
its reserve; below 32 live mirrors it may still grow up to 64 mirrors past its
600 MiB between two reads. A failed mirror upload counts as a full card until
the next read.

| VRAM | dense mirrors | prefill bank | recommended env | expected |
|---|---|---|---|---|
| 6 GB | no (6.3 GB does not fit) | yes (3.2 GiB) | `CUDA_DENSE=0 COLI_CUDA_MOE_BATCH=1 COLI_CUDA_ATTN_BATCH=0` | GPU MoE prefill (generic kernels on non-RTX50), CPU attention/decode; ~90 mirrors between prefills |
| 8 GB | yes | no (~0.9 GiB left) | `CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=0` | GPU attention + GPU decode dense; MoE prefill on the CPU union; ~25 mirrors |
| 10–12 GB | yes | 12 GB: yes (tight); 10 GB: only a pruned checkpoint's smaller bank on generic builds | `CUDA_DENSE=1 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1` | full GPU prefill; 12 GB: the 32-mirror minimum beside a 256-expert bank's reserve (~350 mirrors with `COLI_CUDA_MOE_BATCH=0`), ~150 beside a 132-expert one; 10 GB: the 32-mirror minimum beside a 132-expert bank (256 experts: `COLI_CUDA_MOE_BATCH=0`, ~190 mirrors) |
| 16 GB (this box) | yes | yes | the canonical line above (mirror cap can be left unset) | ~330 mirrors; the numbers in this document predate the bank-sized reserve (the fixed 2800 MiB default left more mirrors, but too little room for the bank once decode had filled the card) |
| 24 GB | yes | yes | canonical | ~1 000 mirrors (~9 %) |
| 32 GB | yes | yes | canonical | ~1 600 mirrors (~15 %); raise `--ram` first if RAM allows |

Mirror counts are estimates for a card of N × 1024 MiB and a 256-expert model
unless noted (DeepGEMM bank on RTX 50, generic elsewhere): free VRAM after the
dense mirrors (6.3 GiB; none in the 6 GB row) and ~0.8 GiB of CUDA context and
attention/KV buffers (~0.3 GiB with CPU attention), minus the default reserve,
divided by 12.75 MiB. `COLI_CUDA_MOE_DOUBLE=1` reserves one more bank (~260
mirrors fewer for 256 experts).

On non-Blackwell cards the generic DLL is selected automatically (fp32
kernels): same settings, prefill roughly 2–3× slower than the DeepGEMM
numbers, still far ahead of CPU.

**Give it RAM.** 43 × 256 routed experts are ~137 GiB on disk and a token
touches 301 of them, so the expert cache hit rate is what sets tok/s: `--ram`
is the single most valuable knob, and it changes speed only, never output.

A serve reads the cache full before it says ready: with RAM for every expert the
first request runs at the warm speed instead of climbing as the cache fills one
miss at a time (#2109), and the start takes as long as the disk needs to read
the cache. `COLI_V4_PREWARM=0` starts with it empty.

Two opt-in GPU levers are looking for community numbers, both default off and
byte-identical when unset: `DSV4_HYBRID=1` splits VRAM-tier misses between the
GPU fill branch and the CPU branch using bandwidths measured at runtime, and
`COLI_CUDA_MOE_DOUBLE=1` (on top of `COLI_CUDA_MOE_BATCH=1`) prefetches the next
layer's full expert set into a second VRAM bank while the current layer
computes, falling back to the single bank when VRAM is short.

### Expert-pruned checkpoints (REAP) and the generic kernels

Expert-pruned checkpoints such as `puwaer/DeepSeek-V4-Flash-0731-reap-150b`
(132 routed experts instead of 256) use the batched prefill MoE path too: the
CPU router picks their experts (tokens in parallel, bit-identical to the CPU
union; Vulkan builds take the chunk's gate logits from one device call, as the
union does) because the GPU top-6 kernel assumes 256 logits, and on generic
(non-DeepGEMM) builds the bank is sized to the model's expert count
(132 slots, 1683 MiB, instead of 256 slots, 3264 MiB). DeepGEMM builds keep a
256-slot bank.

The generic kernels themselves got two bit-identical rewrites (same operands
in the same order for every output; `_VERIFY=1` runs old and new side by side,
keeps the old kernel's results and counts mismatching bits; it does that even
when the switch below is `0`, and says so once at startup):

- `DSV4_MOE_EXPERT_MAJOR` (1): the routed experts run expert-major — one warp
  per expert row walks every token of the chunk that picked that expert, so
  each expert's weights are read once per chunk instead of once per route.
  It takes 2 routes per pass while the chunk averages fewer than 4 routes per
  routed expert (small prompts, tail chunks) and 4 otherwise.
- `DSV4_MM_WARP` (auto): warp-per-row dense batch matmul for fp8 shapes whose
  I is a multiple of 256: short rows only (I ≤ 1024, e.g. `attn.wq_b`; unset,
  empty or `auto`), all of them with `1`, none with `0`. Against the old
  kernel it also measured faster for I = 2048 (~0.8× the time on the
  shared-expert w2, 4096×2048), but slower for I = 4096 (~1.1×) and I = 8192
  (~1.65×).

Measured on 2× RTX 3090 (one used; sm_86, CUDA 12.0 `NO_TC=1`), Ryzen 9 9950X3D,
62 GB RAM (`--ram 40`), one SN850X, REAP 150B, greedy, `--ctx 8192`, a fresh
server per run (the model's `.coli_usage` was empty, so the serve prewarm read
nothing), medians of three runs per cell, against upstream dev 736a0682:

| prompt | CPU union | GPU batch | + `COLI_CUDA_MOE_DOUBLE=1` |
|---|---|---|---|
| 1,992 tokens | 130.2 s | 76.8 s (1.70×) | **64.1 s** (2.03×) |
| 3,005 tokens | 205.9 s | 111.0 s (1.86×) | **97.8 s** (2.10×) |

The kernels' share, measured earlier on 7b55a9b3 (before upstream's faster CPU
decode kernels, #2069, and the serve prewarm, #2109), one run per cell:

| prompt | CPU union (before) | GPU batch, per-route kernels | + expert-major + warp dense | + `COLI_CUDA_MOE_DOUBLE=1` |
|---|---|---|---|---|
| 1,992 tokens | 220.5 s | 125.4 s | 79.7 s | **69.1 s** |
| ~3,000 tokens | 334.4 s\* | 183.5 s | 118.2 s | **105.8 s** |

\* Not measured on upstream: this request ran on the CPU union of a patched
build whose bank had failed to allocate, at upstream's per-token rate
(0.111 s/token, as in the 1,992-token cell).

Answers were identical to the CPU union in every run here. That is not
guaranteed for every prompt: the routing is the same, but the GPU and CPU
expert arithmetic is not bit-identical, so a near-tie can flip a token. Decode (disk-bound) is
unchanged after short prompts, which stay below `COLI_CUDA_MOE_BATCH_MIN` and
never use the bank, but after these two GPU-batch prefills it measured slower
than after a CPU-union prefill: the 24-token answer to the 1,992-token prompt
at 2.18 tok/s instead of 2.42 (~10 %), the answer to the ~3,000-token prompt
at ~2.75 instead of 2.85 (~3 %). Those runs predate the bank keeping the RAM
expert store's routing books. Measured again on one binary and one VRAM
reserve, with the same answer text after either prefill, the 1,992-token case
was 7.4 % slower (2.13 vs 2.30 tok/s), and the cause was the store's books: the
bank reads each routed expert once per layer per prompt and those reads were
all the store counted, so decode started with every RAM-resident expert at
usage 1, the pins ranked by expert id and only 7-9 of 16 pins active, and took
~57 % more RAM-cache misses, most of them in its first ten tokens. The bank now
counts each chunk's routed union as the CPU union's lookups do (usage, the pin
ramp, repins, the `.coli_usage` history) and keeps the members already in RAM
recent, without extra reads. Measured again on the series with and without that
change (the same answer text, five cold runs each), decode after the 1,992-token
bank prefill went from 2.13 to 2.26 tok/s (+6.3 %, medians), against 2.30 after
a CPU-union prefill: 67 % (means) to 79 % (medians) of the gap recovered. The
bank prefill takes as long as before: 82 s to the first token with or without
the change, against 216 s for the CPU union (medians of these later runs, so
not quite the table's single-run times). The rest of the gap sits in the first
~5 decode tokens and comes from which experts are resident at decode start. For
one, the bank reads each routed expert once per layer, so a member the RAM cache
evicted earlier in the layer is not read back as the CPU union's per-chunk
lookups read it. The books also decide which experts a later request's prefill
(or a long prompt's later segments) finds in RAM, so a table cell whose request
followed another may move as well; that has not been measured.
On 736a0682 (the medians above), decode after the bank prefill was 3.2 %
(1,992 tokens) and 1.6 % (3,005) behind decode after the CPU union's; the
bank's share of the VRAM reserve (below) leaves room for fewer decode mirrors.

With `COLI_CUDA_MOE_DOUBLE=1` the default VRAM
reserve also covers the prefetch bank (one more bank: +1683 MiB for this
132-expert model, +3264 MiB for 256 experts), so it still allocates after
decode has grown its mirror cache; the mirrors that room holds back cost some
decode speed, and the reserve drops back to one bank once the prefetch has
turned itself off (`moe-double=off`).

### Decode on two GPUs (peer mirror pool) and VRAM-first lookups

Decode mirrors routed experts in VRAM, but a mirror used to be an upload
cache only: every routed expert was still fetched from the RAM expert store
(and the SSD on a miss) before its mirror was found. Two opt-in levers turn
VRAM into a real tier:

- `V4_GPU_RESIDENT_FIRST=1`: an expert already mirrored in VRAM is computed
  from the mirror without asking the store. The store still records the
  routing (the `.coli_usage` history, the dashboard's HITS and heat), and its
  RAM pin ranking leaves mirrored experts out, so their RAM slots are no
  longer pinned or refreshed, age out of the LRU, and the RAM tier holds
  experts that are not in VRAM. A routing served from VRAM is not a store
  request: the hit rate in `DONE`/`PROF` is then the RAM hit rate of the
  experts the store was asked for. If a token cannot use the fused GPU path
  after all (an attach failed: a failed upload, or a rows16-packed RAM slot),
  its mirrored experts are read from the store through the loader lanes and
  the token runs on the CPU, as without the switch.
- `DSV4_CUDA_PEER_DEVICE=N`: a second mirror pool on GPU N (needs peer
  access, e.g. NVLink). By default (`DSV4_CUDA_PEER_COMPUTE=1`) the fused
  decode MoE runs the experts mirrored there ON that GPU: the token's input
  is copied over P2P, the same kernels run on local weights, and each
  expert's output row is written into the primary GPU's reduce buffer.
  Results are bit-identical to computing every expert on one GPU; peer
  compute needs both GPUs at the same SM version (the same machine code).
  Mirrors are read in place over P2P from the other card only with
  `DSV4_CUDA_PEER_COMPUTE=0`, with `DSV4_CUDA_BATCHED=0` (kernels launched
  expert by expert on the primary GPU; `DSV4_CUDA_PEER_COMPUTE` has no effect
  there), with GPUs of different SM versions, and when the fused MoE cannot
  take the token (no GPU mirror of the shared expert, which is always the
  case for a GGUF checkpoint, or a refused call) and the expert-group
  fallback computes it. The `v4_gpu peer-mirrors ... mode=` line at startup
  says which (`compute` or `p2p-read`); for a GGUF it says `compute`, but
  decode reads the pool over P2P.

GPU N is checked before anything is set up on it (a `DSV4_CUDA_PEER_MIRRORS`
of 8 or more, a visible device other than the primary that no earlier peer
failure in the process has latched off, peer access both ways, and on both
GPUs, the primary's first, an architecture and kernel image the build runs);
when a check fails, GPU N is left untouched, decode runs without it, and the
`peer mirror pool off:` warning gives the reason and, for a missing kernel
image, the build to use (a `CUDA_ARCH`, or the generic build for a DeepGEMM
binary; a GPU that is full or taken by an exclusive-process job is reported
as such). If the two-GPU init fails after the checks passed, GPU N's context
is dropped again and decode retries on the primary alone. A miss grows the
primary cache, then the pool, and once neither can grow recycles the older of
the two LRU mirrors; a cache with fewer than 8 live mirrors (a nearly full
card, `DSV4_CUDA_PEER_RESERVE_MB` above its free VRAM) is never recycled, so
a starved pool only adds mirrors and never sends an expert to the CPU that
the primary cache would have taken. A CUDA failure on the peer during decode
latches it off for the rest of the run, with one
`[DSV4 CUDA] peer device N failed in the decode MoE (...)` line. The token in
flight is recomputed on the primary GPU, or on the CPU from host copies when
the peer no longer answers, and the pool's mirrors count as misses from then
on. That holds for an error the devices survive (an allocation or launch
failure). A kernel fault (launch failure, illegal address, trap) is sticky,
and with peer access between the GPUs it takes the primary's context with it
too: measured on two RTX 3090s with NVLink, a `__trap()` on the peer left
device 0's next calls failing, whether device 0's stream or the host waited
for the peer. So a sticky error anywhere in the backend latches CUDA off for
the whole process, with one `[DSV4 CUDA] ...: CUDA is lost for this process`
line: every backend call is refused from then on without touching CUDA, the
token in flight and every later one take host copies (the CPU paths, as
without `CUDA=1`), and the server keeps answering, more slowly, until it is
restarted. The latch covers peer compute (`mode=compute`); a peer that stops
answering without reporting an error is not detected. The pool's VRAM on that
GPU is released only when the engine closes.

The peer card holds nothing but mirrors (no dense weights, no prefill bank),
at the same 12.75 MiB each. After ~0.3 GiB of CUDA context and the 1024 MiB
`DSV4_CUDA_PEER_RESERVE_MB`, a 24 GB card holds ~1 800 (~16 % of a
256-expert model's ~11 000 experts, ~32 % of a 132-expert one's 5 676), a
16 GB card ~1 200. `DSV4_CUDA_PEER_MIRRORS` (4096, i.e. 51 GiB) is only an
upper bound, like `DSV4_CUDA_EXPERT_MIRRORS`: free VRAM sizes the pool.

Both levers act where the MoE runs one token at a time, i.e. mainly the
ordinary decode step. Decode batches of more than one token (the verification
of speculative drafts, and the steps of a serve that decodes several
conversations together, `KV_SLOTS`) go through the expert union and use
neither (with the default `V4_EXPERT_UNION=1`; with `0` they run token by
token and use both). The RAM pin ranking still leaves mirrored experts out
on those paths and during prefill, as long as `V4_GPU_RESIDENT_FIRST=1` is
set. `DSV4_HYBRID=1` turns `V4_GPU_RESIDENT_FIRST` off and
uploads its split's experts into the primary cache only, so no peer pool is
set up there: GPU N is left untouched, and a warning says so.

Measured on the REAP 150B setup above (2× RTX 3090 with an NVLink bridge,
`--ram 40`, six 300-token answers per run, greedy), with `COLI_CUDA_MOE_BATCH=1`
set on every row, against upstream dev 736a0682: a fresh server per run, three
runs per row interleaved, medians of each run's mean decode tok/s. Every run
gave the same answers as dev's (compared in full).

| configuration | decode tok/s |
|---|---|
| upstream dev | 3.01 |
| this series, every switch off | 2.98 |
| + `V4_GPU_RESIDENT_FIRST=1` | 3.07 |
| + `DSV4_CUDA_PEER_DEVICE=1` (peer compute) | **4.29** |

The second row is below dev because, with `COLI_CUDA_MOE_BATCH=1`, GPU0's
mirror reserve keeps room for the prefill bank (3100 MiB for this model),
which dev does not reserve for a 132-expert model, so fewer decode mirrors fit;
a decode-only server can leave `COLI_CUDA_MOE_BATCH` unset or lower
`DSV4_CUDA_VRAM_RESERVE_MB`. `DSV4_CUDA_PEER_COMPUTE=0` (the peer mirrors read
in place over NVLink) was not measured on this base: in one earlier run per
row it landed between the VRAM-first and peer-compute rows.

Prefill is unchanged (it runs on the primary GPU's bank). On this box the
host→GPU uploads run at ~12 GB/s (PCIe 4.0 x8 per card), so an expert read
from the peer pool beats an upload from RAM by a wide margin.
`V4_SERVE_STATS_LOG=1` prints two lines per turn on stderr. `v4_serve_turn`
repeats the turn's `DONE`/`PROF` numbers: `decode_s` and `tok_s` are the
decode's, the hit rate and the expert and attention seconds cover the whole
turn, prefill included; `vram_first_hits_total` and `store_fetches_total` are
process totals of the decode's routed experts computed from VRAM without a
store lookup and fetched from the store. `v4_dec_moe` gives process totals of
the single-token decode MoE (decode forwards only: prefill chunks, batched
speculative verify and the `KV_SLOTS` multiplexed decode are not counted):
token-layers (`calls`), their wall time from routing on, the wait for store
fetches, the VRAM attach time and the mirror `uploads` it caused, and the
fused GPU compute.

## Prefill: segments, chunks, checkpoints

- **Chunks/segments.** Prefill runs layer-major over 128-token chunks
  (`V4_PREFILL_CHUNK`) inside 4096-token segments (`V4_PREFILL_SEGMENT`).
  Segments are atomic; the client-cancel poll runs between them and completed
  segments are recorded, so an identical retry resumes instead of restarting.
  Each segment re-sweeps every layer's routed experts through the transient
  bank, so fewer/larger segments = fewer disk sweeps, at the price of cancel
  latency (~2.5 min worst case at 4096). Progress lines `v4_prefill N/M tokens`
  print per segment on multi-segment prompts.
- **Prefix checkpoints** (`V4_PREFIX_CKPT`, min length `V4_PREFIX_CKPT_MIN`=512).
  The window/compressor/indexer state cannot rewind, so reuse across
  conversations needs a snapshot at the shared boundary. Three capture rules:
  (1) the gateway tells the engine where the rendered system turn ends
  (optional 8th `SUBMIT` header field) → snapshot there on the very first
  request; (2) fallback: the longest common prefix of two successive fresh
  prompts; (3) **prompt-end** snapshot after every prefill, because agent
  clients re-render the assistant reply, so strict "extends everything fed"
  reuse fails at the reply boundary. `V4_PREFIX_CKPT_SLOTS` (4) in-memory
  slots, LRU with prompt-end evicted first. Log lines: `v4_ckpt store
  prefix=N` / `prompt_end=N`, `v4_ckpt hit prefix=N`.
- **Persistence.** Prefix captures are written to `<model>/.coli_ckpt/`
  (`V4_PREFIX_CKPT_DISK`=1; `2` also persists prompt-end captures, `0` off;
  ~140 MB for an 8.3k-token prefix, config-fingerprinted, temp+rename) and
  loaded lazily on the first request after a restart — a fresh serve no
  longer re-prefills the system prompt.

## Performance (RTX 5080 16 GB, 2× NVMe mirror, 32 GB RAM, i9-class CPU)

| scenario | before this tier | now |
|---|---|---|
| 826-token prefill (cold) | 578 s (CPU) | ~40 s |
| 3324-token prefill (cold) | 271 s (early GPU) | **90 s** (DeepGEMM), 225 s (generic DLL) |
| 8.3k opencode system prompt, first turn after start | ~600 s | **~4 min** — once per model (checkpoint on disk) |
| every later session with the same system prompt | 300 s | **6–9 s** |
| follow-up turn in a conversation (tool result) | 677 s | **6 s** |
| decode at 3.3k context | 0.6 tok/s | **1.5–1.6 tok/s** |

Where the time goes now: prefill ≈ 35 % expert-bank refill (disk-bound,
~6 GB/s across two NVMe), ≈ 35 % attention block (many small GPU stages +
host↔device syncs), ≈ 5 % MoE GEMM, rest indexer/dense/mHC. Decode ≈ 75 %
routed-expert reads (~230 × 12.6 MB per token from disk at 6 % VRAM hit
rate), rest GPU stages. More RAM/VRAM (higher expert hit rate) is the only
lever left below the disk limit — which is what a second card is for (next
section): it adds VRAM for decode mirrors, not FLOPs.

### Second GPU: more decode mirrors (`DSV4_CUDA_DEVICE2`)

With two NVIDIA cards the engine can host a second expert-mirror cache on
the other one (`DSV4_CUDA_DEVICE2=auto` = the first other device the runtime
lists, or an ordinal; unset = one card — opt-in, see the measurement at the
end of this section). The primary
card keeps everything else — dense mirrors, attention, the prefill bank, the
hot set of mirrors that churns — and the second card is the overflow that
fills once: a miss grows the primary while its VRAM reserve allows, then the
second card, then recycles the primary's LRU slot. At decode the routed
experts split by residence: the primary's subset runs its fused MoE/expert
group, the second card's subset its own expert group, the rest the CPU, and
the partial sums add (the same order-of-addition class as the hybrid split).
Mixed generations work: the two DLL flavours load side by side on Windows —
an RTX 50 runs `coli_cuda_dsv4_dg.dll`, an RTX 30/40 beside it gets
`coli_cuda_dsv4.dll` as the *secondary backend*, chosen by
`dsv4_cuda_backend_arch_ok` per device (`COLI_DSV4_DLL2` forces the file).
On Linux one flavour is linked, so both cards must run it (generic build:
any sm_80+). Banner: `v4_gpu second-device=1 backend=generic mirrors<=4096
reserve=600MB`; at exit `v4_gpu second-device=1 mirrors=N hits=H (primary
mirrors=M hits=… misses=…)`. `DSV4_CUDA_EXPERT_MIRRORS2` /
`DSV4_CUDA_VRAM_RESERVE2_MB` (600) bound it; there is no prefill bank to
reserve for there.

Placement with two cards: a miss grows the primary while its VRAM reserve
allows, then the second card; once both are full the recycled uploads are
split between the two cards by `V4_DEV2_SPLIT` (percent to the second card,
default 50), so one layer's uploads ride both PCIe links at once, and the two
expert groups of a layer run concurrently (the second card's on a helper
thread). The decode attach itself runs on one uploader thread per card
(`V4_UPLOAD_THREADS=1`, on): the lane loop hands each missed expert over and
keeps consuming reads, the uploader copies on the backend's aux stream, and
the main thread joins before the partition — `DSV4_DECODE_PROF` shows the
join as `attach`. Exit report: `v4_gpu uploader[i] jobs=… avg=… ms`.

Measured, first wiring (RTX 5080 16 GB + RTX 3080 10 GB, 3278-token prompt,
48 greedy tokens, v1.10.1): the second card held 706 mirrors and 54 % of
attaches hit a mirror, yet decode went **0.79 → 0.90 s/token** — a hit only
saved the upload (the record was still read into host memory) and the second
group was one more synchronous launch per layer. With mirror-first, striped
reads, the split uploads and the threaded groups (same box, both cards at
PCIe gen4 x8 since the second card halves the primary's lanes): **parity**.
At 826-token context (39 tokens) MoE time per token went 346 → 311 ms
(reads 207 → 147 ms: the extra mirrors) but decode 21.95 → 21.82 s; at
3.3k context 19.96 → 20.70 s. The per-layer critical path is one expert
read (~3 ms) plus the *last* expert's upload (1.1 ms on the 5080, 2.0 ms on
the 3080) plus the launch round trips, and a second card shortens none of
them: it trades bandwidth, which is not the bottleneck, for one more sync.
Kept opt-in; it does cut disk traffic by ~30 % if that matters. The lever
that would pay on such a box is the primary's link — a second card in the
other CPU slot costs the primary half its lanes (x16 → x8, uploads 2× slower).

Second measurement (2026-10-28, after the decode-attention work): the
parity had two causes that are now fixed or understood.

1. The second card's group was launched at the partition, after this
   layer's reads and uploads, so its whole runtime (~1 ms per layer on the
   3080's generic kernels) sat on the critical path. Its experts are known
   to be resident there at routing time (nothing to read), so `V4_DEV2_EARLY`
   (on) launches the group right after routing, on its thread, and joins it
   once the reads and uploads are done: at 3.3k context, 36 tokens, MoE
   341 → 319 ms/token.
2. Long decodes ran the *primary* out of memory after ~50 tokens
   (`[DSV4 CUDA] weight allocation: out of memory`, `refill ... upload: out
   of memory`, then `block computation failed`) with 1.2 GB of VRAM still
   free. The cause is Windows commit charge, not VRAM: the sampled free
   commit sat at 0.3–0.7 GB for the whole run while the system-managed page
   file grew from 43 to 77 GB, and the WDDM driver backs device allocations
   with system commit. The second card's 8.7 GB of mirrors is that much more
   commit on a 32 GB box. With `--ram 16` (the planner's budget; the RAM
   expert cache shrinks from 8.6 to 7.5 GiB) 300 tokens ran clean.

   Steady state (last 100 of 300 tokens, 3.3k context, `--ram 16` on both):
   one card 376 ms/token (MoE 294), two cards **331 ms/token** (MoE 249:
   reads 160 → 134, join 92 → 82, the second card's ~45 ms of compute
   hidden under the loads) — **−12 %**. serve.ps1 on the reference box runs
   `DSV4_CUDA_DEVICE2=auto` with `--ram 16` for this reason.

   Where the rest goes: an expert trace (`V4_EXPERT_TRACE=1`, one line per
   decode layer with each routed expert's VRAM/RAM state; simulator
   `tools/v4_expert_cache_sim.py`) over 300 tokens shows the policy is
   not the lever — LFU with decay equals LRU (52.7 % vs 51.4 % at 688
   mirrors), a static hot set is worse (44 %) — only capacity is: LRU hits
   53 % at 688 mirrors, 64 % at 1200, 71 % at 1600, 80 % at 2400; the
   primary + the second card behave as LRU over their sum (misses 47 → 32 %).
   Decode is bound by the bytes that cross into the primary (~1.7 GB per
   token at 688 mirrors: 131 misses × 12.75 MiB, ≈150 ms on a gen4 x8 link)
   and the same bytes off the two NVMe, so each further mirror pays in
   proportion; the 6.3 GB of dense/attention mirrors on the primary are the
   remaining VRAM to trade. (Moving the attention mirrors to the second
   card is not a lever: its VRAM is already full of expert mirrors, and the
   two cards behave as one LRU over their sum.)

Third measurement (2026-10-28, later): what was left per decode layer after
the reads was a host-side tail — the join on the last expert's upload and a
~1.1 ms expert-group round trip. `DSV4_MOE_KPROF` showed the group's
*kernels* were the 1.1 ms: `mv_fp4_grouped` / `mv<8>` read one byte per
lane per step and decode through table lookups, ~15 % of the card's
bandwidth for the ~100 MB a layer's six experts plus the shared expert
weigh. Two changes:

1. `V4_UPLOAD_FENCE` (on): the uploader threads only enqueue their copies
   on the card's aux stream; the expert group records an event there and
   makes its compute stream wait on it (`dsv4_cuda_group_fence`), so the
   host neither joins the DMA nor launches after it. The group's small
   copies (x, descriptors + weights in one transfer, y) go through pinned
   staging (`DSV4_MOE_PINNED`). One card, 36 tokens: MoE 327 → 318 ms/token.
2. `DSV4_MOE_MV` (on): bandwidth-shaped decode matvecs — one warp per
   row, 16-byte weight loads, the input vector in padded shared memory,
   bit-arithmetic fp4/fp8/UE8M0 decode, shuffle reduction. Group kernels
   1.14 → 0.29 ms per layer on the 5080 (0.71 → 0.40 on the 3080). Its
   summation order differs from the old kernels: `DSV4_MOE_MV_VERIFY=1`
   compares every fused call against them (36 tokens: max |Δ| 0.0625 = one
   bf16 ulp at the output's magnitude, 0.1 % of elements one ulp off); the
   36-token greedy texts are unchanged on both configurations, the
   300-token ones differ as any reordering does.

   Steady state (last 100 of 300 tokens, 3.3k context, `--ram 16`): one
   card **376 → 327 ms/token** (MoE 294 → 246), two cards **331 → 306**
   (MoE 249 → 226); 300 tokens after the first: 131.4 → 117.2 s and
   120.0 → 111.6 s. What remains per layer on two cards (~5.2 ms of MoE):
   reads 2.7 (NVMe-bound: the 9100 PRO on C: and the 990 PRO on D: read
   6.4 and 5.4 GB/s here, 10 GB/s striped), upload tail 1.1, group 1.0
   (0.17 ms of kernels; the rest is the WDDM round trip), shared 0.25.

Fourth measurement (2026-10-28, night): the uploader's exit report now
breaks a queued upload down (`v4_gpu uploader breakdown per job`): of its
0.90 ms, 0.66 ms were the six enqueue-only `cudaMemcpyAsync` of the three
mirror tensors (~110 µs each under WDDM, serialized on the card's one
uploader thread), the lock and the LRU pick nothing. `V4_MIRROR_BLOCK`
(on): a mirror is ONE device allocation — the record's scale run, then its
weight run 16-byte aligned — refilled with two copies and re-pointed, so an
expert's tensor order (it varies by shard; the file also keeps a ~3.6 KB
gap between an expert's scales and its weights) never matters: uploader
0.90 → 0.23 ms per job, the join (`attach`) 63 → 4 ms/token on one card.
What it buys depends on the card count: on **two cards** the two uploader
threads were the tail, 36 tokens MoE 293 → 270 ms/token; on **one card**
the time moved into the group's fence wait (`compute` 38 → 87) and MoE
stayed at 278 — there the tail is the link itself: a layer's 2–3 misses
finish reading together and their ~33 MB then cross the gen4 x8 link in
~3 ms, of which ~1.7 ms sit after the last read. Steady state (last 100 of
300 tokens, 3.3k context, `--ram 16`): one card 327 → 329 ms/token (unchanged within noise; `attach` 53 → 3, `compute` 38 → 84: the wait moved to the fence), two cards
306 → 306 (second card 651 mirrors).

Fifth measurement (2026-10-29): `V4_UPLOAD_PIECES` (4, on): a decode lane
reserves its expert's mirror BEFORE the read (the LRU block mirror of the
card the split picks, or a grown one; the entry is `pending`, neither a hit
nor a victim, until the last piece is in) and reads the 12 MiB weight window
in four pieces, all in flight at once (whole pieces alternate between the
two replicas; halving each piece across them made the reads slower), each
reported as it lands; the lane enqueues the piece's copy into the mirror's
block on the card's aux stream at once, so the link works under the read
instead of after it. The scale run (the buffered read) is one more piece.
The main thread then finds the expert mirrored (`coli_v4_gpu_expert_peek`)
and never queues an upload for it; the expert group fences the aux stream
as it does for the uploader's copies. Anything short of a complete report
(a fallback read, a copy that failed) abandons the entry — it stays empty
and is recycled first — and the expert takes the old path. Only inside the
decode step's lane window: the prefill's lanes never reserve mirrors. Exit
line: `v4_gpu pieces: reserved= mirrored= abandoned= copies= enqueue=`.

   Gotcha found on the way: the first copy from a slab page-locks its whole
   region (`cudaHostRegister`), and when that happened while the window's
   other pieces were still landing from the drive, those pieces came up
   with stale bytes (WDDM moves the pages as it locks them; the drive's DMA
   had the old frames). One read in a hundred of the first tokens — each
   slab's first use — carried wrong weights, enough to flip near-tie tokens
   in half the runs, while every mirror matched its slab (the slab itself
   was wrong). The lane now pins the slab before its first piece read
   (`dsv4_cuda_host_pin`); `V4_PIECES_VERIFY=8` re-reads every record with
   the buffered path and compares the slab, `=1` reads each piece-built
   mirror back and compares it with the slab.
   Measured (3.3k context, `--ram 16`): 36 tokens, one card MoE 287 -> 232 ms/token (`loads` 175 -> 175, `compute` 83 -> 30: the fence wait is gone and the reads did not slow down; 2 pieces 237, 8 pieces slower), two cards 270 -> 215. Steady state (last 100 of 300 tokens): one card 329 -> 285 ms/token (`compute` 84 -> 34), two cards 306 -> 260 (second card 651 mirrors, 43 ms of its group under the loads). The lane's enqueue costs 0.09 ms per expert. The 36- and 300-token one-card texts are unchanged; the two-card 300-token text differs because the experts now split between the cards in a different order (the same class of difference the split already had).

Sixth measurement (2026-10-29): `DSV4_ATTN_MV` (1): the decode attention
chain's three fp8 matvecs (wq_b, the grouped wo_a, wo_b: 32 MB each) on
the bandwidth-shaped `mv_fp8_dec` (a warp per row, 16-byte loads, the
activation in padded shared memory; it gained a group dimension and the
fmt-9 per-element rounding) instead of the byte-per-lane `mv` kernel
(`DSV4_WO_KPROF`: wo_a and wo_b 0.137 → 0.05 ms each, 245 → 650 GB/s on
the 5080). Not bitwise the old kernel (a different order of summation:
`DSV4_WO_VERIFY=1` shows a handful of 1-ulp elements per layer), so the
greedy text can differ at near ties; `DSV4_ATTN_MV=0` restores the exact
text at the old speed. Measured (3.3k context, `--ram 16`): attention 55 -> 43 ms/token (`sattn` 33 -> 20); steady state (last 100 of 300 tokens) one card 285 -> 265 ms/token, two cards 260 -> 248.

Seventh measurement (2026-10-29): `DSV4_SATTN_V2` (1): the cached sparse
attention kernel (the decode chain's and the batched prefill's, with and
without the indexer's selection) rewritten for latency with every
summation in the same order — four rows per warp step for the scores, the
max as a block reduction, the exponentials by every thread with the
denominator still summed by one thread in row order, the selection
indices and sixteen rows prefetched into registers for the row-ordered
weighted sum, 512 threads per block. The old kernel spent its time in one
thread's serial softmax and in a dependent index load before every row
load: `DSV4_WO_KPROF` (which now also times the query projection, the
uploads and this kernel) showed 0.39 ms per 640-row indexer layer and
0.10 per 153-row layer for ~40 MFLOP. Bitwise the old kernel (the greedy
text is unchanged). Measured (3.3k context, `--ram 16`): attention 43 -> 37 ms/token; steady state (last 100 of 300 tokens) one card 265 -> 260 ms/token, two cards 248 -> 244.

   Eighth measurement (2026-10-29): `DSV4_SATTN_HPB` (4): from 256 rows up
   a block serves four heads, loading each row once for their four dot
   products and four weighted sums (the rows are the one latent KV per
   position, shared by every head; each head's arithmetic keeps its
   order, so still bitwise). The 640-row layers were at the card's DRAM
   limit — 64 one-head blocks re-reading the same rows twice, 168 MB in
   0.20 ms: now 0.125 ms. Shorter layers keep one head per block (fewer
   blocks under-fill the card), as does any shape whose four score rows
   would pass 48 KB of shared memory. The batched prefill shares the
   kernel: time to first token 62.4 -> 60.9 s at 3.3k context; decode attention 37 -> 36 ms/token, steady state one card 260 -> 259 ms/token, two cards 244 (unchanged).

Ninth measurement (2026-10-29): `V4_PREFILL_CHAIN` (1): the batched
prefill attention's tail as one device chain (`dsv4_cuda_attn_prefill_chain`):
the normalized query ranks go up once; the wq_b GEMM (the batched dense
path the host's matmul already took), the bf16 round, the per-head rmsnorm
and the forward RoPE (the decode chain's kernels, batched over the chunk),
the cached sparse attention, the inverse RoPE and the grouped wo_a + wo_b
follow on the stream; the outputs come down once. Before, per 128-token
chunk, the query came down (16 MB), was normed and RoPE'd on the CPU, went
up again for the attention, and the attended came down for its inverse
RoPE and went up again for wo: `DSV4_ATTN_PROF` summed over a 3.3k prompt
gave qb 7.4 s, rope 1.1 s, attn 7.3 s and wo 3.3 s of 48 s of attention.
A chunk the chain declines (no device attention for it) computes its query
on the host then and takes the usual paths. Same arithmetic in the same
order as the host (the kernels the decode chain verified bitwise). Measured (3.3k prompt): time to first token 60.9 -> 53.3 s; the profiled attention 48.4 -> 36.5 s (qb 7.4 -> 0.3, rope 1.1 -> 0.0, wo 3.3 -> 0.2, attn 7.3 -> 6.0); the greedy text unchanged. What remains of the prefill's attention: the indexer's selection (13.3 s), the sparse attention kernel (6.0 s), the compressor (3.7 s) and the qa projection (2.5 s).

Tenth measurement (2026-10-29): the prefill indexer's 13.8 s (3.3k prompt,
`DSV4_ATTN_PROF` + `DSV4_IDX_KPROF`) split as the projection replica 4.9 s,
the scoring kernel 1.2 s, the host staging of the fused call 1.5 s, the
per-token qsort 1.6 s and the per-token advance 4.3 s. Four fixes, each
bitwise: `DSV4_IDX_COAL` (1), the two kernels read coalesced (each thread
walked its own 4 KB activation row or 512 B key row, so a warp's load split
into 32 cache wavefronts: a transposed copy of the activations, made on the
device per call, and of the resident keys, refreshed on append, puts the
warp's 32 loads in one line; each thread's mul/add sequence is unchanged);
the per-token sorts run across tokens (the comparator is a total order);
`V4_COMP_EMIT_PAR` (100000): the compressor's pool-and-emit no longer forks
an OpenMP region per emitted row (128 or 512 columns of a few microseconds;
the fork cost more; the value is the column count from which it still
forks); the compressor's advance looked its `.ape` tensor up by name on
every token (a strcmp scan over the layer's tensor table): the slot is
cached in the state. The fused call's staging (fp8 qdq of the query ranks,
the input gather, the RoPE tables) runs across tokens as well. Measured (3.3k prompt, 36 tokens): time to first token 53.3 -> 47.4 s; the profiled indexer 13.8 -> 6.4 s (projection kernel 4.9 -> 1.2, scoring 1.2 -> 0.3, sort 1.6 -> 0.1, the compressor's comp 3.8 -> 3.1); the greedy text unchanged. What remains of the indexer: the fused call's host staging and syncs (~1.6 s), the per-token advance (~3 s, the compressor's own state copies and emits), the kernels (1.5 s).

Eleventh measurement (2026-10-29): the prefill attention stage's 5.8 s
(3.3k prompt) against the chain's device time: `DSV4_PF_KPROF=1` (per call
`pfkprof`: uploads, wq_b, the query kernels, the sparse attention, the
inverse RoPE, wo, the download, and the call's host wall) summed to 2.5 s
of device work, the chain's host wall the same; `attnprof` now splits the
stage's host side (`sync=` the KV cache sync, `chain=` the call, `adv=` the
ring append). The KV sync was 1.4 s: the chain's success fell into the
fallback's else branch, which poisons the device KV cache, so every next
chunk-layer reseeded the ring row by row (127 pageable copies) and
re-uploaded the whole compressed prefix; the cache is poisoned only when
no device attention ran. `DSV4_SATTN_HPB_PF` (16): heads per block for the
multi-token launches of the cached sparse attention (the one-token shape's
four heads per block re-read every key row once per four heads; a chunk
has the tokens to fill the card; the per-head arithmetic is unchanged):
the kernel 6.26 -> 5.73 s of stage time. `DSV4_PIN_STAGE` (1): the chain's
seven inputs as one staged copy from a page-locked block and its output
download into it (a pageable cudaMemcpyAsync blocks the host per call on
WDDM): up 273 -> 134 ms, down 576 -> 199 ms. A pinned copy for the kv
appends was tried and dropped: reusing the block safely needs an event
per copy, and each record is a WDDM submit dearer than the pageable copy
(and the event-free variant corrupted the text: the hazard is real). Measured (3.3k prompt, 36 tokens): time to first token 47.4 -> 46.3 s; the KV sync 1.4 -> 0.1 s, the chain's uploads 273 -> 149 ms and its download 576 -> 197 ms; decode unchanged (12.95 s / 36 tokens); the greedy text unchanged. What remains of the stage is the chain's device work (2.5 s: the sparse attention 0.9, wo 0.56, wq_b 0.15, the copies 0.35) and the two dense layers without a chain (0.8 s).

Twelfth measurement (2026-10-29): the prefill block's three hyper-connection
steps (`blockprof` hc1 2.0 s, hc2 4.7 s, hc3 1.8 s over a 3.3k prompt) were
copies, not arithmetic: each step moved the whole four-stream state (8 MB
per chunk-layer) across the link, and the middle step brought the states
down after the attention post only to send them up again for the FFN pre.
`V4_HC_FUSED` (1): the middle step as one device pass through the existing
calls: the post writes a mirror, the FFN pre + norm read it there, and the
states stay on the card (tagged) for the FFN post; a CPU fallback of that
post fetches them first (`coli_v4_gpu_mhc_states_fetch`). The same device
calls in the same order: bitwise. Measured (3.3k prompt, 36 tokens): time to first token 46.3 -> 44.2 s; hc2 4.7 -> 1.3 s (hc1 2.0 and hc3 1.8 s unchanged: the state's upload at the block's start and its download at the end); decode unchanged; the greedy text unchanged. What remains of the three steps is one 8 MB upload and one 8 MB download of the four-stream state per chunk-layer, pageable. Tried and dropped: those two
copies through a page-locked scratch (the engine's parallel memcpy in, then
the DMA) were slower (hc1 2.2 -> 2.6 s, hc3 1.7 -> 2.1 s): the driver's
pageable staging overlaps its copy with the DMA, the scratch serializes
them. Page-locking the session's own state buffers was ruled out: the pin
registry takes a whole allocation region of a lifetime the DLL cannot see.
Keeping a chunk's state on the card across layers would need the whole
prompt's state resident (the prefill is layer-major: every chunk of a
layer before the next layer; ~430 MB at 3.3k): not done.

Thirteenth measurement (2026-10-29): the two compressors' per-token walk
(the attention's, 512 columns, and the indexer's, 128: `comp` 3.1 s and
most of the indexer's advance, ~3 s, over a 3.3k prompt) was the pooling:
every fourth token an emit of 8 rows x columns of expf, serial, 32 times
per chunk. `V4_COMP_BATCH` (1): a chunk's advances at once. Every emit's
eight rows (the previous group's four tokens through the first half of
their projection row, the group's four through the second half) are known
before the chunk starts, from the chunk's batched projections and the four
rows carried in the state, so all the chunk's emits run in one parallel
pass over (emit, column) with the serial emit's expression order per
column, and the state rows are then set as the walk leaves them. `2` =
verify: the walk stays the real path and the batched emits and the final
state are compared against it (`compbatch` lines). Measured (3.3k prompt, 36 tokens): time to first token 44.2 -> 42.8 s; the indexer stage 6.8 -> 5.4 s, the compressor stage 3.3 -> 3.0 s (the batched passes of both compressors are charged to it now); the verify over the whole prompt (546 chunk-layers) reports no mismatch; decode unchanged; the greedy text unchanged. The stage split further
(`attnprof` proj= / cbatch= / comp=): the batched passes of both
compressors 0.26 s, the per-item commits 0.1 s, and 2.2 s in the chunk's
four projection calls (the attention compressor's wkv and wgate, the
indexer's), each uploading the same 2 MB of inputs and syncing. They are
now one multi call (the decode's four-way call, batched: one upload, four
GEMMs of the same kernel, four downloads, one sync); a refusal leaves each
pair to its own call. The qa stage's fp8 activation qdq of the chunk's
inputs (128 x 4096, serial on the CPU) fed only the CPU and Vulkan
fallbacks of wq_a and wkv (the device path takes the raw inputs): it is
computed lazily now, across threads, the first time a fallback needs it. Measured (3.3k prompt, 36 tokens): the projection calls 2.2 -> 1.6 s, the qa stage 2.6 -> 1.5 s (profiled); time to first token 42.8 -> 42.2 s unprofiled (the profiled run 52.8 -> 51.3 s); the greedy text unchanged.

Fourteenth measurement (2026-10-29): the indexer stage's 6.2 s against the
select call's own clocks (3.3 s): the difference is the profile's stderr
lines, which the bench's redirect sends through a pipe to PowerShell at
milliseconds per line; profiled runs are ~10 s slower than unprofiled ones
for that reason, and a stage's figure counts the lines printed inside it.
Judge by the unprofiled time to first token. Inside the fused call the
projection replica was 1.2 s: a block per output row re-read the whole
transposed activation block (512 KB) for its row. `DSV4_IDX_TILE` (1): a
packed tile (8 rows) per block, one 8-byte load per column for the tile's
weight bytes, the activation load shared by the 8 rows' sums (the
transposed block read 1024 times instead of 8192); each (row, token) sum
in its order: bitwise. Measured (3.3k prompt, 36 tokens): the projection kernel 1.21 -> 0.95 s; time to first token 42.25 -> 41.9 s; the greedy text unchanged. The fused call's remainder: its host side ~1.5 s (the quantization and gather across threads, the 3 MB copy into the pinned staging, the upload, the sync), the scoring kernel 0.32 s.

Fifteenth measurement (2026-10-29): the block's two 8 MB state copies per
chunk-layer (hc1 up, hc3 down, 3.8 s per 3.3k prefill, pageable) once
more. The earlier objections were about lifetime (the pin registry takes
a whole allocation region the DLL cannot see freed) and a staging copy
(slower than the driver's pipelined one). `V4_HC_PIN` (1): the prefill
itself page-locks the prompt's range of its two state buffers for the
duration of one prefill (`dsv4_cuda_host_register`, no registry, nothing
in flight at either point) and releases them on every exit; the copies are
then DMA at link speed. Prompts under `V4_HC_PIN_MIN` (256) tokens skip
it. Same bytes: bitwise. Measured (3.3k prompt, 36 tokens): hc1 1.9 -> 1.3 s, hc3 1.8 -> 1.5 s (profiled; the register and release of 2 x 215 MB per prefill included); time to first token 41.9 -> 41.7 s unprofiled; the greedy text unchanged.

Sixteenth measurement (2026-10-29): the MoE stage (5.6 s per 3.3k prefill)
by `V4_MOE_BATCH_PROF=1` (`moebprof`): route 1.2 s (a device kernel and a
host round trip per chunk-layer for the ids), refill 0.6 s (the first
chunk of each layer; the second bank's prefetch covers the rest), compute
3.2 s (the routed and shared kernels, the 2 MB download, the sync). Around
it, the block itself took ten mallocs per chunk-layer (~24 MB: the states,
the normalized and branch arrays, the FFN four), so every upload or
download of them hit fresh pageable pages. `V4_BLOCK_SCRATCH` (1): the ten
arrays from one persistent page-locked scratch (`dsv4_cuda_host_alloc`,
grow-only; a plain malloc when the DLL has none): no fresh-page faults,
and their copies are DMA at link speed. The mHC residency tags are reset
at each block's start, since with stable pointers a tag from the previous
chunk must never hit. Same bytes: bitwise. Measured (3.3k prompt, 36 tokens): time to first token 41.7 -> 40.6 s; the MoE stage 5.6 -> 5.0 s and the attention stage 28.8 -> 27.9 s (profiled); the greedy text unchanged. Tried and dropped: the batched MoE's
route and compute in one device pass (`dsv4_cuda_route_moe_batch`, the
ids fetched on the stream) when the bank holds every expert: 14 ms per
chunk-layer against 4 ms for the router call, the host round trip for
the ids and the second call (the device-ids variant of the grouped MoE
builds its expert layout slowly); time to first token 40.6 -> 52.2 s. Also tried and dropped: the chunk's
bf16 projection GEMM (`mm_bf16_batch`) with four output rows per block
sharing the activation loads (bitwise): the projection stage 1.55 ->
1.61 s, no gain; that stage is its upload, four downloads and sync, not
the kernel. And the fused indexer call's inputs built by the
engine directly in the DLL's pinned staging block (its 3 MB of copies per
chunk-layer skipped, bitwise): the call's host time 2.84 -> 2.71 s per
3.3k prefill, within the unprofiled noise; dropped as not worth its
export. The fused call's host side is its quantization, gather and RoPE
tables (already across threads), the upload and the sync. The fused MoE route once more, with
its cause fixed: the grouped MoE's device-ids path used the replicated
layout (every route's row 64 times); a device-side grouping (one block:
counts per expert, padded bases, the row map by atomics, -1 past the used
rows, which DeepGEMM's contiguous grouped kernel skips) made the fused
pass 3.2 ms per chunk-layer against 3.9 ms for the two calls (profiled,
bitwise), yet the unprofiled time to first token was 0.4 s worse over two
A/B pairs (40.9 / 40.8 s fused, 40.4 / 40.5 s two-call). Dropped; the
device grouping goes with it (it only served the fused path).

Seventeenth measurement (2026-10-29): a chunk's intermediate activations
taken on the card where the producer's device copy still holds them.
`V4_RESIDENT` (1): a small tag table (device activation, host array,
count) that the producers set after their download and the consumers
read to copy device to device instead of uploading: the normalized
attention input (the mHC pre's output) for the qa and kv projections,
the normalized FFN input (the fused hc2's output) for the MoE, the MoE's
output (bf16-rounded on the device already, so the host's round leaves
it equal) for the FFN post. The tags are cleared at each block's start
and dropped whenever a mirror is about to be rewritten. `2` = verify:
the device copy against the host array before each take (`resident`
lines). Same bytes: bitwise. Measured (3.3k prompt, 36 tokens, two A/B pairs): time to first token 40.6 / 40.5 -> 39.8 / 39.7 s; the verify over the whole prompt (4472 takes) reports no mismatch; the greedy text unchanged.

Eighteenth measurement (2026-10-30): the next layer's experts predicted
and read ahead during decode. `V4_PRED_NEXT` (2): a predictor thread
scores layer L+1's gate (bf16 rows, the router's own sqrt(softplus)+bias
top-k order) on layer L's FFN input the moment layer L's routing is
done, and queues the predicted experts that neither the RAM cache nor a
card holds to N lanes of the prefetch pool; a hash-router layer's set is
exact from the token id and is queued before the layer starts. Measured
hit rate 74-78 % of the 258 routed experts per token (~110 queued).
`V4_PRED_PIECES` (1): the lanes reserve VRAM mirrors and copy the pieces
as they land, as the decode lanes do (`V4_UPLOAD_PIECES`), so a hit is a
mirror and not a RAM slab to upload; `0` = RAM only; `2` = the copies on
a prefetch stream of their own with a per-mirror event the expert group
waits on (measured equal to `1`: the fence the group waits for is the
lanes' own uploads, not the prefetch's). `decprof` gained
`pred-hit=P% of N queued=Q`. Never a different token: a wrong guess
costs disk bandwidth in a window the disk would otherwise idle, and the
one-card text is bitwise (md5 bd9c9eae2f2f). Measured (3.3k prompt, 36
tokens, --memory-gb 16): one card moe 224 -> 197 ms per token (loads 167
-> 87, compute 29 -> 82: the lanes' PCIe uploads, ~100 experts x 12 MB
per token, are now the exposed floor under the fence that the reads used
to hide), after the first token 12.58 -> 11.87 s; two cards moe 188 ->
163 ms. 3 or 4 lanes: no better than 2. Found on the way and fixed: with
two cards, an expert that a lane's probe found resident during the layer
(common now that the prefetch lands mirrors mid-layer) was counted into
the second card's late group but excluded from its arrays as if the early
group had covered it, leaving uninitialized expert pointers (a crash at
the first token or a hang after a few); the early group's experts are
now flagged apart and the late group is counted by the test that fills
it.

Nineteenth measurement (2026-10-30): the layer after next predicted as
well, tried and kept off. `V4_PRED_DEPTH=2`: layer L+2's experts scored
by L+2's own gate on layer L's FFN input and queued behind L+1's, a
two-layer read window; the pool now skips a key already queued or in a
lane's hands (`dup=` in `decprof`), so L+1's prediction of the same layer
a layer later joins the read in flight. The guess two layers ahead hits
63-72 % (`pred2-hit=`), the reads per token rise 129 -> 150, loads fall
87 -> 80 ms and compute rises 82 -> 94, on the aux stream and on the
per-mirror event stream alike (`V4_PRED_PIECES=2`), with 2 or 3 lanes:
moe 198 -> 203 / 201 ms, after the first token 11.90 -> 12.36 s,
bitwise. So the window is not the limit: the extra experts' uploads and
mirror reservations cost more than their earlier landing saves (the
non-resident count at routing time, ~100 lane lookups per token, does
not fall). Left in at depth 1 (its default), the dedupe with it.

Twentieth measurement (2026-10-30): the guess cut by rank, and the CPU
split again. `decprof` now reports the next-layer guess's hit rate by
router rank (`rank-hit=`): 99/95/90/75/55/40 % for ranks 1-6, so the two
lowest ranks are the wrong guesses. `V4_PRED_TOPK=M` (16 = all) queues
only the first M of a router layer's predicted set: M=4 cuts the reads
per token 127 -> 106 and compute 82 -> 68, but loads rise 87 -> 100 by
the same amount (the right guesses among the cut ranks are now the
lanes' to read at the layer): moe 197 -> 196 (M=5) / 196 (M=4) / 200
(M=3), bitwise. `DSV4_HYBRID=1` (the misses split between a PCIe fill
and the CPU, measurable only with `V4_UPLOAD_PIECES=0` since a piece read
lands in a mirror before the split decides): moe 251, compute 147, as
the earlier run of it (moe 384 on the build of its day). Three
schedules of the same bytes (a wider window, fewer guesses, a later
landing) all come to ~197 ms: a token's MoE is its ~100 uploads x 12.75
MiB over the x8 link plus the kernels, however they are ordered, and the
second card's link is already taking its half of the pieces. Left at
`V4_PRED_TOPK=16`.

Twenty-first measurement (2026-10-30): the token's time outside the MoE
(hc 26 + attention 43 ms of ~265 at 3.3k context, one card). `decprof`
splits `hc=` into `pre=`/`post=` (the mHC pre with its norm, the post with
its round) and the kernel timers (`DSV4_IDX_KPROF`, `DSV4_WO_KPROF`) run
on decode: the attention chain's device time is its host wall (q 0.08,
attention 0.135 / 0.047, wo 0.1 ms per layer: the card's bandwidth); the
fused indexer selection is 0.4 ms of device time per layer (proj 0.18,
headw 0.10, score 0.09) for one token. The hc pre was 0.21 ms per call for
~400k multiply-adds: `V4_HC_PAR` (1) keeps the OpenMP team only for the
mix (the 24-row loop, a team no larger than its rows) and runs the column
loops of the pre and the whole post on the calling thread (a 4 x 4096
column pass is microseconds, a 32-thread fork and join several times
that); `0` = every loop serial (the mix alone is then 0.3 ms per call),
`2` = every loop on the full team (the old behaviour). Each row's and
column's summation order is the same in all three: bitwise. Measured (36
tokens): hc 26 -> 19 ms per token (pre 18 -> 14, post 7 -> 5; `0`: 31),
after the first token 11.95 -> 11.79 s, md5 bd9c9eae2f2f. The selection
staging's two one-token OpenMP regions (`if(need > 1)`) go with it.

Twenty-second measurement (2026-10-30): the fused indexer selection's
one-token kernels (`DSV4_IDX_KPROF`, 64 heads x 128, hidden 4096, ~830
candidates). The projection (8 MB fp8, one thread per row for the
reference's serial order) spent its time in `e4m3_dev`'s `ldexpf` and
branches per element with one warp per scheduler to issue them: the
kernel now decodes through a 256-entry shared-memory copy of the same
values (0.18 -> 0.085 ms); unrolling the inner loop and 64-thread blocks,
tried first, changed nothing (0.17). The head weights (64 rows of 4096
bf16, one thread per row: 512 strided 16-byte loads in a chain, 0.115 ms)
run with sixteen lanes per head for one token: the lanes fetch the row's
next 128 columns as one coalesced load into shared memory and lane 0 runs
the serial chain from there (0.084 ms). Two staged block designs were
measured slower on the way (64-column chunks 0.20, 256-column chunks with
a per-thread load array 0.15: sync round trips, then local-memory
spills). The device time per indexer layer 0.40 -> 0.28 ms; `DSV4_IDX_VERIFY=1`
reports no mismatch over the prompt and the decode (every fused layer:
q, head weights and scores 0). Measured with the hc change above (3.3k
prompt, 36 tokens, one card): attention 43 -> 40 ms per token (`select`
15 -> 10), hc 26 -> 18, after the first token 11.95 -> 11.64 s, md5
bd9c9eae2f2f.

Twenty-third measurement (2026-10-30): the decode token's six projections
of the attention input in one round trip, tried and kept off.
`V4_DECODE_PROJ6=1`: wq_a, wkv (their fp8 mirrors, the kernel their own
matvec call runs) and the compressor's and the indexer-compressor's bf16
pairs through one upload, six launches and one sync
(`dsv4_cuda_decode_proj_multi`, optional in the loader) where three round
trips used to be. Bitwise (md5 bd9c9eae2f2f), and equal: the three
phases' sum 15.1 -> 13.9 ms per token (qa 4.4 -> 10.0 as the call's
host, compressor 7.8 -> 2.8, kv 2.9 -> 1.1), attention 40 -> 40, after
the first token within noise. The device work is the same ~0.1 ms per
layer; only two sync latencies went. Left off.

Twenty-fourth measurement (2026-10-30): the one-token matvec kernels as a
warp per row, bitwise. `DSV4_MV_WARP` (1): `mv<F>` (fp8 and fp8-bf16
mirrors: wq_a, wkv, the shared expert's three, the chain's three under
`DSV4_ATTN_MV=0`) and `mm_bf16_batch` with a batch of one (the
compressor's and the indexer-compressor's pairs) read one byte or one
bf16 per thread; `mv_warp` / `mv_bf16_warp` give each lane eight
consecutive columns (one 8- or 16-byte load, 256 contiguous bytes per
warp per step) and fold the same 256 partials in the same tree order
(the 128..8 steps as shuffles, 4..1 in-lane), so every output is the
block kernel's operands in its order. `DSV4_MV_WARP_VERIFY=1` runs both
per call and counts differing words: 11,865 calls over a 3.3k prompt and
36 tokens, 0 (`mvwverify`). Measured (3.3k context, one card, 36 tokens,
md5 bd9c9eae2f2f): attention 38 -> 36 ms per token, the shared expert
10 -> 7, after the first token 11.64 -> 11.50 s. On the chain's 32 MB
matrices the warp kernel runs at the shaped kernel's speed
(`DSV4_WO_KPROF`: wq_b 3.9 vs 4.1, wo_a 2.0 vs 2.2, wo_b 2.3 vs 2.3 ms
per token), so `DSV4_ATTN_MV=0`, the exact reference order, now costs
nothing (after the first token 11.53 s, md5 85d209f4e138), so the
default is now 0: the decode chain bitwise the CPU reference, and
85d209f4e138 the bench baseline from here on (`DSV4_ATTN_MV=1` gives the
earlier bd9c9eae2f2f). `V4_IDX_RADIX` (1): the ranking of a
token's indexer candidates (score descending, index ascending) as a
stable byte-radix sort over the score's order-preserving key instead of
qsort with its comparator callback, the same ranked list element for
element (`DSV4_IDX_SORT_VERIFY=1`: 69,510 sorts, candidate counts 1 to
828, 0 mismatches); select 10 -> 9 ms per token; the sort's OpenMP team
no longer starts for one decode token. Tried and reverted: the decode
chain's seven small inputs and its output through the pinned staging
block as one copy each (`DSV4_PIN_STAGE`, as the prefill chain does) and
the projection calls' outputs the same way: the chain 14 -> 16 ms per
token, the projections equal. On this driver the small pageable copies
are cheaper than pinned DMA, and the pageable downloads between launches
were not stalling the host (the `DSV4_PROJ_KPROF` per-launch times in
`dsv4_cuda_decode_proj_multi` sit at a ~35 us floor either way).

Twenty-fifth measurement (2026-10-30): the hyper-connection post and
the pre's collapse as plain loops when `V4_HC_PAR` <= 1. Their OpenMP
forms with the `if` clause false were still outlined by the runtime and
ran scalar (`collapse(2)` recovering the two indices per iteration); the
plain loops vectorize across columns with each element's chain in its
order: bitwise (md5 85d209f4e138). `decprof` hc post 4.8 -> 3.8 ms per
token, pre unchanged (14).

Twenty-sixth measurement (2026-10-30): the hyper-connection pre's mix.
`DSV4_HC_PROF=1` (`hcprof`, the pre's parts per call): the mix took 145
of the call's 165 us, the row-per-thread OpenMP loop over 24 rows x
16384 columns: ~20 us of work per thread (a 16384-long fma chain over 64
KB) and the rest the wake and join of a 24-thread team. `V4_HC_MIX_GROUP`
(8): the rows in groups of eight, each group one column loop with eight
accumulators (every row's own serial order: bitwise), the mean square as
one more task; `V4_HC_MIX_TEAM` (1): on the calling thread alone. The
single thread's three groups run memory-bound at 108 us per call; a
team of four measured 113 (the fork and join cost what the threads
saved), two threads 144, the row-per-thread loop 145; spinning workers
(`OMP_WAIT_POLICY=active`, `GOMP_SPINCOUNT`) changed nothing. `decprof`
hc 18.3 -> 14.2 ms per token (pre 14.8 -> 10.6), md5 85d209f4e138. What
is left of the pre is that 1.5 MB read per call (F32 in the checkpoint,
full mantissas: not storable smaller) and the L3 cannot hold it across
the MoE's expert copies.

Twenty-seventh measurement (2026-10-30): the grouped mix over persistent
helper threads. `V4_HC_MIX_TEAM` (4): the calling thread and three
helpers (created at the first call) take the groups round-robin, handed
a job through a flag they spin on for `V4_HC_MIX_SPIN` (50000 pauses,
~100 us) and then sleep on a condition variable; each row's chain is
unchanged, so bitwise (md5 85d209f4e138). `hcprof` mix per call: one
thread 108 us, two 82, three 75, four 70; helpers that never sleep 76
and 73, so the wake is not the floor: at three or four threads the mix
runs at ~20 GB/s, the DRAM bandwidth left beside the expert loaders'
copies. `decprof` hc 13.6 -> 10.2 ms per token (pre 10.0 -> 7.0); after
the first token 11.40 / 11.26 s (36 tokens). A concurrent caller finds
the helpers taken and runs the groups itself.

Twenty-eighth measurement (2026-10-30): the indexer's two latency-bound
one-token kernels (`DSV4_IDX_KPROF`, per token over the 21 indexer
layers). `DSV4_IDX_HEADW` (2): the head weights as one warp per head on
its own block (the lanes kernel put all 64 heads' lanes in one block on
one SM), the row's 256-column chunks double-buffered so the next chunk's
load overlaps the chain's steps, x staged in shared memory once; lane 0
runs the same chain: headw 1.84 -> 0.38 ms. `DSV4_IDX_SCORE_K` (8): the
scoring with a block of 64 head threads walking eight candidates over
queries staged once (the per-candidate kernel re-staged the token's 32 KB
of queries 830 times, ~27 MB of traffic per layer): score 1.90 -> 0.83
ms (K=4 1.09, K=16 1.13, K=1 the old kernel). Both bitwise:
`DSV4_IDX_VERIFY=1` over the 3.3k prompt and 36 tokens, 1,281 calls
(prefill batches and decode tokens), 0 differing head weights or
scores; md5 85d209f4e138. Attention 36 -> 34 ms per token; after the
first token 11.20 s. The indexer's device time per token is now proj
1.8 (the serial reference chain over 32 MB, at ~400 GB/s) + 1.6 of
kernels; the select phase's rest is host. Also tried: `restrict` on
the hyper-connection pre/post pointers (no change: post stays 3.0 ms,
so its time is not the loop), kept as documentation of the contract.

Twenty-ninth measurement (2026-10-31): the decode token's projections of
the attention input in one round trip for every layer, on by default.
`V4_DECODE_PROJ6` (1): the round trip of the twenty-third measurement
took only the indexer layers (20 of 43: those were the ones with all
four bf16 pairs); the 19 layers compressing at ratio 128 paid the qa and
kv matvecs on the CPU (`coli_fp8_matvec_pre`, an OpenMP region each) and
the compressor pair its own round trip, and the four plain layers the two
matvecs. Now every layer sends the input once and runs wq_a, wkv, the
compressor's pair where the layer compresses and the indexer-compressor's
where it indexes (two, four or six launches). `DSV4_PROJ_ONEDL` (1): the
outputs come back as one download of the contiguous device block,
scattered on the host after the sync, where a pageable download per
output made the host wait for each kernel in turn (six stalls per call).
Measured (3.3k prompt, 36 tokens, one card, per token): qa + compressor +
kv 4 + 7 + 2 = 13 ms (the old default) and 8 + 3 + 2 (the old PROJ6
alone, equal again) -> 7 + 1 + 0 = 8 (one download) or 12 + 1 + 0 (a
download per output); attention 34 -> 30 ms, after the first token 11.35
-> 11.20 s, md5 85d209f4e138 on all. The ~160 us per call left is the
round trip (upload, launches, one sync) over ~30 us of kernels.

Thirtieth measurement (2026-10-31): the decode token's cached sparse
attention as two launches. `DSV4_SATTN_SPLIT` (1): the v2 kernel runs a
token on 16 blocks (four heads per block of 512 threads), so a layer's
~390 rows took 0.09 ms of latency on 16 of the card's SMs
(`DSV4_WO_KPROF`, `attn=`). The score kernel runs the v2 kernel's score
phase over (head quads x tiles of 64 rows) into a device buffer; the
value kernel, over (head quads x slices of `DSV4_SATTN_SPLIT_DS` (128)
columns), takes its heads' scores through the v2 kernel's softmax (the
block max, the exponentials, the serial denominator, the bf16 round) and
the row-ordered weighted sum for its columns: 64 + 64 blocks where 16
were, every output the v2 kernel's operands in its order. Measured (3.3k
prompt, 36 tokens, one card, per token): the attention kernel 0.093 ->
0.049 ms per layer, `sattn` 14 -> 13 ms, attention 30 -> 28, after the
first token 11.20 -> 11.10 s; slices of 64 and 256 within noise of 128;
md5 85d209f4e138 on all, `DSV4_WO_VERIFY` clean. The chain's device time
per layer is now q 0.10 (the wq_b matvec with its uploads, round, norm,
RoPE), attention 0.05, wo_a 0.05, wo_b 0.05 ms (the three 32 MB matrices
at ~650 GB/s).

Thirty-first measurement (2026-10-31): the chain's q span taken apart
(`DSV4_WO_KPROF` now prints `qup=` and `qmv=`): of its 0.10 ms per layer
the wq_b matvec is 0.056 (32 MB at 570 GB/s) and 0.047 is the stream
idle before it while the host stages three pageable uploads (qa, cos,
sin; ~15 us each; the four uploads after the launch hide behind the
kernel). `DSV4_CHAIN_PACK` (1): the call's seven inputs as one host block
and one upload, the device block's regions standing in for the separate
buffers: the gap 0.047 -> 0.024 ms per layer, `sattn` 13 -> 12 ms per
token, after the first token 11.16 -> 11.15 s (within noise); bitwise
(md5 85d209f4e138, `DSV4_WO_VERIFY` clean). The pinned block of the
twenty-fourth measurement lost where this pageable one gains: the loss
there was the pinned DMA per copy, not the packing. `DSV4_MV_WARP_UNROLL`
(1): the warp matvec's loop with U chunks' loads issued before their
FMAs (the partials' order unchanged), tried at 2 and 4 and kept off:
within noise end to end (11.14-11.17 s), `DSV4_MV_WARP_VERIFY` clean
over 16,380 calls. The chain's device time per layer: upload gap 0.02,
q 0.06, attention 0.05, wo_a 0.05, wo_b 0.05, small kernels and the
download 0.04 ms.

Thirty-second measurement (2026-10-31): a decode token's attention layer
as one device round trip. `V4_LAYER_FUSED` (1): on the tokens that emit
no compressed row (three of four on the ratio-4 layers, all but one in
128 on the others, every token on the plain layers) the layer's three
trips (the input's projections; the indexer's selection; the chain) and
the host work between them run on the stream from one upload (x, the
RoPE tables, the sinks, meta, the two norm weights) to one download (the
output, the kv row, qa and the compressor projections the host's
canonical state advances with). New on the device, each the host
routine's arithmetic in its order: the qa and kv norms (`vec_norm`: the
sequential mean square, `DSV4_VNORM_FMA` as `DSV4_Q_NORM_FMA`), the kv's
fp8 round trip and the query ranks' (`fp8_qdq`, the reference's e8m0
scale and e4m3 round-to-nearest-even on the binary32 bits), the
selection's rank (`idx_topk`: a bitonic sort of the host's composite key,
score then index, one block, up to 4096 candidates; past that the token
takes the old path), the ring write. The emit tokens keep the old path
(the host's compressor state produces the row). `V4_LAYER_FUSED_VERIFY=1`
runs the fused layer into side buffers beside the usual path: 1,316
layer calls over the 3.3k prompt's 36 tokens, no differing word in the
output, the kv row, qa, q, the projections or the selected ordinals;
md5 85d209f4e138 throughout. `DSV4_LF_GRAPH` (1): the layer's ~20
launches captured once per layer into a CUDA graph (the token's scalars
in the uploaded meta block, the grids sized for the layer's most rows,
the buffers at the layer's maxima, a pinned host block; recaptured when
a baked pointer or shape changes) and replayed per token; measured equal
to the direct launches, kept for the one submission. `DSV4_LF_GPROF=1`:
the graph's device time, 0.38 ms per layer call; `DSV4_LF_KPROF=1`: the
phases with direct launches (its gaps inflate them). Measured: attention
27 -> 23 ms per token (a fused token 21, an emit token 27), after the
first token 11.15 -> 11.03-11.06 s. Of the ~0.5 ms a fused layer costs
the host, 0.38 is the kernels (the three 32 MB matrices ~0.2, the
indexer ~0.1 on its layers, the attention 0.05, the rest small) and
~0.1 the launch, the sync and the host's own work around the call; the
launch count was not it (the graph changed nothing). Tried on the
kernels, each bitwise and each slower on the graph's device time:
`DSV4_MV_WARP_LUT=0` (the e4m3 decode as bit arithmetic instead of the
shared table: +0.025 ms per layer), `DSV4_IDX_PROJ_WARP=1` (the indexer's
projection as a warp per eight-row tile with the bytes shuffled to the
chain lanes: +0.011), `DSV4_MV_WARP_UNROLL=2/4` (equal). What remains:
the emit tokens (the compressor's pooling on the device would fuse them
too), the host's ~0.1 ms per layer around the call, and the matrices at
~640 GB/s.

Thirty-third measurement (2026-10-31): the compressed row emitted on the
device, so the fused layer takes every token. `V4_LAYER_FUSED_EMIT` (1):
the device keeps a mirror of each compressor's state rows (the attention's
and the indexer's; resynced from the host's rows when a token does not
follow the last one written there), the prep kernel writes the token's
row (kv_row = the projection, score_row = the gate + ape[slot]), and on
an emit token `lf_emit` pools the row as compressor_pool_and_emit does
(per column the max over the pool rows in the comparison form, the
exponentials' total and the weighted sum in row order with
`DSV4_CS_POOL_FMA` (1) as the build contracts it, the division, bf16; the
norm; the RoPE at position + 1 - ratio; the fp8 round trip of the first
dim - rope_dim values, or the indexer's hadamard + fp4 round trip),
appends it to the comp cache and the key cache, and shifts rows 4..7 of
the state down. The host's advance then finds the row there (the caches'
counts are marked). `V4_LAYER_FUSED_VERIFY`: 1,505 layer calls, 189 of
them emits, no differing word in the emitted rows either. An emit token
27 -> 25 ms of attention (the host still pools its own canonical row,
~1 ms), the mean 24 -> 23.

Thirty-fourth measurement (2026-10-31): the output head on the device,
and the bench's own overhead. The CPU head (head_bf16_dot over the
resident bf16 rows, 1 GiB per token) took 23.4 ms per token.
`V4_GPU_HEAD` (1): a bf16 mirror of the head (uploaded on the first
call) and `head_dot_warp_kernel`, a warp per vocabulary row with lane
0 adding the 4096 products in column order as the CPU does, then the
first-maximum argmax: 4.5 ms per token, the token and the logit the
CPU's on every one of 36 tokens (`V4_GPU_HEAD_VERIFY=1`, `headverify`),
in both the one-card and the `DSV4_CUDA_DEVICE2=auto` configuration.
The mirror's 1 GiB displaces decode expert mirrors on one card (MoE 204
-> 212 ms per token there; with the second card none): net a token
308 -> 297 ms, after the first token 11.35 -> 10.94 s; with the second
card 10.92 -> 10.27 s. `V4_GPU_HEAD_DEVICE` picks the card (unset: the
second when the engine opened one with the same DLL flavour, else the
primary; the other flavour's module is not reached, the primary takes
it). While looking for the rest of a token's time (the stages summed
to 238 ms of a 300 ms step): the `decprof` line itself cost 37 ms per
token (`pre=` in the line, now `[alloc free total pre marks tail]`): an
unbuffered stderr behind the bench's redirect wrote it in dozens of
small writes. Every after-first figure in these notes carries that
~1.5 s (the comparisons between runs stand; the decode without
`DSV4_DECODE_PROF` is 9.65 s after the first token, 276 ms per token).
The line is now formatted once and written once. Also noted: with
`DSV4_CUDA_DEVICE2=auto` the generated text can differ between runs of
different speed (the partial sums add by residence, documented as an
order-of-addition class), which the head verify there confirmed is not
the head.

Thirty-fifth measurement (2026-10-31): the MoE floor, looked at once
more with the attention and the head out of the way. Without the
profile line a token is 276 ms (9.65 s after the first token): hc 10,
attention 23, head 4.5, the rest 10, and the MoE ~218 (shared 7, loads
~120, compute ~70: the group kernels are 12 of that, the rest the fence
on the lanes' uploads). `V4_EXPERT_TRACE`: at routing time 44 % of the
258 routed experts are VRAM-resident, 67 % RAM-cached, 33 % in neither,
so ~85 experts (1 GiB) come from the drives per token and ~100 go over
PCIe; the drives are busy about 35 % of the token. Tried, each equal or
worse, the text bitwise throughout: `V4_LOADER_LANES=4/6`, `V4_PRED_NEXT=3/4`
(9.67-9.91 s), `--memory-gb 22` (9.67 s: a RAM hit still rides the PCIe
link, which is the binding floor at ~90 ms per token of uploads);
`V4_PRED_DEPTH=2` again (10.52 s) and `V4_PRED2_RAM` (1), the
layer-after-next prediction reading into RAM only, no mirror pieces
(10.52 s too: loads 121 -> 133 ms; its 34 % wrong guesses, ~1 GiB per
token, compete with the layer's own reads on the same drives, so the
window is not the limit, the link traffic is). What would move it:
experts resident on more VRAM, fewer bytes per expert, or a decode that
batches several tokens per expert load. The `decprof` line gained a
`[block: alloc free total pre marks tail]` tail (the block function's
time around its stage marks; `pre` was the print itself, now ~0.2 ms).

DeepInfer (2026-11-03): DeepSeek's production inference engine, open
since 2026-10-20 (Apache-2.0, github.com/deepseek-ai/DeepInfer), read
for what this engine could borrow. It serves DeepSeek-V4.1-Flash on
NVIDIA SM100 and Ascend 950 across a scale-up domain (attention DP,
MoE expert parallelism with load balancing, FlashMLA, DeepGEMM, DeepEP,
DeepSelect, TileLang kernels), built on vLLM; the experts live in GPU
memory and the serving cost is compute and interconnect, so the kernels
and the parallelism do not transfer to one consumer card streaming
experts from NVMe, where the token is expert bytes over the drives and
PCIe. Two ideas were worth a measurement here. Speculative decoding
(DSpark with confidence-scheduled verification, drafts up to 5):
this engine carries DSpark's drafter and full MTP; on today's build
`V4_MTP=1` 10.19 s and `V4_DRAFT=1` 9.99 s after the first token
against 9.6 s plain, the text the same. The reason is structural: a
verification batch of k tokens loads the union of their experts, and
consecutive tokens share only ~35 % of them, so the MoE cost grows
almost with k and the win per accepted token is small even before the
replay of a rejected suffix; the setups where it pays have the experts
resident. Bounded replay ("SWA bounded replay", a kept suffix of 140
tokens) addresses the replay cost on their YOCO decoder; this model's
recurrent compressor state has no such suffix, though an undo log of the
few rows a token changes (its ring slot, its compressor rows, an emitted
row every fourth token) would make a rejection cheap if speculation
ever paid. Nothing else applies.

Merge of dev (2026-11-12, 531 commits, 38 on this engine and its CUDA
backend). dev had worked the same ground from another side: an exact
vocabulary head on the device, the compressor and indexer preparation on
the device, the decode projections in one round trip, wider sparse
attention kernels, a fused cooperative MoE launch, resident routed
experts, layer placement over several cards, a peer mirror pool with
peer compute, a CUDA fault latch. Measured first, dev alone on this box
(same prompt, 36 tokens, --memory-gb 16, one card): 88.5 s to the first
token and 22.5 s after it, against this branch's 39.1 s and 9.6 s, the
text the same (md5 85d209f4e138 on both). So wherever the two sides
rewrote the same function, this branch's stays whole, and dev's work is
kept beside it where it stands on its own:

- kept from dev, working: the whole CUDA backend (its kernels, the
  cooperative MoE `DSV4_CUDA_MOE_FUSED6`, the peer-compute pass, the
  fault latch, `DSV4_CUDA_PIN_CAP_GB`, the rewritten device init with
  this branch's `DSV4_CUDA_SCHED` in it), the exact head, the device
  output projection (`DSV4_CUDA_WO_DECODE`, taken where this branch's
  chain does not take the token), the request-level fixes, the route
  scratch reuse, the serve statistics, the error stage names.
- this branch's, unchanged: the decode attention (the fused layer), the
  indexer selection, the MoE token pipeline (the lanes, the predictor,
  the read-to-upload pieces, the uploader threads), the mirror cache
  and its second card (`DSV4_CUDA_DEVICE2`, either DLL flavour), the
  engine open. `dsv4_cuda_moe` keeps both: this branch's pass when every
  expert sits on the shared expert's device, dev's passes for a
  peer-mirrored expert or `DSV4_CUDA_MOE_FUSED6`.
- taken from dev because it measured better: the head. dev's exact head
  is 1.5 ms per token where this branch's `V4_GPU_HEAD` was 4.6; it is
  now preferred, and this branch's mirror is the fallback where dev's
  is off or refused. `DSV4_CUDA_HEAD_LAZY` (1): it is uploaded at the
  first head call, after the prompt, instead of at engine open, where
  its 1 GiB (2 while packing) left a 16 GB card's prefill without room
  (39 -> 63 s to the first token, `weight allocation: out of memory`).
- off by default here, on with `V4_LAYER_FUSED=0`: dev's packed
  compressor copies (`DSV4_CUDA_COMPRESSOR`) and resident indexer
  mirrors (`DSV4_CUDA_RESIDENT_INDEXER`), 769 MiB of VRAM the fused
  layer never reads (~60 expert mirrors; MoE 226 -> 218 ms per token).
- ported into this branch's functions: the fault latch at the mirror
  entry points (`dsv4_cuda_lost()`: after a sticky fault the mirrors are
  refused and decode goes on from host copies; the token in flight when
  the fault happens still fails), and the store's books for a routing
  served from VRAM (`coli_v4_expert_store_note_vram`).
- not available on this branch: dev's peer mirror pool
  (`DSV4_CUDA_PEER_DEVICE`), layer placement (`DSV4_CUDA_DEVICES` with
  more than one device) and full expert residency
  (`DSV4_CUDA_RESIDENT_EXPERTS`). They build on dev's per-device mirror
  caches, which this branch's cache replaces; each prints a warning and
  the engine runs as before. The peer pool needs two cards of the same
  SM version or P2P, which this box (sm_120 + sm_86 under WDDM) has
  not, so it could not have been tested here; `DSV4_CUDA_DEVICE2` is
  this branch's second card. `V4_GPU_RESIDENT_FIRST`, `V4_IDX_FAST`,
  `V4_SPARSE_FAST` and `V4_ATTN_BATCH_PROJ` are dev's opt-in forms of
  what this branch does by default (mirror-first, the radix rank, the
  split attention, `V4_DECODE_PROJ6`) and have no effect in the paths
  this branch replaced.

After the merge (same bench): 38.9 s to the first token, 9.55 s after it
without the profile (9.64 before), md5 85d209f4e138;
`V4_LAYER_FUSED_VERIFY` and `DSV4_WO_VERIFY` clean over 1,505 calls
each; with the second card 9.84 s profiled (10.27 before).

Upstream pull requests from this work (2026-11-12, each a branch cut from
dev, measured on dev itself with an experiment manifest under
`docs/experiments`):

- `fix/v4-head-lazy-upload` (`DSV4_CUDA_HEAD_LAZY`): the exact head
  uploaded at the first head call. On dev, same commit: 81.5 -> 66.2 s to
  the first token of the 3.3k prompt, the 556 `kv ring / kv comp
  allocation: out of memory` lines per run gone.
- `perf/v4-hc-decode-calling-thread` (`V4_HC_TEAM`): a decode token's
  hyper-connection steps on a team of 6 for the mix and on the calling
  thread for the column loops when the GPU tier is on; dev's loops kept,
  only their team sized (`if` / `num_threads` clauses), so identical by
  construction. On dev: hc 27.2 -> 17.0 ms per token, 22.45 -> 22.29 s
  after the first token (a token there is 640 ms). Team sizes: 0 (the
  default team) 27.2, 1 30.5, 4 18.4, 6 16.8, 8 18.7, 12 18.4, 16 19.7.
- dev's `decprof` line is one write already: the 37 ms per token of the
  thirty-fourth measurement was this branch's own longer line.

What the second one found about this branch's hyper-connection forms
(the twenty-fifth to twenty-seventh measurements: the plain column
loops, `V4_HC_MIX_GROUP`, `V4_HC_MIX_TEAM`). They were called bitwise on
the strength of the generated text. Compared directly now (each result
of `coli_hc_pre` / `coli_hc_post` hashed against dev's header, GCC 16.1
`-O3 -march=native -fopenmp`, the shape known only at run time as in the
engine): identical in every mode for hc 4 with d 4096, 1021, 2048 and
5120, for hc 8 and for hc 2; for hc 3, d 37 the single-thread grouped
mix (`V4_HC_MIX_TEAM=1`) differs in `comb`. A first form of the pull
request with the same regrouping inside one function differed by 1 ulp
for most shapes, hc 4 x 4096 included. So this branch's forms are equal
where the engine uses them (V4 Flash is hc 4, d 4096), but by
measurement, not by construction: on FMA targets GCC contracts a
differently shaped loop differently. A change to these loops needs the
direct comparison, not the text md5.

Profilers: `DSV4_ATTN_PROF=1` (per chunk-layer `attnprof`/`blockprof`/
`idxprof`), `DSV4_DECODE_PROF=1` (per-token `decprof`), `DSV4_CUDA_MOE_PROF=1`,
`DSV4_IDX_VERIFY=1` (GPU indexer projection/scoring and the fused selection
vs CPU, bitwise: `idxproj`, `idxverify`, `idxfused`), `DSV4_IDX_KPROF=1`
(device time per stage of the fused selection), `DSV4_WO_VERIFY=1` /
`DSV4_WO_KPROF=1` (the decode attention tail chain: bitwise check, stage times),
`DSV4_MOE_KPROF=1` / `DSV4_MOE_MV_VERIFY=1` (the decode expert group: kernel
time vs host wall per card, the new matvecs vs the old in bf16 ulps),
`V4_PREFIX_LOG=1` (checkpoint/hint decisions), `v4_mirror` per-drive I/O
telemetry after each request.

## GPU coverage

The generic kernels do not require Ampere. bf16 appears only as *storage* —
every use is `__bfloat162float` / `__float2bfloat16`, and the arithmetic is
fp32 — and fp8 weights are decoded through the 256-entry constant lookup
table the engine already carries, not fp8 tensor-core MMA. The generic
build's two activation-quantise kernels (`fp8_sim64`, `fp8_sim64_dynamic`)
do call `__nv_cvt_float_to_fp8`, but `cuda_fp8.h` resolves that to native
`cvt` PTX only at `__CUDA_ARCH__ >= 890` and to its own software path
below, so it builds and runs on older cards. All of it compiles unchanged
for sm_61 and sm_75, so `CUDA_ARCH=portable-pre-ampere NO_TC=1` extends the
generic tier to Pascal and Turing.

`NO_TC=1` is required because the opt-in `DSV4_CUDA_TC` path calls cuBLASLt
block-scaling APIs (`CUBLASLT_MATMUL_DESC_A_SCALE_MODE` and friends) that need
CUDA >= 12.8 — a *toolkit* dependency, not an architecture one.

This preset stops at `sm_90` plus `compute_90` PTX rather than mirroring
`portable` up to `sm_120`. `compute_120` requires CUDA >= 12.8 — the same
toolkit floor `NO_TC=1` exists to stay below — so including it would make the
preset unbuildable by exactly the older toolkits its users tend to have. Newer
cards are still covered, through driver JIT of the `compute_90` PTX.

DeepGEMM is not available below sm_90 and cannot be made available: it calls
`cuTensorMapEncodeTiled` (TMA), a hardware unit introduced in Hopper, so there
is no software path to fall back to.

| build | GPUs | prefill | decode |
|---|---|---|---|
| CPU only (no DLL / `DSV4_CUDA=0` / Linux default) | — | CPU reference | CPU reference |
| generic DLL / `CUDA=1` | any sm_80+ | GPU attention block, indexer, generic batched MoE on the VRAM bank | GPU attention, indexer, expert mirrors |
| generic, pre-Ampere / `CUDA=1 CUDA_ARCH=portable-pre-ampere NO_TC=1` | sm_61 (Pascal), sm_75 (Turing) | as generic | as generic |
| DeepGEMM DLL / `CUDA=1 DEEPGEMM=1` | compute 12.x | as generic + tensor-core dense/MoE GEMMs | same as generic |
| second GPU | any sm_80+ beside the primary, mixed generations allowed (Windows: two DLL flavours) | — | a second expert-mirror cache (`DSV4_CUDA_DEVICE2=auto`, opt-in: measured slower until mirrored experts skip the host read) |
| 3+ GPUs / expert-parallel prefill | — | single primary today; the second card only hosts decode mirrors | — |
| multi-GPU layer placement | up to 16 visible CUDA devices | contiguous layer ranges, with dense mirrors and KV on the owning device | per-device expert caches; layers still execute sequentially |

The runtime check `dsv4_cuda_backend_arch_ok` admits sm_60 and up for the
generic build. It is deliberately independent of what the binary contains: a
`CUDA_ARCH=portable` build has no sm_61/sm_75 cubin, and running it on such a
card fails at launch with `"no kernel image is available"` rather than producing
a wrong answer. Build with `portable-pre-ampere` for those cards.

### Multiple GPUs: layer placement

Set `DSV4_CUDA_DEVICES` to opt into contiguous layer placement:

```bash
DSV4_CUDA_DEVICES=0,1,2,3,4,5 COLI_CUDA_ATTN_BATCH=1 COLI_CUDA_MOE_BATCH=1 \
  python c/coli run --model /path/to/DeepSeek-V4-Flash-0731 --ram 48 --ngen 64 \
  "Explain why the sky is blue."
```

Build the CUDA engine first (`make -C c -f Makefile.deepseek-v4 deepseek-v4
CUDA=1 CUDA_ARCH=sm_120` for RTX 5090). Ordinals refer to the devices visible
through `CUDA_VISIBLE_DEVICES`, and list order determines layer order. For
43 layers and six cards, the ranges are 0–6, 7–13, 14–20, 21–27, 28–34 and
35–42. Startup prints each assignment. Invalid/duplicate lists and unavailable
multi-device configurations fail startup instead of silently running one card.
The single-device default is unchanged.

Each layer's dense tensors, attention KV and routed-expert mirrors live on its
assigned GPU. Cache limits and free-VRAM probes are per device. The transient
prefill bank moves at device boundaries; double-bank lookahead stays within a
device's layer range. MTP's separate expert cache stays on the first device.

This increases usable cache capacity, not parallel compute within a layer.
The engine still executes layers sequentially with CPU-canonical activations;
there is no six-way TP, EP, or overlapped pipeline. CPU expert-store reads can
still occur before a GPU cache hit, so RAM budget and disk traffic still matter.
Do not interpret aggregate VRAM occupancy as a throughput or numerical-parity
claim. Validate representative prompts against the single-device configuration.

Peer access is best-effort: if the driver cannot enable it for a pair, startup
logs the reason and continues, because contiguous layer placement does not
require P2P. A build with `COLI_DSV4_NCCL` initializes NCCL only for six devices
(the EP2 path); other counts do not need it. The exact head goes to the owning
(last) device with room, then any other device, then the CPU. A GGUF keeps its
dense matrices on device 0; `DSV4_CUDA_DEVICES` splits only the layer/expert
tier for a GGUF.

Each layer's KV ring and compressed-KV buffer are tied to their device; an
append from a different device is refused, so restart the engine after changing
the list.

Linux backend lifecycle check (requires at least two GPUs; exercises all listed
devices, duplicate-list refusal, active-context protection, and reopen with a
different KV owner). Run the whole CUDA unit set with:

```bash
make -C c -f Makefile.deepseek-v4 dsv4-cuda-tests CUDA=1 CUDA_ARCH=sm_120 DSV4_TEST_DEVICES=0,1
```

or one test directly:

```bash
make -C c -f Makefile.deepseek-v4 tests/test_dsv4_multigpu_cuda CUDA=1 CUDA_ARCH=sm_120
c/tests/test_dsv4_multigpu_cuda 0,1,2,3,4,5
```

## Vulkan (`VK=1`, any GPU with a Vulkan 1.2 driver)

`make deepseek-v4 VK=1` (the Vulkan headers and `glslc` at build time) links the shared
Vulkan backend and its routed-expert tier; `COLI_VULKAN=1` opens the device once
the engine has loaded. No usable device, or no shaders, and the run stays on the
CPU with one `[VK] deepseek_v4:` line that says so. Two things go to the device:

- **The matrices the engine keeps for its whole life**: the resident dense layers
  (fp8 128x128 blocks as fmt 12, fed the activation after the CPU's own E4M3
  rounding), the bf16 head, router and compressors (fmt 11). The `--oracle` path's
  per-forward copies, the indexer's `weights_proj` and the DSpark stages stay on
  the CPU.
- **The routed experts**, on the shared tier ([vulkan.md](vulkan.md#the-routed-expert-tier-vk_tierc)):
  a cache on the device that fills from the history and adapts while you chat,
  computed while the CPU computes the experts it does not hold. The experts go up
  as the store reads them, fp4 with a ue8m0 scale per 32 (MXFP4, fmt 7), with an
  activation of this engine's own that makes every rounding its CPU expert makes:
  x to E4M3 per 128 (on the host), gate and up to bf16, the weighted activation to
  bf16, down's input to E4M3 per 128, the output to bf16
  ([DeepSeek V4's activation](vulkan.md#deepseek-v4s-activation)). The decode step
  and the prefill union hand the device their routes first, the store lends the CPU
  only the experts the device did not take, and every position adds its experts in
  the CPU path's order, so on what was measured (Lavapipe, the tiny fixtures) the
  device's rows are the CPU's after those roundings: a served prompt's logprobs
  with every expert on the device come back byte for byte.

Where the dense matrices go follows the backend's one rule: on a device that shares
the CPU's RAM (an integrated GPU, Lavapipe) they stay on the CPU while the tier is
on, and `COLI_VK_DENSE=1` puts them on the device anyway. The tier's history is the
store's `.coli_usage`: its warm start reads the hottest experts straight from disk
outside the cache, and a routing the device served is counted as a lookup counts
one (pins, HITS and heat, the saved history). A hot expert the store keeps in its
rows16 layout is unpacked for the tier, and only when the tier will take it. EMAP
shows a device-resident expert as tier 2; each run and serve turn ends with a
`[VK] tier deepseek_v4` line (device and CPU shares, uploads, evictions, the time
the device hid). With the CUDA tier on, CUDA wins and the Vulkan tier stays off.
Variables: `COLI_VULKAN`, `COLI_VK_DENSE` and the `COLI_VK_TIER*` family in
[ENVIRONMENT.md](ENVIRONMENT.md#vulkan-any-gpu-with-a-vulkan-12-driver). CI runs it
on Lavapipe (`tests/vulkan_engines.sh deepseek`, `deepseek-sanitize`). The only GPU
it has run on is an Intel Iris Xe through Mesa's Dozen (Direct3D 12 under WSL),
for correctness: the same ids as the CPU in every configuration above. No speed
has been measured.

`COLI_VK_CHAIN=1` runs every layer as the dense chain instead
([vulkan.md](vulkan.md#deepseek-v4-on-the-chain)): the streams, the attention with its
window ring, compressors and indexer, every bf16 and E4M3 rounding the CPU makes, the
mHC sites and the shared expert on the device, one host round trip per layer for the
router and the routed experts; the host's state stays canonical. It needs resident
dense layers and declines under the CUDA tier. Off by default on an integrated GPU (not
measured on a V4 checkpoint); on the tiny fixtures every configuration gives the CPU's
tokens (`tests/vulkan_engines.sh deepseek-chain`, `deepseek-chain-sanitize`).

## Environment reference (V4 engine)

Defaults in parentheses; all read by `c/deepseek_v4.c` unless noted `.cu`.

**GPU tier**
| var | meaning |
|---|---|
| `DSV4_CUDA` (1) | master switch for the V4 GPU tier; `0` = CPU only |
| `DSV4_CUDA_DEVICE` (0) | CUDA device ordinal of the primary card |
| `DSV4_CUDA_DEVICE2` (unset = off) | second card for decode expert mirrors: `auto` = first other device, or an ordinal; opt-in, see "Second GPU" |
| `DSV4_CUDA_EXPERT_MIRRORS2` (4096) / `DSV4_CUDA_VRAM_RESERVE2_MB` (600) | the second card's mirror cap and VRAM kept free there |
| `COLI_DSV4_DLL2` | Windows: force the secondary backend DLL file name (default: the other flavour) |
| `V4_MIRROR_FIRST` (1) | experts already mirrored in VRAM are computed from there without reading the record into host memory (53 % of a 3k-context token's experts measured); `0` = read first, attach after |
| `V4_SLAB_SHIFT` (1) | a direct expert read stays where its 4 K-aligned window landed in the slab and the views carry the offset, instead of a 12.6 MB memmove per expert (decode 0.62 -> 0.58 s/token); `0` = memmove |
| `V4_DEV2_EARLY` (1) | with a second card, launch its expert group at routing time (its experts are resident there, nothing to read) and join it after this layer's reads and uploads; `0` = launch it at the partition, after them (measured: +22 ms/token at 3.3k context) |
| `V4_EXPERT_TRACE` (0) | decode: one `xtrace` line per layer with the routed experts and their VRAM-mirror / RAM-cache state at routing time, for offline cache simulation (slows the token: it probes every expert) |
| `V4_DEV2_SPLIT` (50) | with a second card, the share (percent) of recycled uploads placed on it once both mirror caches are full; `0` = the second card fills once and stays a static store |
| `V4_MIRROR_BLOCK` (1) | decode expert mirrors as one device allocation each (the record's scale run, then its weight run), refilled with two copies and re-pointed to the record's tensor order; `0` = three tensors, six copies. Needs the DLL's `dsv4_cuda_mirror_alloc`/`_refill` |
| `V4_PREFILL_CHAIN` (1) | the batched prefill attention tail (wq_b, norm, RoPE, sparse attention, inverse RoPE, wo) as one device chain; `0` = the query and the attended cross the link and the CPU does the norm and RoPE |
| `V4_COMP_BATCH` (1) | a prefill chunk's compressor advances (the attention's and the indexer's) in one parallel pass from the batched projections, the serial emit's arithmetic per column (bitwise); `2` = verify against the per-token walk; `0` = the walk |
| `V4_RESIDENT` (1) | a chunk's intermediate activations copied device to device from the producer's mirror instead of uploaded (the normalized inputs for qa/kv and the MoE, the MoE output for the FFN post); `2` = verify against the host arrays; `0` = uploads |
| `V4_BLOCK_SCRATCH` (1) | the prefill block's ten per-call arrays from one persistent page-locked scratch (their copies at link speed, no per-chunk mallocs); `0` = a malloc'd scratch |
| `V4_MOE_BATCH_PROF` (0) | one `moebprof` line per chunk-layer: the batched MoE's switch / route / refill / compute (its own prints inflate the stage) |
| `V4_HC_PIN` (1) | the prefill page-locks the prompt's range of its state buffers for its duration (the block's 8 MB copies at link speed); `V4_HC_PIN_MIN` (256) the prompt length from which; `0` = pageable copies |
| `V4_HC_FUSED` (1) | the prefill block's attention post + FFN pre/norm as one device pass, the states resident for the FFN post (16 MB per chunk-layer not copied); `0` = the two-step path with the states round trip |
| `DSV4_SATTN_HPB_PF` (16) | heads per block for the multi-token (prefill chunk) launches of the cached sparse attention: 16, 8 or 4 (else the one-token rule); the key rows are shared across the heads of a token, the per-head arithmetic is unchanged (bitwise) |
| `DSV4_PIN_STAGE` (1, `.cu`) | the prefill chain's inputs as one copy from a page-locked staging block and its output download into it; `0` = seven pageable copies and a pageable download |
| `DSV4_PF_KPROF` (0, `.cu`) | per-call device time of the prefill chain's stages and its host wall (`pfkprof`) |
| `DSV4_SATTN_HPB` (4) | heads per block of the shaped sparse attention from 256 rows up (`1` = one head per block, the rows read once per head) |
| `DSV4_SATTN_V2` (1) | the cached sparse attention kernel shaped for latency (same arithmetic, same order, bitwise); `0` = the old kernel |
| `DSV4_ATTN_MV` (0) | decode: the attention chain's fp8 matvecs (wq_b, wo_a, wo_b) in the reference order (the `DSV4_MV_WARP` kernel; bitwise the CPU path, md5 85d209f4e138 on the 36-token bench, the baseline since the twenty-fourth measurement); `1` = the bandwidth-shaped kernel of the sixth measurement, the same speed, 1-ulp elements at near ties (the earlier baseline md5 bd9c9eae2f2f) |
| `V4_UPLOAD_PIECES` (4) | decode: a lane's expert read lands in this many pieces, all in flight at once, each copied to a mirror reserved before the read as it lands (the link works under the read); the slab is page-locked before the read. `0`/`1` = one read, the uploader thread copies after it. Needs `V4_MIRROR_BLOCK`, `V4_UPLOAD_FENCE` and the DLL's `dsv4_cuda_mirror_copy`/`_point`/`_host_pin`. `V4_PIECES_VERIFY=1` compares each piece-built mirror with its slab, `=8` compares each slab with a buffered re-read (diagnostics) |
| `V4_UPLOAD_FENCE` (1) | decode: the uploader threads only enqueue their copies; the expert group waits for them on the device (`dsv4_cuda_group_fence`, aux-stream event) instead of the host joining the DMA before launching. Needs the export in every backend module in use (else, or `0`: drained as before) |
| `DSV4_MOE_PINNED` (1, `.cu`) | pinned staging for the decode group's small copies (x, descriptors + weights in one transfer, y); `0` = pageable |
| `DSV4_MOE_MV` (1, `.cu`) | decode expert group / fused MoE: bandwidth-shaped fp4 and fp8 matvecs (one warp per row, 16-byte loads; kernels 1.14 → 0.29 ms per layer on the 5080); `0` = the byte-per-lane `mv_fp4_grouped` / `mv<8>` kernels (a different summation order) |
| `DSV4_MOE_MV_VERIFY` (0, `.cu`) | every fused MoE call recomputed through the old kernels and compared (`moemv verify`: max abs diff, bf16-ulp histogram) |
| `DSV4_MOE_KPROF` (0, `.cu`) | per card: device time of the group's kernels vs the call's host wall (`moekprof`, every 430 calls) |
| `V4_UPLOAD_THREADS` (1) | decode attach on one uploader thread per card (aux-stream copies, joined before compute); `0` = attach inline from the lane loop |
| `V4_READ_STRIPE` (off) | `1` = read each expert window as two halves from the two mirror replicas at once (Windows overlapped twins). Measured with `COLI_MODEL_MIRROR` on a second NVMe (826-token prompt, 60 tokens): decode 24.2 -> 21.3 s (-12 %), prefill 52.4 -> 44.6 s (-15 %), text identical; needs a real second drive — two copies on one drive gain nothing |
| `V4_PRED_NEXT` (2) | decode: `=N` lanes reading the next layer's experts as its gate, scored on this layer's FFN input by a predictor thread, predicts them (hash layers: exact from the token id), skipping what RAM or a card already holds; 74-78 % hit; `0` = off |
| `V4_PRED_PIECES` (1) | the prefetch lanes also reserve VRAM mirrors and copy the pieces as they read (a hit is a mirror); `0` = RAM only; `2` = the copies on a prefetch stream of their own, the group waiting per mirror (measured equal to `1`) |
| `V4_PRED_DEPTH` (1) | `2` = the layer after next predicted too, by its own gate on this layer's FFN input, queued behind the next layer's; measured slower (moe 198 -> 203 ms: the extra uploads and mirror reservations outweigh the wider window) |
| `V4_PRED_TOPK` (16) | `=M` queues only the first M of a router layer's predicted set in the router's rank order (ranks 5 and 6 hit ~55 / 40 %); measured equal (fewer reads, the same time) |
| `V4_HC_PAR` (1) | the hyper-connection steps' OpenMP use: `1` = a team (no larger than its 24 rows) for the pre's mix only, the column loops and the post on the calling thread; `0` = all serial; `2` = every loop on the full team (bitwise in all three; measured hc 26 / 31 / 19 ms per token for 2 / 0 / 1) |
| `DSV4_MV_WARP` (1) | the one-token fp8 / bf16 matvec kernels as a warp per row with 8/16-byte lane loads, the block kernels' summation order (bitwise); `DSV4_MV_WARP_VERIFY=1` runs both and counts differing words (`mvwverify`) |
| `V4_IDX_RADIX` (1) | the indexer candidates ranked by a stable byte-radix sort instead of qsort (the same list); `DSV4_IDX_SORT_VERIFY=1` compares the two per token (`idxsort`) |
| `DSV4_PROJ_KPROF` (off) | per-launch device times of `dsv4_cuda_decode_proj_multi` (`projkprof`, with `V4_DECODE_PROJ6=1`) |
| `V4_HC_MIX_GROUP` (8) | the hyper-connection pre's mix rows in groups of this many chains interleaved on one thread (bitwise); `0` = the row-per-thread OpenMP loop |
| `V4_HC_MIX_TEAM` (4) | the grouped mix's threads: the calling thread and N-1 persistent helpers (spin, then sleep on a condition variable); `1` = the calling thread alone |
| `V4_HC_MIX_SPIN` (50000) | pauses a helper spins for the next job before it sleeps |
| `DSV4_HC_PROF` (off) | the pre's parts timed per call, printed every 1000 calls (`hcprof`) |
| `DSV4_IDX_HEADW` (2) | the indexer head-weight kernel for a token: `2` = a warp per head, double-buffered chunks; `1` = the lanes-in-one-block kernel; `0` = one thread per head (all bitwise) |
| `DSV4_IDX_SCORE_K` (8) | candidates scored per block in the few-token indexer scoring kernel (queries staged once per block); `1` = a block per candidate |
| `V4_DECODE_PROJ6` (1) | a decode token's wq_a, wkv and (where the layer has them) compressor and indexer-compressor projections in one device round trip, every layer (bitwise); `0` = the CPU matvecs and a round trip per pair |
| `V4_PRED2_RAM` (1) | with `V4_PRED_DEPTH=2`: the layer-after-next prediction reads into RAM only, the mirror upload waits for the layer's routing (measured equal to the pieces: the drives are the limit) |
| `DSV4_STEP_PROF` (off) | per-token step prints beside `decprof`: `headprof`, `finalprof`, `tokprof` (the whole step and the layer loop), `loopprof` (the loop's loads, predictor, block, tap); each line costs a stderr write |
| `DSV4_CUDA_HEAD_LAZY` (1) | dev's exact head uploaded at the first head call instead of at engine open (a 16 GB card's prefill needs the room); `0` = at open |
| `V4_LAYER_FUSED_EMIT` (1) | the tokens that emit a compressed row through the fused layer too, the row pooled on the device (bitwise); `0` = those tokens take the three trips |
| `DSV4_CS_POOL_FMA` (1) | the device pooling's weighted accumulate fused, as the build contracts the host loop |
| `V4_GPU_HEAD` (1) | the output head's argmax on the device over a bf16 mirror, the CPU's sums and tie rule (bitwise); `0` = the CPU head |
| `V4_GPU_HEAD_VERIFY` (off) | the CPU head beside the device head, differing tokens or logits reported (`headverify`) |
| `V4_GPU_HEAD_DEVICE` (unset) | the head mirror's card ordinal; unset = the second card when it runs the primary's DLL flavour, else the primary |
| `V4_LAYER_FUSED` (1) | a decode token's attention layer as one device round trip on the tokens that emit no compressed row (bitwise); `0` = the three trips |
| `V4_LAYER_FUSED_VERIFY` (off) | the fused layer beside the usual path, differing words counted (`lfverify`) |
| `DSV4_LF_GRAPH` (1) | the fused layer's launches as a CUDA graph per layer, replayed per token; `0` = direct launches (equal) |
| `DSV4_LF_GPROF` (off) | the fused layer's graph device time per call (`lfgprof`, every 1000 calls) |
| `DSV4_LF_KPROF` (off) | the fused layer's phases with direct launches (`lfkprof`; the gaps between the events inflate them) |
| `DSV4_VNORM_FMA` (1) | the qa and kv norms' mean square with the accumulate fused, as the build contracts coli_v4_rmsnorm |
| `DSV4_MV_WARP_LUT` (1) | the warp matvec's e4m3 decode from the shared table; `0` = bit arithmetic (bitwise; +0.025 ms per fused layer) |
| `DSV4_IDX_PROJ_WARP` (0) | the indexer projection as a warp per eight-row tile (bitwise; +0.011 ms per fused layer) |
| `DSV4_CHAIN_PACK` (1) | the decode chain's seven inputs as one host block and one upload (bitwise); `0` = a pageable copy per input |
| `DSV4_MV_WARP_UNROLL` (1) | chunks of 256 columns loaded ahead of their FMAs in the warp matvec (1, 2 or 4; bitwise; measured equal) |
| `DSV4_SATTN_SPLIT` (1) | the decode token's cached sparse attention as a score kernel over row tiles and a value kernel over column slices (bitwise); `0` = the one v2 launch |
| `DSV4_SATTN_SPLIT_DS` (128) | columns per value-kernel block (32..512) |
| `DSV4_PROJ_ONEDL` (1) | that round trip's outputs as one download scattered on the host; `0` = a pageable download per output (bitwise, +5 ms per token) |
| `V4_PREFETCH_NEXT` (off) | `=N` lanes reading the next layer's experts as the previous token routed them into the RAM cache while this layer runs; measured useless here: those repeats (41 % of a layer's experts) are already VRAM-resident or RAM-cached, so nothing is ever queued |
| `DSV4_CUDA_DEVICES` (unset) | ordered comma-separated distinct CUDA ordinals; split contiguous layers across these devices |
| `COLI_DSV4_DLL` | Windows: force a backend DLL file name (loader) |
| `COLI_CUDA_ATTN_BATCH` (0; 1 for a GGUF on the CUDA expert tier) | `1` = GPU batched prefill attention block + GPU decode attention/indexer; unset, it is on for a GGUF whose dense matrices the CUDA expert tier's device holds (`COLI_CUDA=1`, `COLI_CUDA_DENSE`), `0` keeps it on the CPU |
| `COLI_CUDA_MOE_BATCH` (0) | `1` = prefill MoE on the transient VRAM expert bank |
| `COLI_CUDA_MOE_BATCH_MIN` (256) | min fresh tokens to engage the bank |
| `DSV4_CUDA_EXPERT_MIRRORS` (4096) | upper bound on decode VRAM expert mirrors (12.75 MiB each); free VRAM sizes the cache at run time |
| `DSV4_CUDA_VRAM_RESERVE_MB` (600) | VRAM kept free on the primary card while decode mirrors grow. With `V4_BANK_MAKE_ROOM=1` (default) the prefill expert bank is not reserved: it releases LRU mirrors when it is created (RTX 5080, 3.3k context: 2800 -> 600 MB gave 18 % fewer uploads, decode -5 %, text identical); `V4_BANK_MAKE_ROOM=0` restores the bank-sized default reserve (600 + the bank, x2 with `COLI_CUDA_MOE_DOUBLE=1`, + 64 mirrors, logged as `v4_gpu mirror-reserve=`). An explicit value is used as given |
| `DSV4_CUDA_RESIDENT_RESERVE_MB` (2800, min 256) | VRAM kept free while resident experts preload and the exact head uploads; separate from the mirror-growth reserve above |
| `DSV4_CUDA_RESIDENT_EXPERTS` (auto) | `auto` loads every routed expert when the budget fits and keeps the LRU cache on a budget refusal; `1` requires residency (a budget refusal or hard failure rejects engine open); `0` keeps the cache |
| `DSV4_CUDA_HEAD` (1) | exact vocabulary head on the GPU; `0` keeps it on the CPU |
| `DSV4_CUDA_HEAD_DEVICE` (unset) | exact-head device: unset tries the owning (last) device with room, then any other; set an ordinal to try that device first |
| `DSV4_CUDA_HEAD_BATCH` (1) | batched exact head; `0` per-token |
| `DSV4_HEAD_VERIFY` (0) | `1` checks the batched head against the per-token path (diagnostic) |
| `DSV4_CUDA_RESIDENT_INDEXER` (0; 1 with `V4_LAYER_FUSED=0`) | upload indexer query weights once per layer for dev's device paths; the fused layer keeps its own |
| `DSV4_CUDA_INDEXER_HEADS` (1) | indexer score heads on the device; `0` the block kernel |
| `DSV4_CUDA_INDEXER_PREP` (1) | indexer preparation on the device; `0` host path |
| `DSV4_CUDA_WO_DECODE` (1) | output projection's decode intermediates on the device; `0` host |
| `DSV4_CUDA_COMPRESSOR` (0; 1 with `V4_LAYER_FUSED=0`) | dev's packed compressor copies and its device projection; the fused layer reads the plain bf16 mirrors |
| `DSV4_CUDA_RESIDENT_ROUTE` (1) | device-side routing/MoE for a complete 256-expert table; `0` host routing |
| `DSV4_CUDA_BATCHED` (1) | batched BF16 routing path; `0` legacy |
| `DSV4_CUDA_HEAD` (1) | exact vocabulary head on the GPU; `0` keeps it on the CPU |
| `DSV4_CUDA_HEAD_BATCH` (1) | batched exact head; `0` per-token |
| `DSV4_HEAD_VERIFY` (0) | `1` checks the batched head against the per-token path (diagnostic) |
| `DSV4_CUDA_RESIDENT_INDEXER` (0; 1 with `V4_LAYER_FUSED=0`) | upload indexer query weights once per layer for dev's device paths; the fused layer keeps its own |
| `DSV4_CUDA_INDEXER_HEADS` (1) | indexer score heads on the device; `0` the block kernel |
| `DSV4_CUDA_INDEXER_PREP` (1) | indexer preparation on the device; `0` host path |
| `DSV4_CUDA_WO_DECODE` (1) | output projection's decode intermediates on the device; `0` host |
| `V4_MTP_GPU_MIRRORS` (16) | separate mirror cache for the MTP drafter |
| `COLI_WIN_COMMIT_BOUND` (auto) | Windows: whether the commit still grantable bounds the RAM plan. Auto = only with a fixed page file; a system-managed one grows the commit on demand (measured: 9 GB of commit free beside 14.7 GB physical, and the plan fell to dense=streamed, 10x slower). `1` forces the bound, `0` lifts it |
| `V4_DECODE_GPU_PROJ` (1) | decode: the compressor's and the indexer-compressor's wkv/wgate projections on the card (the prefill kernel with a batch of one; needs the `COLI_CUDA_ATTN_BATCH=1` mirrors); `0` = the CPU loops. 3.3k context: attention 127 -> 111 ms/token. On an indexer layer the four projections share one round trip (`dsv4_cuda_matmul_bf16_multi`: indexer-advance 4 -> 2 ms/token) |
| `V4_IDX_FUSED` (1) | the indexer selection as one device round trip (decode and prefill chunks): the layer's `wq_b` bytes + f32 block scales in the CPU reference layout and its bf16 head-weight rows resident on the card (uploaded with the attention mirrors, 8 MB per indexer layer), the compressed keys appended as the host produces them, and projection -> query prep (bf16 round, RoPE, hadamard, fp4 round trip) -> head weights -> scoring in four launches; the sort stays on the host. Bitwise the CPU path: `DSV4_IDX_VERIFY=1` compares the prepared queries, head weights and scores (651 of 651 selections identical on the reference box, prefill chunks and decode tokens). 3.3k context: attention 111 -> 72 ms/token, decode 19.40 -> 18.20 s / 36 tokens, text unchanged. `0` = the projection replica + scoring kernel with their per-call uploads (8 MB of weights per indexer layer per token) |
| `DSV4_IDX_ROPE_FMA` (1), `DSV4_IDX_HW_FMA` (1) | which products the fused prep fuses, matching the GCC build of the reference (read off the binary: the RoPE's cosine products — `vfmaddsub` — and the head-weight accumulate are FMAs, nothing else). For a build with another contraction pattern (`DSV4_IDX_VERIFY` would report mismatches): `0` = none fused, ROPE `2` = the sine products instead |
| `DSV4_IDX_TILE` (1, `.cu`) | the fused indexer's packed projection replica a tile of 8 rows per block (the activation loads shared by the tile's sums; bitwise); `0` = a row per block |
| `DSV4_IDX_COAL` (1, `.cu`) | the fused indexer's projection replica and scoring kernel over transposed copies of the activations and the resident keys (coalesced loads, the same per-thread arithmetic); `0` = the row-major kernels. 3.3k prefill: projection 4.9 -> 1.2 s, scoring 1.2 -> 0.3 s |
| `V4_COMP_EMIT_PAR` (100000) | the column count from which the compressor's pool-and-emit forks an OpenMP region; below it the loop runs serially (the indexer's 128 and the attention's 512 columns: ~1.5 s less per 3.3k prefill, bitwise) |
| `DSV4_IDX_KPROF` (0, `.cu`) | per-call device time of the fused selection's stages (`idxkprof`); 0.16 + 0.01 + 0.10 + 0.09 + 0.02 ms per indexer layer at one token / 820 candidates |
| `V4_WO_CHAIN` (1) | decode: the attention from the query rank to the hidden vector in one device round trip — `wq_b` matvec, bf16 round, per-head rmsnorm, forward RoPE (the query never crosses), the cached sparse attention, inverse RoPE + bf16 round, the grouped `wo_a` and `wo_b` matvecs through the mirrors' own mv kernels (what the separate calls run: `coli_fp8_matvec_ref` dispatches to the mirror when one exists), one download of the bf16-rounded output. Four syncs per layer become one; arithmetic unchanged, `DSV4_WO_VERIFY=1` compares q, attended, `oa` and output bitwise (215 of 215 layer-tokens identical on the reference box). 3.3k context: attention 70 -> 56 ms/token, decode 17.9 -> 17.4 s / 36 tokens, text unchanged. `0` = the separate calls as before |
| `DSV4_WO_ROPE_FMA` (1), `DSV4_Q_NORM_FMA` (1) | the chain's fused products: the RoPE's cosine products (as `DSV4_IDX_ROPE_FMA`) and the query rmsnorm's mean-square accumulate, matching the GCC build of the reference |
| `DSV4_WO_KPROF` (0, `.cu`) | per-layer device time of the chain's stages (`wokprof`): rope 0.005, wo_a 0.14, round 0.004, wo_b 0.14, download 0.01 ms |
| `DSV4_CUDA_SCHED` (unset = runtime default, `.cu`) | how the host waits in a stream sync: `spin`, `yield`, `blocking`. Measured on the reference box: spin 19.35 s, yield 19.42, default 19.40, blocking 20.6 (36 tokens) — the default already spins there |
| `DSV4_CUDA_PIN_HOST` (1, `.cu`) | page-lock expert-cache slabs for DMA uploads |
| `DSV4_CUDA_MOE_GROUPED` (1, `.cu`) | expert-grouped MoE GEMM rows; `0` = legacy replicated |
| `V4_MOE_BANK_FULL` (off) | `=N`: prefetch a whole layer's experts above N tokens (measured worse) |
| `DSV4_MOE_EXPERT_MAJOR` (1, `.cu`) | generic batched MoE: expert-major fp4 kernel (weights read once per chunk); `0` = one matvec per route; `_VERIFY=1` compares both bit for bit (keeps the per-route results, also with `0`) |
| `DSV4_MOE_EM_TB` (auto, `.cu`) | routes per pass of the expert-major kernel: unset = 2 below 4 routes per routed expert in the chunk, else 4; `2`, `3`, `4` or `6` forces one (anything else is ignored with a warning) |
| `DSV4_MM_WARP` (auto, `.cu`) | generic dense batch matmul, fp8 shapes whose I is a multiple of 256: warp-per-row kernel for I ≤ 1024 (unset, empty or `auto`), `1` = all of them, `0` = off; `DSV4_MM_WARP_VERIFY=1` compares bit for bit (keeps the old kernel's results, also with `0`) |
| `V4_MOE_BATCH_PROF` (0) | `1` = one `moebprof` line per chunk-layer: switch (layer change: prefetch-worker wait, bank swap) / route / refill / compute ms |
| `V4_GPU_RESIDENT_FIRST` (0) | `1` = decode computes VRAM-mirrored experts without an expert-store lookup, and the RAM pins skip them (VRAM adds to the RAM tier); off under `DSV4_HYBRID=1` |
| `DSV4_CUDA_PEER_DEVICE` (unset) | second GPU for a decode mirror pool (peer access required; unused under `DSV4_HYBRID=1`) |
| `DSV4_CUDA_PEER_COMPUTE` (1, `.cu`) | run the peer pool's experts on the peer GPU (both GPUs at the same SM version; `mode=` in the `peer-mirrors` line); `0` = read them in place over P2P; no effect with `DSV4_CUDA_BATCHED=0`, which reads them in place |
| `DSV4_CUDA_PEER_FAULT` (unset, `.cu`) | tests: `n` = the n-th peer-compute pass fails at its join with an error the devices survive, and the peer is latched off (that call recomputes on the primary); `n:lost` = the peer also counts as lost (the call fails, decode takes host copies); `n:sticky` = the error is a kernel fault's, which latches CUDA off for the whole process (every later call refused, host paths), as a real fault does |
| `DSV4_CUDA_PEER_MIRRORS` (4096) / `DSV4_CUDA_PEER_RESERVE_MB` (1024) | upper bound on peer pool mirrors (12.75 MiB each; free VRAM sizes the pool), minimum 8 (a smaller or non-numeric value leaves the pool off) / VRAM kept free on the peer |
| `DSV4_CUDA_PIN_CAP_GB` (20, `.cu`) | cap on page-locked expert-slab bytes for uploads, in GiB; negative = 0 (nothing pinned), empty or unparsable = 20, at most 1 PiB (no cap) |
| `V4_SERVE_STATS_LOG` (0) | `1` = per-turn `v4_serve_turn` (the turn's DONE/PROF numbers, VRAM-first totals) and `v4_dec_moe` (decode MoE totals) lines on stderr |

**Prefill / checkpoints**
| var | meaning |
|---|---|
| `V4_PREFILL_CHUNK` (128) | chunk width, clamp [1,128] |
| `V4_PREFILL_SEGMENT` (4096) | atomic segment length (cancel/resume granularity, expert re-sweeps) |
| `V4_PREFIX_CKPT` (1) / `V4_PREFIX_CKPT_MIN` (512) / `V4_PREFIX_CKPT_SLOTS` (4) / `V4_PREFIX_CKPT_DISK` (1) | prefix checkpoints, see above |
| `V4_PREFIX_LOG` | log hint boundary / reuse decisions |
| `V4_IDX_BATCH` (1) | batched indexer selection in prefill; `0` = legacy per-token |
| `V4_IDX_IDENTITY` (0) | `1` = skip indexer scoring when every candidate fits under `index_topk` (index order instead of upstream's score order; ~8 % faster prefill, changes rounding) |
| `COLI_V4_ROWS16` (1) | `0` = never repack hot experts into the rows16 layout (reference matvec for all; numerics comparisons) |
| `CTX` (4096) / `NGEN` | context window / generation ceiling (CLI + serve) |

**Experts / I/O**
| var | meaning |
|---|---|
| `COLI_MODEL_MIRROR`, `SNAP_MIRROR`, `COLI_DISK_WEIGHTS` | dual-SSD read split (see ENVIRONMENT.md) |
| `V4_MOE_REFILL_GROUP` (6, max 16) | parallel lookups per bank refill group (bounded by pin slots; 12 measured best) |
| `V4_LOADER_LANES` (9, max 16) | persistent expert-loader threads for decode. The default suits the CPU path (1.41× decode, #1097); **with the GPU tier set `3`**: mirrors/RAM absorb most decode misses (9 lanes ≈ +4 % decode) while the extra readers contend with the MoE bank refill for the NVMe queues (~10 % slower prefill; measured on the reference box, v1.7.0) |
| `V4_EXPERT_UNION` (1) | per-chunk expert union batching on the CPU path |
| `COLI_V4_DIRECT` (1) | O_DIRECT streaming reads |
| `COLI_V4_AUTOPIN`, `COLI_V4_SAVE_USAGE` | hot-expert pinning from `.coli_usage` history |
| `COLI_V4_PREWARM` (unset) | the expert cache filled before the first request: unset, a serve (chat, web) reads every expert its RAM plan holds, or with a smaller cache each layer's most used ones in `.coli_usage`, before it says ready, in parallel with a progress line, and stops short of the RAM the plan keeps for the system; a one-shot run fills it as experts are used. `0` skips it, `1` fills it in every run (#2109) |

**Speculative decoding** (off by default): `V4_DRAFT`, `V4_MTP`, `V4_MTP_GB`,
`V4_MTP_MIN`, `V4_MTP_DRAFT`, `V4_MTP_PARTIAL_KEEP`, `V4_NGRAM` (1),
`V4_NGRAM_PARTIAL_KEEP`, `COLI_V4_MARKOV_SPEC`, `COLI_V4_MARKOV_BLOCK`,
`COLI_V4_MARKOV_KEEP`.

Why it is off: DSpark's markov drafter and full MTP are both implemented and
verified. A draft can save forward passes but never change a token, because
every accepted token is still the target's own argmax. Measured on real
multi-turn chat, they accepted 1 in 15 and 10 in 24, and the rejected-suffix
replay of this engine's recurrent attention state cost more than the drafts
saved: one 14-token answer took 495 seconds. So `V4_DRAFT` and `V4_MTP` default
to `0` and the code stays, with the numbers beside it, for whoever retries this
on faster storage.

**Diagnostics**: `DSV4_ATTN_PROF`, `DSV4_DECODE_PROF`, `DSV4_IDX_VERIFY`,
`DSV4_CUDA_MOE_PROF`, `DSV4_CUDA_DG_PROFILE`/`_AB`/`_DUMP` (`.cu`),
`COLI_NO_OMP_TUNE`, `OMP_NUM_THREADS`.

Note: `COLI_CUDA`, `COLI_GPU`, `CUDA_DENSE`, `COLI_CUDA_PIPE` in the serve
line are read by the CLI/`colibri.c` conventions; the V4 engine's own switch is
`DSV4_CUDA` and the two `COLI_CUDA_*_BATCH` gates above.

## Validation

Tiny fixture (generated locally, ignored):

```bash
python -m pip install -r tools/requirements-deepseek-v4-tiny.txt
make deepseek-v4-tiny-check
```

Real checkpoint oracle:

```bash
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash \
  MEMORY_GB=32 ORACLE_TEACHER_FORCING=32 ORACLE_GREEDY=20
```

GPU tier evidence (2026-08): each stage was accepted with greedy text
byte-identical to the engine's own reference chain at the time (826- and
3324-token benchmark prompts, 8 tokens; strip the CLI's trailing
`TUNE decode:` line before comparing); the GPU indexer projection/scoring and
the CPU-replica fp8 matmul are checked bitwise by `DSV4_IDX_VERIFY=1` (the fused
selection too: queries after the device prep, head weights, scores — the
device prep reproduces the CPU build's FMA pattern, see `DSV4_IDX_ROPE_FMA`);
the CUDA numerics test is `c/tests/test_dsv4_sparse_attn_batch_cuda.c`. Different
kernels are not bit-identical to each other in general — a GPU run and a CPU
run of the same prompt diverge by a rounding flip after some tokens, exactly
as two CPU runs with different hot-expert sets do (next section) — so text
identity is a regression check within one configuration, not a proof across
configurations. The same holds across builds: a binary built with
`ARCH=native` and the release's (`x86-64-v3`) can differ by such a flip, the
compiler having chosen other instructions (FMA contraction, vector width) for
the same arithmetic (#1906 saw one word change after 384 characters), so a
check of exact text compares runs of one binary.

### Linux CUDA tier under WSL2 (2026-08-16)

Ubuntu 22.04 on WSL2 (15 GB guest RAM, model read over `/mnt/c` + `/mnt/d`,
`--memory-gb 11`, canonical GPU env), prompt "Say hello in exactly one short
sentence.", 12-token ceiling. Both Linux binaries produced `Hello!` + EOS,
the same as the Windows engine on the same prompt; the wall-clock gap is the
9p file system (expert streaming), not the tier:

| build | tier | ttft | 3 decode tokens |
|---|---|---|---|
| Linux `CUDA=1 DEEPGEMM=1` (sm_120a) | `[DSV4 CUDA] device 0 ... sm_120`, dense mirrors 6.27 GiB | 119.9 s | 32.4 s |
| Linux `CUDA=1 CUDA_ARCH=portable` (generic) | same | 122.1 s | 33.2 s |
| Windows `deepseek_v4.exe` + `coli_cuda_dsv4_dg.dll` (NTFS, 2× NVMe) | same | 16.8 s | 2.0 s |

A native Linux box with the checkpoint on local NVMe is still the missing
measurement (host-slab pinning is Windows-only, so refill is pageable there).

### CPU-only audit (2026-08-15/16)

Question: does the CPU path still compute what upstream `dev` computes after
the GPU work? Method: 826-token prompt, 48 greedy tokens, real checkpoint,
upstream `dev` engine built from its own tree vs this tree's engine built
without the GPU tier (the Linux/macOS build shape) and with the tier
disabled.

- Compiles clean without `COLI_V4_GPU_TIER`; the tiny torch-reference suite
  (`tests/test_deepseek_v4_tiny.py`) is token-exact on the CPU-only build for
  every target case.
- On the real checkpoint, greedy text is **not** a stable identity check
  across configurations, in upstream itself: hot (cache-resident) experts run
  the vectorized `rows16` kernel while cold ones run the reference matvec, the
  two accumulate in different orders, and which experts are hot depends on the
  autopin history (`.coli_usage`, rewritten by every run), the chunk width and
  the hit pattern. Upstream produced different 48-token texts with and without
  its history; so did this tree; every variant is a coherent answer differing
  by a rounding flip a few tokens in.
- With that variable removed on both sides (`COLI_V4_ROWS16=0`
  `COLI_V4_AUTOPIN=0 COLI_V4_SAVE_USAGE=0`: reference kernels only, no
  history) **upstream and this tree's CPU-only build produce identical
  48-token text** — the CPU math is unchanged.
- One change of this branch did alter CPU accumulation order and is now off
  by default: the indexer identity short-circuit for `count <= index_topk`
  (returned candidates in index order; upstream returns them in score order,
  and the sparse attention reference sums in that order). `V4_IDX_IDENTITY=1`
  re-enables it (~8 % faster prefill on GPU, e.g. 96 s -> 88 s at 3.3k).
- CPU-only prefill speed: chunk 128 vs 64 identical within noise (788 vs
  797 s for 826 tokens); the pinned-slab, pipelined refill and all kernel
  work are GPU-only and do not touch the CPU path.
- Degenerate config guarded: a decode expert mirror cache below 8 entries
  (`DSV4_CUDA_EXPERT_MIRRORS<8`) recycled a slot still referenced within one
  layer and produced garbage; such caches now stay on the CPU.

## Follow-ups

- Non-greedy sampling and more serving slots.
- Linux CUDA tier: measure on a native Linux box (WSL2 verified), POSIX host pinning.
- Expert-parallel prefill across GPUs (the second card hosts decode mirrors
  only today); three or more cards.
- Tensor/expert parallel execution within a layer (not provided by layer placement).
- Shared replacements for the two temporary private quant paths (rows16 cache).

### Resident decode experts

The CUDA decode pipeline checks the owning device's expert cache before starting
host expert loaders. If every routed expert and the shared expert are mirrored,
it runs the fused MoE directly from those device weights. The cache mutex stays
held until the synchronous backend returns, preventing refill/eviction of the
selected buffers. Partial or incomplete residency and declined fusion use the
existing loader/group fallback. Shared expert bindings use w1 (gate), w3 (up),
and w2 (down).

This adapts Naruto's resident expert-table execution approach
(`core/model/deepseek_v4_moe.cc`, `DeviceRoutedExpertsInvocation`, revision
6634772) to Colibri's existing CUDA mirrors. It does not import Naruto's TP2,
EP2, SM121 kernels, or graph runtime. Cold loads and partial-hit tokens still
use host expert storage; this is not full model preloading.

Validation covers six-device cache ownership, full/partial/incomplete residency,
locking through compute, backend-decline fallback, and CPU compilation. The subsequent
[six-RTX-5090 test](experiments/dsv4-resident-2026-09-27.md) measured about 3.8%
median decode improvement against an otherwise identical fused-path control.
Its output matches that control, but differs from the original unfused baseline;
this is not a claim of numerical equivalence or general model-quality validation.

### Full routed-expert residency (automatic)

With the CUDA build, the engine automatically loads every target routed expert
onto its layer's owning device when the resident runtime tier and per-device
VRAM budgets allow it. `DSV4_CUDA_RESIDENT_EXPERTS=auto` is the default; `0`
keeps the LRU cache, and `1` requires full residency or rejects engine open.
Insufficient VRAM or an unsupported tier in automatic mode retains the cache
path and logs the reason. Other loading failures still reject engine open. The immutable table is indexed directly by layer
and expert, following Naruto's prebuilt device expert-table approach. Target
experts are never evicted; MTP experts retain their separate cache.

The loader uploads dense mirrors first, checks every device's available VRAM
against its complete expert footprint plus `DSV4_CUDA_RESIDENT_RESERVE_MB`
(default 2800 MiB, minimum 256 MiB), then uploads experts. That reserve is
separate from `DSV4_CUDA_VRAM_RESERVE_MB`, which sizes the decode mirror cache's
bank-aware growth guard. The resident reserve overrides the LRU mirror capacity:
the table has exactly one slot per assigned expert. A failed host lookup,
unsupported tensor layout or upload aborts engine open and releases partial
allocations; in `auto` those hard failures still reject the engine, while a
budget refusal keeps the cache. Publication occurs only when all cards have
completed loading. A resident execution failure is reported rather than
silently loading experts on the CPU.

Prefill currently uses the per-token GPU MoE pipeline with these same tables,
not the CPU expert union or a duplicate streaming expert bank. This removes
expert host reads after preload but does not implement batched resident MoE,
TP/EP, device-only activations, or SM120 MMA kernels. Startup reads the complete
expert set; cold-start latency and steady-state decode must be measured
separately. Host expert-cache memory is still governed by the existing budget.

On six RTX 5090 GPUs, a short 64-token test measured 6.155 token/s median
(three runs) versus 2.149 with the option off (one run), with identical output.
Preloading took about 73 seconds and increased total cold-process latency.
See the [full report and raw results](experiments/dsv4-full-resident-2026-09-27.md)
for the scope, startup cost, and remaining validation limits.

On 24 GB cards the complete table can exceed the budget. Measured at the
2026-11-03 head on two RTX 3090s (500/450 W), both the full 256-expert model and
the 150B REAP checkpoint logged `resident-experts=cache-fallback` at
`free≈16.6 GiB reserve=2800 MiB`; the gains there are kernel-side. REAP decoded
1.22x one GPU and 1.54x two GPUs against dev one GPU; the full model 1.19x and
~1.36x; long-prompt TTFT 0.89x. The exact head (`DSV4_CUDA_HEAD`) goes to the
owning (last) device with room, then any other device, then the CPU.

### Device-resident routing (default with full residency)

For the supported 256-expert/top-6 profile, the resident table also owns a small
borrowed-weight CUDA descriptor table per layer. Routing, ascending expert-ID
ordering, descriptor selection and MoE now execute on one device stream: there
is one activation upload and one result download, without copying top-k results
back to the host. The existing BF16 rounding and expert accumulation order are
preserved; no extra input quantization is introduced.

`DSV4_CUDA_RESIDENT_ROUTE=0` selects the previous host-routed path. The default
uses device routing when the complete table exists. An older Windows DLL can
decline the optional new entry point and retain the original path. Explicit
`DSV4_CUDA_BATCHED=0` and builds disabling BF16 routing retain their original
execution paths. Attention, mHC, layer boundaries and the output head are not
made device-only by this change.

The [device-routing validation](experiments/dsv4-device-route-2026-09-27.md)
covers exact kernel parity, six-device default selection, single-device budget
fallback, and the remaining Attention bottleneck. Measured short-request decode
remains around 6.1–6.2 token/s; the route-only comparison is too small to claim a
stable throughput gain.

### Native quantization and resident indexer projections

The generic CUDA path uses native E4M3 conversion instead of searching 255
codes per activation, and bit decoding for E2M1. Ties-to-even, positive zero
from the old encoder, and the existing activation scaling remain unchanged.
SM89+ uses hardware E4M3 decoding; older targets use the equivalent bit layout.
Resident routing uses the existing warp Top-6 kernel, retaining the original
expert-ID tie break and sequential normalization of the six selected weights.

With batched CUDA attention enabled, indexer query weights now upload once
during layer loading. These mirrors preserve the original rows8/row-major
layout, float scales, and exact reference reduction order; they are not generic
FP8 GEMM mirrors. They count toward dense residency before full expert budget
checks and are released with their owning layer on the assigned device.
`DSV4_CUDA_RESIDENT_INDEXER=0` retains per-call weight uploads. Missing optional
DLL exports or unavailable mirrors retain that original fallback.

These changes are default behavior for the applicable CUDA paths. They do not
enable speculation, tensor parallelism, or a device-only Attention pipeline.
See the [Naruto comparison and measurements](experiments/dsv4-naruto-gap-2026-09-28.md).
