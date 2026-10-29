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
sizes it at run time — you normally do not need to set it. Free VRAM is
re-read every 64 mirror misses; in between, an explicit reserve and the 600 MiB
headroom also count the mirrors grown since the last reading, so the cache
stops at the reserve instead of up to 64 mirrors (~800 MiB) past it, where the
compressed-KV buffer and the prefill's attention buffers no longer fit and
their layers' attention falls back to the CPU (`[DSV4 CUDA] kv comp
allocation: out of memory`). The bank-sized default below already carries
those 64 mirrors as margin, so its decisions do not change. Decode is
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

The runtime check `dsv4_cuda_backend_arch_ok` admits sm_60 and up for the
generic build. It is deliberately independent of what the binary contains: a
`CUDA_ARCH=portable` build has no sm_61/sm_75 cubin, and running it on such a
card fails at launch with `"no kernel image is available"` rather than producing
a wrong answer. Build with `portable-pre-ampere` for those cards.
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
| `V4_HC_FUSED` (1) | the prefill block's attention post + FFN pre/norm as one device pass, the states resident for the FFN post (16 MB per chunk-layer not copied); `0` = the two-step path with the states round trip |
| `DSV4_SATTN_HPB_PF` (16) | heads per block for the multi-token (prefill chunk) launches of the cached sparse attention: 16, 8 or 4 (else the one-token rule); the key rows are shared across the heads of a token, the per-head arithmetic is unchanged (bitwise) |
| `DSV4_PIN_STAGE` (1, `.cu`) | the prefill chain's inputs as one copy from a page-locked staging block and its output download into it; `0` = seven pageable copies and a pageable download |
| `DSV4_PF_KPROF` (0, `.cu`) | per-call device time of the prefill chain's stages and its host wall (`pfkprof`) |
| `DSV4_SATTN_HPB` (4) | heads per block of the shaped sparse attention from 256 rows up (`1` = one head per block, the rows read once per head) |
| `DSV4_SATTN_V2` (1) | the cached sparse attention kernel shaped for latency (same arithmetic, same order, bitwise); `0` = the old kernel |
| `DSV4_ATTN_MV` (1) | decode: the attention chain's fp8 matvecs (wq_b, wo_a, wo_b) on the bandwidth-shaped kernel; `0` = the byte-per-lane kernel, bitwise the reference (slower by ~0.27 ms per layer) |
| `V4_UPLOAD_PIECES` (4) | decode: a lane's expert read lands in this many pieces, all in flight at once, each copied to a mirror reserved before the read as it lands (the link works under the read); the slab is page-locked before the read. `0`/`1` = one read, the uploader thread copies after it. Needs `V4_MIRROR_BLOCK`, `V4_UPLOAD_FENCE` and the DLL's `dsv4_cuda_mirror_copy`/`_point`/`_host_pin`. `V4_PIECES_VERIFY=1` compares each piece-built mirror with its slab, `=8` compares each slab with a buffered re-read (diagnostics) |
| `V4_UPLOAD_FENCE` (1) | decode: the uploader threads only enqueue their copies; the expert group waits for them on the device (`dsv4_cuda_group_fence`, aux-stream event) instead of the host joining the DMA before launching. Needs the export in every backend module in use (else, or `0`: drained as before) |
| `DSV4_MOE_PINNED` (1, `.cu`) | pinned staging for the decode group's small copies (x, descriptors + weights in one transfer, y); `0` = pageable |
| `DSV4_MOE_MV` (1, `.cu`) | decode expert group / fused MoE: bandwidth-shaped fp4 and fp8 matvecs (one warp per row, 16-byte loads; kernels 1.14 → 0.29 ms per layer on the 5080); `0` = the byte-per-lane `mv_fp4_grouped` / `mv<8>` kernels (a different summation order) |
| `DSV4_MOE_MV_VERIFY` (0, `.cu`) | every fused MoE call recomputed through the old kernels and compared (`moemv verify`: max abs diff, bf16-ulp histogram) |
| `DSV4_MOE_KPROF` (0, `.cu`) | per card: device time of the group's kernels vs the call's host wall (`moekprof`, every 430 calls) |
| `V4_UPLOAD_THREADS` (1) | decode attach on one uploader thread per card (aux-stream copies, joined before compute); `0` = attach inline from the lane loop |
| `V4_READ_STRIPE` (off) | `1` = read each expert window as two halves from the two mirror replicas at once (Windows overlapped twins). Measured with `COLI_MODEL_MIRROR` on a second NVMe (826-token prompt, 60 tokens): decode 24.2 -> 21.3 s (-12 %), prefill 52.4 -> 44.6 s (-15 %), text identical; needs a real second drive — two copies on one drive gain nothing |
| `V4_PREFETCH_NEXT` (off) | `=N` lanes reading the next layer's experts as the previous token routed them into the RAM cache while this layer runs; measured useless here: those repeats (41 % of a layer's experts) are already VRAM-resident or RAM-cached, so nothing is ever queued |
| `COLI_DSV4_DLL` | Windows: force a backend DLL file name (loader) |
| `COLI_CUDA_ATTN_BATCH` (0; 1 for a GGUF on the CUDA expert tier) | `1` = GPU batched prefill attention block + GPU decode attention/indexer; unset, it is on for a GGUF whose dense matrices the CUDA expert tier's device holds (`COLI_CUDA=1`, `COLI_CUDA_DENSE`), `0` keeps it on the CPU |
| `COLI_CUDA_MOE_BATCH` (0) | `1` = prefill MoE on the transient VRAM expert bank |
| `COLI_CUDA_MOE_BATCH_MIN` (256) | min fresh tokens to engage the bank |
| `DSV4_CUDA_EXPERT_MIRRORS` (4096) | upper bound on decode VRAM expert mirrors (12.75 MiB each); free VRAM sizes the cache at run time |
| `DSV4_CUDA_VRAM_RESERVE_MB` (600) | VRAM kept free on the primary card while decode mirrors grow. With `V4_BANK_MAKE_ROOM=1` (default) the prefill expert bank is not reserved: it releases LRU mirrors when it is created (RTX 5080, 3.3k context: 2800 -> 600 MB gave 18 % fewer uploads, decode -5 %, text identical); `V4_BANK_MAKE_ROOM=0` restores the bank-sized default reserve (600 + the bank, x2 with `COLI_CUDA_MOE_DOUBLE=1`, + 64 mirrors, logged as `v4_gpu mirror-reserve=`). An explicit value is used as given |
| `V4_MTP_GPU_MIRRORS` (16) | separate mirror cache for the MTP drafter |
| `COLI_WIN_COMMIT_BOUND` (auto) | Windows: whether the commit still grantable bounds the RAM plan. Auto = only with a fixed page file; a system-managed one grows the commit on demand (measured: 9 GB of commit free beside 14.7 GB physical, and the plan fell to dense=streamed, 10x slower). `1` forces the bound, `0` lifts it |
| `V4_DECODE_GPU_PROJ` (1) | decode: the compressor's and the indexer-compressor's wkv/wgate projections on the card (the prefill kernel with a batch of one; needs the `COLI_CUDA_ATTN_BATCH=1` mirrors); `0` = the CPU loops. 3.3k context: attention 127 -> 111 ms/token. On an indexer layer the four projections share one round trip (`dsv4_cuda_matmul_bf16_multi`: indexer-advance 4 -> 2 ms/token) |
| `V4_IDX_FUSED` (1) | the indexer selection as one device round trip (decode and prefill chunks): the layer's `wq_b` bytes + f32 block scales in the CPU reference layout and its bf16 head-weight rows resident on the card (uploaded with the attention mirrors, 8 MB per indexer layer), the compressed keys appended as the host produces them, and projection -> query prep (bf16 round, RoPE, hadamard, fp4 round trip) -> head weights -> scoring in four launches; the sort stays on the host. Bitwise the CPU path: `DSV4_IDX_VERIFY=1` compares the prepared queries, head weights and scores (651 of 651 selections identical on the reference box, prefill chunks and decode tokens). 3.3k context: attention 111 -> 72 ms/token, decode 19.40 -> 18.20 s / 36 tokens, text unchanged. `0` = the projection replica + scoring kernel with their per-call uploads (8 MB of weights per indexer layer per token) |
| `DSV4_IDX_ROPE_FMA` (1), `DSV4_IDX_HW_FMA` (1) | which products the fused prep fuses, matching the GCC build of the reference (read off the binary: the RoPE's cosine products — `vfmaddsub` — and the head-weight accumulate are FMAs, nothing else). For a build with another contraction pattern (`DSV4_IDX_VERIFY` would report mismatches): `0` = none fused, ROPE `2` = the sine products instead |
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
- Shared replacements for the two temporary private quant paths (rows16 cache).
