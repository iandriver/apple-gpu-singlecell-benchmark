# Single-cell preprocessing on the Apple GPU: a feasibility benchmark

**Can you accelerate a single-cell RNA-seq preprocessing pipeline on an Apple
Silicon GPU (Metal), the way [rapids-singlecell](https://github.com/scverse/rapids-singlecell)
does on NVIDIA GPUs with RAPIDS/CUDA?**

Short answer, measured on an **Apple M5 Pro**: **not usefully, today** — and the
reason is more interesting than "Macs are slow." This repo contains the
standalone, reproducible benchmark and probes behind that conclusion.

> Origin: this started as a feasibility study for porting rapids-singlecell to
> Apple GPUs. It is **not** affiliated with that project and is not intended as a
> pull request — it's published for general interest because the findings about
> the Apple-GPU numerical stack apply to any array-heavy scientific workload.

## TL;DR

![Speedup by step: memory-bound elementwise steps win on the GPU, compute-bound PCA/KNN do not](results.png)

| Step | Bound by | CPU (scanpy/sklearn) | Apple GPU (PyTorch-MPS) | |
|---|---|--:|--:|--|
| normalize_total + log1p | memory | 161 ms | **15 ms (10.7×)** | GPU wins |
| scale (z-score + clip) | memory | 157 ms | **28 ms (5.7×)** | GPU wins |
| PCA (50 comps) | compute | 321 ms | 230 ms (1.4×) | ~wash |
| exact KNN (k=15) | compute | 581 ms | 6867 ms (0.08×) | **GPU loses** |

*(MPS = kernel time with data resident on the GPU; transfer-inclusive numbers and
methodology are in [RESULTS.md](RESULTS.md).)*

**The result inverts the naive expectation.** You'd guess the cheap elementwise
steps wouldn't benefit (the CPU and GPU share one unified-memory bandwidth pool)
and the heavy linear-algebra steps would. The opposite happened:

- The **cheap elementwise steps won big** — but mostly because scanpy's CPU path
  is effectively single-threaded, and the absolute savings (~150 ms → ~30 ms) are
  trivial in a real pipeline.
- The **expensive steps that actually dominate runtime — PCA, neighbors, UMAP,
  clustering — are exactly what the Apple GPU can't do**, because there is **no GPU
  eigendecomposition / SVD / QR on Apple Silicon today**, in *either* major
  framework:

  | routine | PyTorch-MPS 2.12 | MLX 0.31 |
  |---|---|---|
  | `qr` (tall) | **hangs** | CPU-only (`ValueError`) |
  | `svd` | silent CPU fallback | CPU-only (`ValueError`) |
  | `eigh` | `NotImplementedError` | CPU-only (`ValueError`) |
  | `pca_lowrank` | **hangs** | n/a |

So the blocker is not unified-memory bandwidth and not the choice of framework —
it's a **gap in the entire Apple-GPU numerical stack**. The steps that are easy to
move to Metal aren't worth moving; the steps worth moving can't be moved.

## Why this matters beyond single-cell

Any GPU-accelerated scientific Python workload that leans on SVD / eigendecomposition
/ QR (PCA, spectral methods, least-squares, many ML algorithms) hits the same wall
on Apple Silicon right now. Apple's `Accelerate`/LAPACK is CPU-only; getting these
onto the GPU currently means hand-writing Metal kernels (e.g. a Jacobi eigensolver)
or waiting for the frameworks to ship GPU linalg.

## Run it yourself

Requires Apple Silicon + macOS. Uses [`uv`](https://github.com/astral-sh/uv) for a
clean isolated environment (any venv works):

```bash
uv venv --python 3.12 .venv
uv pip install --python ./.venv/bin/python -r requirements.txt
uv pip install --python ./.venv/bin/python mlx   # for the MLX probe only

./.venv/bin/python bench.py              # the 4-step pipeline benchmark
./.venv/bin/python probe_mps_linalg.py   # which PyTorch-MPS linalg ops work
./.venv/bin/python probe_mlx_linalg.py   # which MLX linalg ops run on GPU
```

The benchmark uses synthetic data (50k cells × 20k genes, ~7% dense) so it runs
anywhere with no download.

## Repo contents

| File | What |
|---|---|
| [`bench.py`](bench.py) | The benchmark: 4 pipeline steps, each timed on CPU vs Apple GPU, with warm-up, MPS synchronization, transfer-cost accounting, correctness checks, and a watchdog so a hung kernel can't lock the run. Heavily commented. |
| [`probe_mps_linalg.py`](probe_mps_linalg.py) | Shows which PyTorch-MPS linalg routines work, fall back to CPU, or hang. |
| [`probe_mlx_linalg.py`](probe_mlx_linalg.py) | Shows that MLX's `svd`/`qr`/`eigh` are GPU-unsupported. |
| [`make_chart.py`](make_chart.py) | Regenerates `results.png` (the chart above) from the measured numbers. |
| [`RESULTS.md`](RESULTS.md) | Full results, interpretation, and methodology notes. |
| `*.log` | Raw captured output from the runs on an M5 Pro. |

## Methodology notes (important caveats)

- **Hardware:** Apple M5 Pro, 20-core, 48 GB unified memory, macOS, Metal 4.
  Numbers will differ on other chips, but the *linalg gap* is platform-wide.
- **Timing hygiene:** MPS is asynchronous, so every GPU timing is bracketed by
  `torch.mps.synchronize()`; the first launches are discarded as warm-up; results
  are medians of repeats.
- **Fairness:** the elementwise GPU win is partly an artifact of scanpy's
  single-threaded CPU path — a well-threaded CPU implementation would narrow it.
  Read it as "GPU vs stock scanpy," not "GPU vs the best possible CPU code."
- **KNN:** compared as *exact* brute-force on both sides for a fair kernel
  comparison. Production scanpy uses *approximate* neighbors (pynndescent), which
  is faster than the exact CPU baseline shown — so the GPU loses by even more in
  practice.

## When to revisit

The day a GPU eigensolver / SVD lands in PyTorch-MPS or MLX, the PCA verdict flips
and this becomes worth re-running. Track the PyTorch MPS and MLX linalg issue
trackers. `bench.py` re-measures it in one command.

## License

MIT — see [LICENSE](LICENSE).
