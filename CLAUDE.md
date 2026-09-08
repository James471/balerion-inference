# balerion-inference

UltraNest nested sampling over a trained Keras emulator (SMF regressor), replacing an expensive
simulator likelihood with a neural network forward pass.

## Files

- `config.py` / `config.toml` — machine-local paths (`models_dir`, `inference_dir`). Copy
  `config.toml.example` to `config.toml` and fill in the paths for your machine; `config.toml` is
  gitignored. On gadi `inference_dir` is `/g/data/cm25/jw5893/dragons/inference` — keep it off
  /home, a full sweep is 24 UltraNest log dirs.
- `run_vectorized.py` — main entry point. Runs `ReactiveNestedSampler` with `vectorized=True`
  against `log_likelihood`, which calls the emulator and computes normalized residuals against
  Stefanon et al. 2021 SMF data (`data/stefanon2021_z*.json`) across redshifts 6-10.
  - CLI: `python run_vectorized.py <u|p> [--model PATH] [--max-ncalls N] [--output-dir DIR]
    [--resume MODE]`. The positional arg selects the "uniform" or "posterior" flavour, which fixes
    `threshold_smf` (a property of the training set, not of the architecture). `--model` picks the
    emulator, defaulting to the deepest arch; `--output-dir` defaults to
    `<inference_dir>/<flavour>/<arch>/<repeat>` derived from the model path. `--max-ncalls` caps
    likelihood calls for repeatable profiling runs; omitted, the sampler runs to convergence.
  - `--resume` defaults to `resume` (not `overwrite`), so a job killed at the walltime continues
    from UltraNest's saved state when resubmitted. This is what makes the sweep's requeue work.
  - Writes `result.json` only after `sampler.run()` returns, so its presence is the "this run is
    done" marker the sweep scripts key off. `run_info.json` alongside it records which model,
    threshold and call count produced it.
- `sweep/sweep.py` — drives the full architecture sweep on gadi: one run per (flavour, arch) over
  every arch in `<models_dir>/{uniform,posterior}/reg_arc/depth/`, 24 in total.
  - `./sweep/sweep.py status` prints the done/running/todo table; `submit` writes each job script
    into that run's own output dir and qsubs enough of them to fill the free GPU slots; `submit -n`
    reports without writing anything. `submit` is idempotent — rerun it to top the queue back up.
  - Per-run bookkeeping lives in `<output_dir>/.sweep/`: `job.sh`, `job_<attempt>.o` (one log per
    attempt, so a requeue after a walltime kill doesn't overwrite the previous log), `attempts`
    and `jobid`. Nothing sweep-related is written into the repo.
  - `run_status` decides "did this run actually start" by looking for any entry in the output dir
    **not** prefixed `.sweep` — that is what keeps the job script and log it just wrote from being
    mistaken for sampler state. See the duplicate-submission note below.
  - Runs under a bare `python3` (tested on the login node's 3.10): `config.toml` is read with a
    regex, not `tomllib`, so it doesn't need 3.11+ or the `tomli` backport on whatever node it
    lands on. Only stdlib.
  - Job names are `bal_<u|p>_<md5(arch)[:4]>`, keyed on the architecture rather than its position
    in the sorted list, so training a new architecture doesn't renumber jobs already queued.
    Short enough that `qstat -f` never wraps them, which would break the state parse.
  - `queued_jobs()` treats a non-zero `qstat` exit as fatal rather than as an empty queue —
    otherwise a PBS outage would look like "no jobs running" and `submit` would fire duplicates
    of everything in flight.
  - One GPU per job, capped at 2 uniform + 2 posterior in the queue at once (4 GPUs total).
    UltraNest's batches top out at the live-point count (~400), so a second GPU has nothing to do.
    Two caps: `MAX_JOBS_PER_FLAVOUR` keeps both flavours progressing together, `MAX_TOTAL_JOBS`
    bounds GPU use. Once a flavour runs out of eligible work the other spills into its slots
    rather than idling them. Override per-invocation with `--max-total` / `--max-per-flavour`.
  - Architectures are ordered smallest-first, so cheap points land early. (The resulting order
    matches `depth_list` in `model_scripts/reg_arc.py` exactly.)
  - `REPEAT=0` is the `<i>` in `.../depth/<arch>/<i>/` — balerion's `StatisticalTest` retrains
    each architecture `num_tests` times and numbers the runs from 0. It is *not* an RNG seed;
    nothing in the balerion package seeds anything. `reg_arc.py` used `NUM_TESTS=1`, so `0` is
    the only one that exists today.
  - A run whose output dir exists without `result.json` and has no live job is `INCOMPLETE` and
    gets resubmitted (UltraNest resumes), up to `MAX_ATTEMPTS` tracked in `.sweep_attempts`, after
    which it reports `STUCK` and is left alone.
  - PBS resources, the venv, and the per-flavour cap are the variables at the top of the script.
    `gpursaa` routes to the four `gadi-gpu-rsaa` nodes, which are 56 cores / 4 GPUs / 512GB — not
    the 48-core gpuvolta layout — so a quarter-node per GPU is `ncpus=14, mem=128GB` and four
    jobs tile one node exactly. Check `pbsnodes -a` before assuming the standard 12:1 ratio.
  - Single-process only (no MPI). The emulator dominates wall-clock time (per profiling), and
    when it runs on GPU there's nothing to gain from multiple sampler ranks — they'd just
    contend for the same GPU context. Detects GPU via `tf.config.list_physical_devices('GPU')`
    and pins the emulator load + every `predict()` call to it with `tf.device(...)`, falling
    back to CPU otherwise.
  - Has built-in profiling (`PROFILE` dict / `print_profile_summary()`) that reports emulator
    time vs. residual-bookkeeping time vs. batch-size distribution UltraNest actually produces —
    printed even on Ctrl-C/exception.
- `benchmark_emulator_device.py`, `plot_benchmark_results.py` — standalone CPU/GPU inference
  speed benchmarks for the emulator, decoupled from the sampler. Writes/reads
  `benchmark_results_{cpu,gpu}.json` and `benchmark_results.png` / `benchmark_speedup.png`.
- `posterior_comparison.ipynb` — notebook, not otherwise wired into the scripts above.

## Gotchas

- Models are `.keras` files (Keras 3 native zip format). Loading them requires TF/Keras new
  enough to read that format (TF 2.16+ bundles Keras 3) — an older TF/Keras (e.g. 2.11, which
  predates the format) will misdetect the file as legacy HDF5 and fail in `h5py` with
  "file signature not found", not a real corruption.
- `config.py` imports `tomllib` (stdlib, Python 3.11+) and falls back to the `tomli` backport on
  older Python — make sure whichever interpreter actually runs the script has one or the other
  installed, since the fallback import itself can silently fail on an environment that has
  neither.
- This is run on NCI Gadi; watch for interpreter/environment mismatches between what a script was
  last run under and what's currently active (module-loaded Python vs. `~/.local` site-packages
  for a different Python version). The env with both TF-GPU and ultranest is
  `/g/data/cm25/jw5893/dragons/tf-env`.
- Never call `emulator.predict()` in the likelihood. It's a training-loop API that rebuilds a
  `tf.data` pipeline, callback list and result aggregator per invocation, costing a flat **~95 ms
  regardless of architecture or batch size** (measured 93.9 / 94.3 / 95.3 / 94.6 ms across 1 to 12
  layers on an A100). UltraNest calls the likelihood tens of thousands of times with a mean batch
  of 17-47 points, so that fixed cost was ~94-101% of the first four runs' wall time, at 0-11% GPU
  utilisation. `run_vectorized.py` uses a direct `emulator(x, training=False)` call instead.
  - Measured alternatives at UltraNest's batch sizes (`benchmark_predict_vs_call.py`, results in
    `benchmark_predict_vs_call_gpu.json`): eager direct call is 23.6x (12 layers) to 51.8x
    (1 layer) faster than `predict()`; a `tf.function` with a pinned `input_signature` is a
    further 4-5x on top (deepest arch: 94.6 ms -> 19.8 ms eager -> 4.0 ms traced).
  - The eager win *shrinks* as models grow (51.8x -> 23.6x) because eager re-dispatches each layer
    from Python, so cost scales with depth (4.4 ms at 1 layer, 19.8 ms at 12). `tf.function` is
    nearly flat (1.9 -> 4.0 ms). We're on eager for now; the traced version needs a cache of
    concrete functions keyed on batch size, since an unpinned `tf.function` retraces on every new
    shape and UltraNest varies the shape constantly — that would be slower than what it replaces.
  - Switching from `predict()` changes outputs by ~1e-6 *relative* (float32 kernel-path rounding),
    with zero change to the sub-threshold reject flags, verified over the real `PARAMETER_BOUNDS`
    volume on small/middle/deepest archs. Compare over the actual prior box, not `uniform(-1,1)`:
    outside its training domain the emulator extrapolates to values ~1e4, where a 1e-6 relative
    difference looks like a huge absolute one.
- A sub-threshold SMF prediction means the point is rejected, but the likelihood returns a finite
  `LOG_ZERO` (-1e100) rather than `-inf`. UltraNest asserts finiteness on a couple of random
  start-up draws, so a true `-inf` makes whether a run starts at all a coin flip — the smaller
  architectures lose it often enough to matter. Rejection is equally hard either way.
