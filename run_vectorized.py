import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from ultranest import ReactiveNestedSampler

from config import load_config, get_models_dir, get_inference_dir

gpus = tf.config.list_physical_devices('GPU')
use_gpu = len(gpus) > 0
device = '/GPU:0' if use_gpu else '/CPU:0'
print(f"GPUs visible to TensorFlow: {gpus}")
print(f"Running emulator on: {device}")

REDSHIFTS = (6, 7, 8, 9, 10)
DATA_DIR = Path(__file__).resolve().parent / "data"

REDSHIFT_INDICES = {
    6: [[8.051645433238704,
         8.451645433238705,
         8.851645433238705,
         9.251645433238705,
         9.651645433238706,
         10.051645433238704,
         10.451645433238705],
        [0, 1, 2, 3, 4, 5, 6]],
    7: [[8.101645433238705,
         8.551645433238704,
         8.951645433238705,
         9.351645433238705,
         9.751645433238705,
         10.151645433238706],
        [7, 8, 9, 10, 11, 12]],
    8: [[8.251645433238705,
         8.751645433238705,
         9.201645433238705,
         9.601645433238705,
         10.001645433238705],
        [13, 14, 15, 16, 17]],
    9: [[8.101645433238705, 8.601645433238705, 9.351645433238705], [18, 19, 20]],
    10: [[8.101645433238705, 8.601645433238705], [21, 22]]
}
NUM_OUTPUTS = 23

# Stand-in for a zero likelihood. UltraNest asserts that the likelihood is
# finite on its start-up test draws, so a hard -inf rejection makes whether a
# run even starts depend on which two points it happens to draw — which is a
# coin flip that some of the smaller architectures lose. A number this negative
# is rejected in favour of any real point just as surely as -inf, without
# tripping that check or putting inf/nan through the residual arithmetic.
LOG_ZERO = -1e100

# Architecture used when --model is not given, i.e. the deepest one in the
# sweep. The sweep scripts always pass --model explicitly.
DEFAULT_ARCH = "32_64_128_256_512_512_512_512_256_128_64_32"
DEFAULT_REPEAT = "0"

# Keyed on flag_U. These are properties of the *training set*, not of the
# network architecture, so they stay fixed as we sweep over architectures:
# threshold_smf is the floor below which the emulator's SMF output is treated
# as "no galaxies" (-inf) rather than a real number.
EMULATOR_CONFIG = {
    True: {
        "flavour": "uniform",
        "regressor_inf_val": -6.3,
        "threshold_smf": -5.797041,
    },
    False: {
        "flavour": "posterior",
        "regressor_inf_val": -7.1,
        "threshold_smf": -6.6721025,
    },
}


def default_model_path(flavour, arch=DEFAULT_ARCH, repeat=DEFAULT_REPEAT):
    return (Path(get_models_dir(load_config())) / flavour / "reg_arc" / "depth"
            / arch / repeat / f"model_reg_depth_{arch}_{repeat}.keras")


def parse_flag_u(value):
    lowered = value.lower()
    if lowered in {"u", "uniform", "true", "1"}:
        return True
    if lowered in {"p", "posterior", "false", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Unrecognized emulator flag: {value}")


def derive_run_id(model_path):
    """(arch, repeat) for a model at .../depth/<arch>/<repeat>/model_*.keras.

    <repeat> is balerion's StatisticalTest run index, not an RNG seed: it
    retrains each architecture num_tests times and numbers them from 0.

    Falls back to the filename stem when the path doesn't have that shape, so
    an ad-hoc model outside the sweep tree still gets a usable output dir
    instead of silently colliding with another run's.
    """
    repeat = model_path.parent.name
    arch = model_path.parent.parent.name
    if arch and repeat and repeat.isdigit() and all(part.isdigit() for part in arch.split("_")):
        return arch, repeat
    return model_path.stem, "0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested sampling over a trained SMF emulator.",
    )
    parser.add_argument(
        "flag", type=parse_flag_u,
        help="which emulator flavour: u/uniform or p/posterior. Selects the "
             "SMF threshold, and the model when --model is omitted.",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help=f"path to the .keras emulator. Default: the {DEFAULT_ARCH} "
             "architecture for this flavour.",
    )
    parser.add_argument(
        "--max-ncalls", type=int, default=None,
        help="cap on total likelihood calls, for repeatable profiling runs. "
             "Omitted, the sampler runs to convergence.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="where to write the UltraNest log dir. Default: "
             "<inference_dir>/<flavour>/<arch>/<repeat>.",
    )
    parser.add_argument(
        "--resume", default="resume",
        choices=("resume", "resume-similar", "overwrite", "subfolder"),
        help="UltraNest resume mode. Default 'resume' so a job that hits the "
             "walltime continues where it left off when resubmitted.",
    )
    return parser.parse_args()


args = parse_args()
flag_U = args.flag
FLAVOUR = EMULATOR_CONFIG[flag_U]["flavour"]
MAX_NCALLS = args.max_ncalls

MODEL_PATH = (args.model if args.model is not None
              else default_model_path(FLAVOUR)).resolve()
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No emulator at {MODEL_PATH}")

ARCH, REPEAT = derive_run_id(MODEL_PATH)
OUTPUT_DIR = (args.output_dir if args.output_dir is not None
              else Path(get_inference_dir(load_config())) / FLAVOUR / ARCH / REPEAT)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"flavour:   {FLAVOUR}")
print(f"model:     {MODEL_PATH}")
print(f"output:    {OUTPUT_DIR}")
print(f"max_ncalls:{MAX_NCALLS if MAX_NCALLS is not None else ' run to convergence'}")

PARAMETER_NAMES = [
    r"$\alpha_{\rm SF}$",
    r"$\Sigma_{\rm SF,crit}$",
    r"$\epsilon_{\rm ej}$",
    r"$\gamma_{\rm ej}$",
    r"$\delta_{\rm ej}$",
    r"$\epsilon_{\rm rh}$",
    r"$\epsilon_{\rm rh,lim}$",
    r"$\gamma_{\rm rh}$",
    r"$\delta_{\rm rh}$",
    r"$\gamma_{\rm reinc}$",
]

PARAMETER_BOUNDS = [(-4, 0),
                    (-3, 1),
                    (-5, 2),
                    (-5, 5),
                    (-5, 5),
                    (-3, 4),
                    (-3, 2),
                    (-5, 5),
                    (-5, 5),
                    (-3, 0)]


def load_data_by_redshift():
    data_by_redshift = {}
    for z in REDSHIFTS:
        data_path = DATA_DIR / f"stefanon2021_z{z}.json"
        with data_path.open(encoding="utf-8") as handle:
            data = json.load(handle)

        rows = np.asarray(data["data"], dtype=float)
        logm = rows[::-1, 0]
        mask_data = logm > 8.0

        data_by_redshift[z] = {
            "columns": data["columns"],
            "logm": logm[mask_data],
            "phi_obs": rows[::-1, 1][mask_data],
            "dphi_up": rows[::-1, 2][mask_data],
            "dphi_lo": rows[::-1, 3][mask_data],
        }
    return data_by_redshift


def load_emulator(flag_U, model_path):
    threshold_smf = EMULATOR_CONFIG[flag_U]["threshold_smf"]
    with tf.device(device):
        emulator = tf.keras.models.load_model(str(model_path), compile=False)
    return emulator, threshold_smf


def build_inputs(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    elif theta.ndim != 2:
        raise ValueError("theta must be 1D or 2D")
    return theta


# --- the forward pass ------------------------------------------------------
# Not emulator.predict(): that's a training-loop API which rebuilds a tf.data
# pipeline, callback list and result aggregator per invocation, costing a flat
# ~95ms whatever the architecture or batch size. UltraNest calls the likelihood
# tens of thousands of times with a median batch of ~3-50 points, so that fixed
# cost was ~94-101% of the first four runs' wall time at 0-11% GPU use.
#
# A plain eager call fixes most of that, but re-dispatches every layer from
# Python, so its cost scales with depth: 4.4ms at 1 layer, 19.8ms at 12. A
# tf.function traces the whole net into one graph and is nearly flat instead
# (1.9ms -> 4.0ms), which matters because every run left in the sweep is
# deeper than the ones already done.
#
# The catch is that a graph is tied to one input shape, and UltraNest varies
# the batch size constantly -- an unpinned tf.function would retrace on nearly
# every call and be slower than what it replaces. So keep one traced function
# per batch size. Measured on a real 60k-call run: 91 distinct sizes, 72 of
# them <= 100, and tracing pays for itself after ~8 calls at that size
# (128ms to trace on the deepest net, ~16ms saved per call).
_TRACED = {}

# Bounds the cache if some future posterior explores far more distinct sizes
# than we've seen. At ~1.9 MiB/graph on the deepest net this caps the cache
# near 380 MiB, against a 128GB request that currently peaks at 2.4GB. Past
# the cap we fall back to the eager call, so the worst case is exactly today's
# performance rather than an unbounded memory climb.
MAX_TRACED_SHAPES = 200


def emulator_forward(emulator, x):
    """Emulator outputs for x, shape (n_points, 10) -> (n_points, NUM_OUTPUTS)."""
    x_tf = tf.constant(x, dtype=tf.float32)
    n_points = x.shape[0]

    traced = _TRACED.get(n_points)
    if traced is None:
        if len(_TRACED) >= MAX_TRACED_SHAPES:
            return emulator(x_tf, training=False).numpy()
        signature = [tf.TensorSpec(shape=(n_points, 10), dtype=tf.float32)]
        traced = tf.function(
            lambda t: emulator(t, training=False), input_signature=signature)
        _TRACED[n_points] = traced
    return traced(x_tf).numpy()


# --- profiling instrumentation -------------------------------------------
# Tracks how many times the likelihood is called, the batch size each time
# (to see whether UltraNest's vectorized mode is actually handing us large
# batches), and where wall-clock time goes: emulator forward pass vs. the
# rest of the residual bookkeeping. Printed as a summary at the end of the
# run so we can tell whether the sampler or the emulator call dominates.
PROFILE = {
    "n_calls": 0,
    "batch_sizes": [],
    "t_predict": 0.0,
    "t_residuals": 0.0,
    "t_total": 0.0,
}


def compute_normalized_residuals(theta, emulator, threshold_smf, data_by_redshift):
    x = build_inputs(theta)

    t0 = time.perf_counter()
    with tf.device(device):
        # See emulator_forward: traced graph per batch size, not .predict().
        # Outputs agree with predict() to ~1e-6 relative -- float32
        # kernel-path rounding -- with no change to the sub-threshold reject
        # flags, checked across the real PARAMETER_BOUNDS volume.
        y_pred = emulator_forward(emulator, x)
    t1 = time.perf_counter()
    PROFILE["t_predict"] += t1 - t0

    # A prediction at or below the SMF floor means "no galaxies here", which
    # the data contradicts, so the point is rejected outright. Flag it rather
    # than propagating -inf through the residual arithmetic: log_likelihood
    # substitutes a finite LOG_ZERO for these, and the inf/nan that -inf would
    # produce here would poison the arithmetic and warn on every call.
    invalid = np.any(y_pred <= threshold_smf, axis=1)

    residuals = np.zeros((x.shape[0], NUM_OUTPUTS), dtype=float)
    for z in REDSHIFTS:
        _, bin_idx = REDSHIFT_INDICES[z]
        phi_pred = y_pred[:, bin_idx]

        data = data_by_redshift[z]
        logm = data["logm"]
        phi_obs = data["phi_obs"]
        dphi_up = data["dphi_up"]
        dphi_lo = data["dphi_lo"]

        if phi_pred.shape[1] != len(logm):
            raise ValueError("Incompatible array lengths")

        residual = phi_pred - phi_obs
        normalization = np.where(residual > 0.0, dphi_up, dphi_lo)
        normalized_residual = residual / normalization
        residuals[:, bin_idx] = normalized_residual

    t2 = time.perf_counter()
    PROFILE["t_residuals"] += t2 - t1

    return residuals, invalid


def log_likelihood(theta):
    t_start = time.perf_counter()

    normalized_residual, invalid = compute_normalized_residuals(
        theta,
        emulator,
        threshold_smf,
        data_by_redshift,
    )
    result = -0.5 * np.sum(normalized_residual**2, axis=1)
    result = np.where(invalid, LOG_ZERO, result)

    PROFILE["n_calls"] += 1
    PROFILE["batch_sizes"].append(np.asarray(theta).shape[0] if np.asarray(theta).ndim == 2 else 1)
    PROFILE["t_total"] += time.perf_counter() - t_start

    return result


def print_profile_summary():
    n_calls = PROFILE["n_calls"]
    if n_calls == 0:
        print("\n[profile] log_likelihood was never called.")
        return

    batch_sizes = np.array(PROFILE["batch_sizes"])
    total_points = batch_sizes.sum()
    t_predict = PROFILE["t_predict"]
    t_residuals = PROFILE["t_residuals"]
    t_total = PROFILE["t_total"]
    t_other = max(t_total - t_predict - t_residuals, 0.0)

    print("\n" + "=" * 60)
    print("[profile] log_likelihood call summary")
    print("=" * 60)
    print(f"  total calls:                {n_calls}")
    print(f"  total points evaluated:     {total_points}")
    print(f"  batch size  min/median/max: "
          f"{batch_sizes.min()}/{int(np.median(batch_sizes))}/{batch_sizes.max()}")
    print(f"  batch size  mean:           {batch_sizes.mean():.1f}")
    # If traced shapes ever hits the cap, calls at new sizes are silently
    # falling back to the eager path -- worth seeing rather than guessing.
    distinct = len(set(batch_sizes.tolist()))
    capped = " (AT CAP, extra sizes ran eager)" if len(_TRACED) >= MAX_TRACED_SHAPES else ""
    print(f"  distinct batch sizes:       {distinct}")
    print(f"  traced graphs cached:       {len(_TRACED)}/{MAX_TRACED_SHAPES}{capped}")
    print("-" * 60)
    print(f"  time in emulator.predict(): {t_predict:8.2f}s  ({100 * t_predict / t_total:5.1f}%)")
    print(f"  time in residual bookkeep:  {t_residuals:8.2f}s  ({100 * t_residuals / t_total:5.1f}%)")
    print(f"  time unaccounted/other:     {t_other:8.2f}s  ({100 * t_other / t_total:5.1f}%)")
    print(f"  total time in likelihood:   {t_total:8.2f}s")
    print("-" * 60)
    print(f"  mean time per call:         {1000 * t_total / n_calls:8.3f} ms")
    print(f"  mean time per point:        {1000 * t_total / total_points:8.4f} ms")
    print("=" * 60)

    # Small histogram of batch sizes so we can see the distribution UltraNest
    # actually produces, not just min/median/max.
    edges = np.array([1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, np.inf])
    counts, _ = np.histogram(batch_sizes, bins=edges)
    print("  batch-size histogram:")
    for lo, hi, count in zip(edges[:-1], edges[1:], counts):
        if count == 0:
            continue
        hi_label = "inf" if np.isinf(hi) else str(int(hi))
        print(f"    [{int(lo):>6}, {hi_label:>6}): {count}")
    print("=" * 60)


def prior_transform(cube):
    lower = np.array([bound[0] for bound in PARAMETER_BOUNDS], dtype=float)
    upper = np.array([bound[1] for bound in PARAMETER_BOUNDS], dtype=float)
    return lower + cube * (upper - lower)


output_dir = OUTPUT_DIR

data_by_redshift = load_data_by_redshift()
emulator, threshold_smf = load_emulator(flag_U, MODEL_PATH)

sampler = ReactiveNestedSampler(
    PARAMETER_NAMES,
    log_likelihood,
    prior_transform,
    vectorized=True,
    resume=args.resume,
    log_dir=str(output_dir),
)

try:
    run_kwargs = {} if MAX_NCALLS is None else {"max_ncalls": MAX_NCALLS}
    result = sampler.run(**run_kwargs)
finally:
    # Print even on Ctrl-C / an exception so a long run interrupted
    # midway still tells us the call-count/batch-size/timing breakdown
    # so far.
    print_profile_summary()

# result.json is written only once sampler.run() has returned, so the sweep
# scripts use its presence as the "this model is done" marker. Anything else
# in output_dir is UltraNest's own resumable state.
result_path = output_dir / "result.json"
with result_path.open("w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else None)

with (output_dir / "run_info.json").open("w", encoding="utf-8") as handle:
    json.dump({
        "flavour": FLAVOUR,
        "arch": ARCH,
        "repeat": REPEAT,
        "model_path": str(MODEL_PATH),
        "threshold_smf": threshold_smf,
        "max_ncalls": MAX_NCALLS,
        "n_likelihood_calls": PROFILE["n_calls"],
        "n_points_evaluated": int(np.sum(PROFILE["batch_sizes"])),
        "device": device,
    }, handle, indent=2)
