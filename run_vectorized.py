import json
import time
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from ultranest import ReactiveNestedSampler

from config import load_config, get_models_dir

# Under `mpirun`, this whole script runs once per rank. Ask MPI directly for
# our own rank rather than relying on the sampler's mpi_rank attribute
# (UltraNest sets that internally behind a broad try/except around its own
# mpi4py import, so reading it back couples us to that internal fallback
# instead of asking MPI directly). Falls back to rank 0 when mpi4py isn't
# installed or the script isn't run under mpirun, so this works unchanged
# in the single-process case.
try:
    from mpi4py import MPI
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
except ImportError:
    MPI_RANK = 0

if len(sys.argv) > 1:
    flag_arg = sys.argv[1].lower()
    if flag_arg in {"u", "uniform", "true", "1"}:
        flag_U = True
    elif flag_arg in {"p", "posterior", "false", "0"}:
        flag_U = False
    else:
        raise ValueError(f"Unrecognized emulator flag: {sys.argv[1]}")

# Optional second CLI arg: cap on total likelihood calls, so a profiling run
# stops at a known, repeatable point instead of running to full convergence
# (or needing a Ctrl-C). Passed straight to sampler.run(max_ncalls=...).
MAX_NCALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

REDSHIFTS = (6, 7, 8, 9, 10)
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / f"ultranest_output_vec_{'U' if flag_U else 'P'}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

_ARCH = "32_64_128_256_512_512_512_512_256_128_64_32"
_MODELS_DIR = get_models_dir(load_config())

EMULATOR_CONFIG = {
    True: {
        "path": str(Path(_MODELS_DIR) / "uniform" / "reg_arc" / "depth" / _ARCH / "0"
                    / f"model_reg_depth_{_ARCH}_0.keras"),
        "regressor_inf_val": -6.3,
        "threshold_smf": -5.797041,
    },
    False: {
        "path": str(Path(_MODELS_DIR) / "posterior" / "reg_arc" / "depth" / _ARCH / "0"
                    / f"model_reg_depth_{_ARCH}_0.keras"),
        "regressor_inf_val": -7.1,
        "threshold_smf": -6.6721025,
    },
}

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


def load_emulator(flag_U):
    config = EMULATOR_CONFIG[flag_U]
    emulator_path = config["path"]
    threshold_smf = config["threshold_smf"]
    emulator = tf.keras.models.load_model(emulator_path, compile=False)
    return emulator, threshold_smf


def build_inputs(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    elif theta.ndim != 2:
        raise ValueError("theta must be 1D or 2D")
    return theta


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
    y_pred = emulator.predict(x, verbose=0)
    t1 = time.perf_counter()
    PROFILE["t_predict"] += t1 - t0

    mask = y_pred <= threshold_smf
    y_pred = np.where(mask, -np.inf, y_pred)

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

    return residuals


def log_likelihood(theta):
    t_start = time.perf_counter()

    normalized_residual = compute_normalized_residuals(
        theta,
        emulator,
        threshold_smf,
        data_by_redshift,
    )
    result = -0.5 * np.sum(normalized_residual**2, axis=1)

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


output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

data_by_redshift = load_data_by_redshift()
emulator, threshold_smf = load_emulator(flag_U)

sampler = ReactiveNestedSampler(
    PARAMETER_NAMES,
    log_likelihood,
    prior_transform,
    vectorized=True,
    resume='overwrite',
    log_dir=str(output_dir),
)

try:
    result = sampler.run(max_ncalls=MAX_NCALLS)
finally:
    # Under mpirun this script runs once per rank, and all ranks reach this
    # point after sampler.run() returns. Only rank 0 prints/writes so we
    # don't get N-way duplicated console output and N processes racing to
    # write the same result.json.
    if MPI_RANK == 0:
        # Print even on Ctrl-C / an exception so a long run interrupted
        # midway still tells us the call-count/batch-size/timing breakdown
        # so far.
        print_profile_summary()

if MPI_RANK == 0:
    result_path = output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else None)
