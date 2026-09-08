#!/usr/bin/env python3
"""Architecture sweep driver for run_vectorized.py on gadi.

One nested-sampling run per (flavour, architecture): every arch under
<models_dir>/{uniform,posterior}/reg_arc/depth/. Each run gets one GPU --
UltraNest asks for well under 500 points per likelihood call, so a second
GPU would have nothing to do -- and we keep at most 2 uniform + 2 posterior
jobs in the queue at once, for 4 GPUs total.

    ./sweep.py status     what is done / running / left to do
    ./sweep.py submit     generate + qsub jobs to fill the free GPU slots
    ./sweep.py submit -n  generate the job scripts but don't qsub them

To use more (or fewer) GPUs for one invocation, without editing this file:

    ./sweep.py submit --max-total 8 --max-per-flavour 4

submit is idempotent: run it as often as you like (from cron, or by hand
after a batch finishes) and it only ever tops the queue back up to the cap.
"""

import argparse
import getpass
import hashlib
import re
import subprocess
import sys
from pathlib import Path

# --- PBS resources ---------------------------------------------------------
# gpursaa runs on the four gadi-gpu-rsaa nodes, which are 56 cores / 4 GPUs /
# 512GB each -- not the 48-core gpuvolta layout. So a quarter node per GPU is
# 14 cpus and 128GB, and four of these jobs tile one node exactly.
PROJECT = "dg97"
QUEUE = "gpursaa"
# The first four runs (with the old predict() call) finished in 30-85 min
# against a 48h request. 12h is ~8x the worst of those and queues far faster
# on a partition this contended; --resume means an overrun costs a requeue,
# not the run. Revisit if a deep arch ever gets close to it.
WALLTIME = "12:00:00"
NCPUS = 14
NGPUS = 1
MEM = "128GB"
STORAGE = "gdata/cm25+scratch/cm25"
EMAIL = "James.Watt@anu.edu.au"
VENV = "/g/data/cm25/jw5893/dragons/tf-env"

# --- sweep policy ----------------------------------------------------------
# The per-flavour cap keeps uniform and posterior progressing together instead
# of one flavour monopolising the GPUs. The total cap is what actually bounds
# GPU use: once a flavour runs out of work, the other is allowed to spill into
# the spare slots rather than leaving them idle.
MAX_JOBS_PER_FLAVOUR = 2   # 2 uniform + 2 posterior, so both trends fill in together
MAX_TOTAL_JOBS = 4         # the 4 GPUs we're using
MAX_ATTEMPTS = 4           # give up resubmitting a run that keeps dying

# The <i> in .../depth/<arch>/<i>/: balerion's StatisticalTest retrains each
# architecture num_tests times and numbers the runs from 0 (no RNG seeding
# involved -- it's just repeat i). reg_arc.py used NUM_TESTS=1, so 0 is all
# there is; bump this if a later training pass produces 1, 2, ...
REPEAT = "0"

SWEEP_DIR = Path(__file__).resolve().parent
REPO_DIR = SWEEP_DIR.parent

# Per-run bookkeeping (job script, PBS log, attempt count, job id) lives in a
# .sweep/ subdirectory of that run's own output dir, not in the repo: a run's
# job script and log belong with its results, and the repo stays clean.
# The name is prefixed so run_status can tell our bookkeeping from actual
# sampler state -- see the "started" check there, which is what stops a
# submitted-but-never-run directory being mistaken for a half-finished run.
SWEEP_SUBDIR = ".sweep"
CONFIG = REPO_DIR / "config.toml"

FLAVOURS = {"u": "uniform", "p": "posterior"}

# States meaning "this job is still alive, don't touch it": Running, Queued,
# Held, Begun, Suspended, Transiting, User-suspended, Moved. Deliberately not
# F/X (finished/exited) -- those fall through to the filesystem check, where
# result.json decides DONE vs INCOMPLETE.
LIVE_STATES = set("RQHBSTUM")


class SweepError(Exception):
    """Anything that should stop the run with a message rather than a traceback."""


# --- config ----------------------------------------------------------------

def load_paths():
    """models_dir and inference_dir out of config.toml.

    Parsed with a regex rather than tomllib: gadi login nodes default to a
    python older than 3.11, and the file is flat `key = "value"`. Keeping this
    dependency-free means the sweep runs under whatever python3 is on PATH.
    """
    if not CONFIG.exists():
        raise SweepError(f"no config.toml at {CONFIG} (copy config.toml.example)")
    text = CONFIG.read_text(encoding="utf-8")
    paths = {}
    for key in ("models_dir", "inference_dir"):
        match = re.search(rf'^\s*{key}\s*=\s*"([^"]*)"', text, re.MULTILINE)
        if not match or not match.group(1):
            raise SweepError(f"config.toml has no {key}")
        paths[key] = Path(match.group(1))
    if not paths["models_dir"].is_dir():
        raise SweepError(f"models_dir does not exist: {paths['models_dir']}")
    return paths["models_dir"], paths["inference_dir"]


MODELS_DIR, INFERENCE_DIR = load_paths()


# --- the runs we could do --------------------------------------------------

def model_path(flavour, arch):
    return (MODELS_DIR / flavour / "reg_arc" / "depth" / arch / REPEAT
            / f"model_reg_depth_{arch}_{REPEAT}.keras")


def output_dir(flavour, arch):
    return INFERENCE_DIR / flavour / arch / REPEAT


def arch_size(arch):
    """(layer count, total width) -- the sort key, so cheap archs run first.

    Sorting by this reproduces depth_list in model_scripts/reg_arc.py exactly.
    """
    widths = [int(part) for part in arch.split("_")]
    return len(widths), sum(widths)


def job_name(flag, arch):
    """PBS job name, e.g. bal_u_a3f9.

    Derived from the architecture rather than its position in the sorted list,
    so training a new architecture doesn't renumber the jobs already queued.
    Short enough that qstat -f never wraps it onto a continuation line, which
    would break the parse in queued_jobs().
    """
    digest = hashlib.md5(arch.encode()).hexdigest()[:4]
    return f"bal_{flag}_{digest}"


def list_archs(flavour):
    """Architectures with a trained model, smallest first."""
    depth_dir = MODELS_DIR / flavour / "reg_arc" / "depth"
    if not depth_dir.is_dir():
        raise SweepError(f"no arch dir: {depth_dir}")
    archs = [
        d.name for d in depth_dir.iterdir()
        if d.is_dir()
        and all(part.isdigit() for part in d.name.split("_"))
        and model_path(flavour, d.name).exists()
    ]
    return sorted(archs, key=arch_size)


# --- what PBS thinks is happening ------------------------------------------

def queued_jobs():
    """{job name: state} for our jobs still in the queue.

    Parses `qstat -f` rather than the compact table because that table
    truncates the job-name column, and a truncated name can't be matched.
    """
    proc = subprocess.run(
        ["qstat", "-u", getpass.getuser(), "-f"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        # An empty parse is indistinguishable from "no jobs queued", which
        # would make submit fire duplicates of everything already running.
        # Refuse rather than guess.
        raise SweepError(
            f"qstat failed (rc={proc.returncode}); refusing to submit blind\n"
            f"{proc.stderr.strip()}"
        )

    jobs, name = {}, None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Job_Name = "):
            name = stripped[len("Job_Name = "):]
        elif stripped.startswith("job_state = ") and name:
            state = stripped[len("job_state = "):]
            if state in LIVE_STATES:
                jobs[name] = state
    return jobs


# --- status ----------------------------------------------------------------

DONE, RUNNING, QUEUED, INCOMPLETE, STUCK, TODO = (
    "DONE", "RUNNING", "QUEUED", "INCOMPLETE", "STUCK", "TODO")
ELIGIBLE = (TODO, INCOMPLETE)
LIVE = (RUNNING, QUEUED)


def sweep_dir(out):
    return out / SWEEP_SUBDIR


def _read_bookkeeping(out, name, legacy):
    """Read a bookkeeping file, falling back to its pre-.sweep/ location.

    Runs submitted before the move keep their files at the top level of the
    output dir; reading both means the fix stays correct for jobs that were
    already in flight rather than silently losing their attempt count.
    """
    for path in (sweep_dir(out) / name, out / legacy):
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        if value:
            return value
    return None


def attempts_of(out):
    value = _read_bookkeeping(out, "attempts", ".sweep_attempts")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def jobid_of(out):
    """The job id try_submit last recorded for this run, if any."""
    return _read_bookkeeping(out, "jobid", ".sweep_jobid")


def job_is_live(jobid):
    """Ask PBS about one job by id.

    Belt-and-braces against the `qstat -u` listing missing a job we submitted:
    that gap used to read as INCOMPLETE and resubmit a job already in the
    queue. A non-zero exit here means PBS doesn't know the id (finished and
    purged), which is a real answer; anything else we treat as still live
    rather than risk a duplicate.
    """
    proc = subprocess.run(["qstat", "-f", jobid], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("job_state = "):
            return stripped[len("job_state = "):]
    return None


def run_status(flavour, arch, live_jobs):
    out = output_dir(flavour, arch)
    # run_vectorized.py writes result.json only after sampler.run() returns,
    # so its presence is the completion marker.
    if (out / "result.json").exists():
        return DONE

    state = live_jobs.get(job_name(flag_of(flavour), arch))
    if state is None:
        # The name wasn't in the listing. Before concluding nothing is queued
        # for this run -- which would make it eligible and risk a duplicate --
        # check the specific job id we recorded at submit time.
        jobid = jobid_of(out)
        if jobid:
            state = job_is_live(jobid)
    if state == "R":
        return RUNNING
    if state is not None:
        return QUEUED

    if out.is_dir():
        # Only a directory with actual sampler state is a half-finished run.
        # try_submit creates this directory at submit time, so a bare one
        # (just our own bookkeeping) means "submitted, never ran" -- counting
        # that as INCOMPLETE is what caused duplicate submissions. The job
        # script and PBS log live in .sweep/ inside this directory, so match
        # on the prefix: that covers the .sweep/ subdir and the legacy
        # top-level .sweep_attempts / .sweep_jobid alike.
        started = any(not child.name.startswith(SWEEP_SUBDIR)
                      for child in out.iterdir())
        if not started:
            return TODO
        # UltraNest resumes from its own state, so a job that just ran out of
        # walltime only needs resubmitting -- but cap that so a model which
        # dies on startup every time doesn't cycle forever.
        return STUCK if attempts_of(out) >= MAX_ATTEMPTS else INCOMPLETE
    return TODO


def flag_of(flavour):
    return "u" if flavour == "uniform" else "p"


def survey(live_jobs):
    """[(flag, flavour, arch, jobname, status)] for every run in the sweep."""
    rows = []
    for flag, flavour in FLAVOURS.items():
        for arch in list_archs(flavour):
            rows.append((flag, flavour, arch, job_name(flag, arch),
                         run_status(flavour, arch, live_jobs)))
    return rows


def cmd_status(_args):
    rows = survey(queued_jobs())
    width = max(len(arch) for _, _, arch, _, _ in rows)
    header = f"{'FLAVOUR':<10} {'JOB':<12} {'ARCH':<{width}} {'STATUS':<11} OUTPUT"
    print(header)
    for _, flavour, arch, name, status in rows:
        print(f"{flavour:<10} {name:<12} {arch:<{width}} {status:<11} "
              f"{output_dir(flavour, arch)}")

    print()
    for flag, flavour in FLAVOURS.items():
        mine = [status for f, _, _, _, status in rows if f == flag]
        print(f"{flavour:<10} done={mine.count(DONE)}  "
              f"in-queue={sum(s in LIVE for s in mine)}/{MAX_JOBS_PER_FLAVOUR}  "
              f"todo={sum(s in ELIGIBLE for s in mine)}  "
              f"stuck={mine.count(STUCK)}")


# --- submit ----------------------------------------------------------------

JOB_TEMPLATE = """\
#!/bin/bash
#PBS -P {project}
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l ncpus={ncpus}
#PBS -l ngpus={ngpus}
#PBS -l mem={mem}
#PBS -l storage={storage}
#PBS -l wd
#PBS -N {jobname}
#PBS -j oe
#PBS -o {log}
#PBS -m ae
#PBS -M {email}

# Generated by sweep.py into this run's own output dir -- regenerated on every
# submit, so edit sweep.py rather than this file.
#   flavour: {flavour}
#   arch:    {arch} (repeat {repeat})
#   attempt: {attempt}

module unload hdf5
module unload python3
module load intel-python3/2020.4.912
module load hdf5/1.10.5p
module load python3/3.10.4
source {venv}/bin/activate

cd {repo_dir} || exit 1
nvidia-smi || true

# No --max-ncalls: run to convergence. Resubmitting after a walltime kill
# picks up from UltraNest's saved state (run_vectorized.py defaults to
# --resume resume).
python3 run_vectorized.py {flag} \\
    --model {model} \\
    --output-dir {out}
"""


def write_job_script(flag, flavour, arch, jobname):
    """Write this run's job script into its own output dir, return its path."""
    out = output_dir(flavour, arch)
    book = sweep_dir(out)
    # PBS needs the -o directory to exist when the job starts, so create it
    # now rather than leaving it to the job.
    book.mkdir(parents=True, exist_ok=True)

    # One log per attempt: --resume means a run can be requeued after a
    # walltime kill, and overwriting would throw away the log of the attempt
    # that got furthest.
    attempt = attempts_of(out) + 1
    job = book / "job.sh"
    job.write_text(JOB_TEMPLATE.format(
        project=PROJECT, queue=QUEUE, walltime=WALLTIME, ncpus=NCPUS,
        ngpus=NGPUS, mem=MEM, storage=STORAGE, email=EMAIL, venv=VENV,
        log=book / f"job_{attempt}.o", repo_dir=REPO_DIR, jobname=jobname,
        flavour=flavour, arch=arch, repeat=REPEAT, flag=flag, attempt=attempt,
        model=model_path(flavour, arch), out=out,
    ), encoding="utf-8")
    job.chmod(0o755)
    return job


def try_submit(flag, flavour, arch, jobname, status, width, dry_run):
    """True if the slot was consumed."""
    if dry_run:
        # Report only. Now that job scripts live in the run's output dir,
        # writing one would create that directory, so a dry run would leave
        # the inference tree littered with dirs for runs it didn't submit.
        print(f"  [dry-run] {jobname:<12} {flavour:<10} {arch:<{width}} ({status})"
              f"  -> {sweep_dir(output_dir(flavour, arch))}/job.sh")
        return True

    job = write_job_script(flag, flavour, arch, jobname)

    proc = subprocess.run(["qsub", str(job)], capture_output=True, text=True)
    jobid = proc.stdout.strip()
    if proc.returncode != 0 or not jobid:
        print(f"  FAILED to qsub {jobname} ({flavour} {arch}): "
              f"{proc.stderr.strip()}", file=sys.stderr)
        return False

    out = output_dir(flavour, arch)
    book = sweep_dir(out)
    book.mkdir(parents=True, exist_ok=True)
    (book / "attempts").write_text(f"{attempts_of(out) + 1}\n")
    # Record the job id we just created. run_status uses this to tell "a job
    # exists for this run" from "a run started and died": without it, the
    # directory this function creates is itself read as a half-finished run
    # and resubmitted on the next invocation.
    (book / "jobid").write_text(f"{jobid}\n")
    print(f"  submitted {jobname:<12} {flavour:<10} {arch:<{width}} ({status}) {jobid}")
    return True


def cmd_submit(args):
    rows = survey(queued_jobs())
    width = max(len(arch) for _, _, arch, _, _ in rows)

    # Eligible work per flavour, cheapest arch first, consumed by both passes.
    pending = {
        flag: [row for row in rows if row[0] == flag and row[4] in ELIGIBLE]
        for flag in FLAVOURS
    }
    live = {
        flag: sum(1 for row in rows if row[0] == flag and row[4] in LIVE)
        for flag in FLAVOURS
    }
    total_live = sum(live.values())

    for flag, flavour in FLAVOURS.items():
        print(f"{flavour + ':':<11}{live[flag]} in queue, {len(pending[flag])} eligible")
    print(f"{'caps:':<11}{args.max_per_flavour} per flavour, {args.max_total} total")

    def take(flag, limit):
        """Submit from flag's queue while under both `limit` and the total cap."""
        nonlocal total_live
        while (pending[flag] and live[flag] < limit
               and total_live < args.max_total):
            f, flavour, arch, jobname, status = pending[flag].pop(0)
            if try_submit(f, flavour, arch, jobname, status, width, args.dry_run):
                live[flag] += 1
                total_live += 1

    # Pass 1: bring each flavour up to its own cap, so neither starves.
    for flag in FLAVOURS:
        take(flag, args.max_per_flavour)

    # Pass 2: a flavour that has run out of work leaves its slots free, so let
    # the other one have them rather than idling GPUs. Alternate so the
    # spillover stays as even as the remaining work allows.
    while total_live < args.max_total and any(pending.values()):
        before = total_live
        for flag in FLAVOURS:
            take(flag, live[flag] + 1)
        if total_live == before:
            break  # every remaining candidate failed to qsub

    if total_live == 0:
        print("nothing eligible to submit")


# --- cli -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[7:]),
    )
    sub = parser.add_subparsers(dest="cmd")

    status = sub.add_parser("status", help="what is done / running / left to do")
    status.set_defaults(func=cmd_status)

    submit = sub.add_parser("submit", help="fill the free GPU slots")
    submit.add_argument("-n", "--dry-run", action="store_true",
                        help="write the job scripts but don't qsub them")
    submit.add_argument("--max-total", type=int, default=MAX_TOTAL_JOBS,
                        help=f"GPUs to use at once (default {MAX_TOTAL_JOBS})")
    submit.add_argument("--max-per-flavour", type=int, default=MAX_JOBS_PER_FLAVOUR,
                        help=f"cap per flavour (default {MAX_JOBS_PER_FLAVOUR})")
    submit.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    if args.cmd is None:
        args = parser.parse_args(["status"])

    try:
        args.func(args)
    except SweepError as exc:
        print(f"sweep: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
