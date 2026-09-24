#!/usr/bin/env python3
"""Training-set-size sweep driver for run_vectorized.py on gadi.

The companion to sweep.py: same sampler, same emulator flavours, same GPU
budget, but what varies is the fraction of training data the emulator saw
rather than its architecture. One nested-sampling run per (flavour, fraction)
over every fraction under <models_dir>/{uniform,posterior}/reg_dataset_size/,
14 in total.

    ./dataset_size_sweep.py status     what is done / running / left to do
    ./dataset_size_sweep.py submit     qsub jobs to fill the free GPU slots
    ./dataset_size_sweep.py submit -n  report without submitting anything

    ./dataset_size_sweep.py submit --max-total 8 --max-per-flavour 4

submit is idempotent: run it as often as you like and it only tops the queue
back up to the cap.

This file is deliberately standalone -- it imports nothing from sweep.py, so
that file can be read and trusted on its own and nothing here can perturb it.
The cost is real duplication: the PBS resource request, the queue parsing, the
per-run bookkeeping and the scheduling logic all appear in both. If you change
the module stack, the walltime, the queue or the resource shape, change it in
both files.

The two drivers do share the GPU budget, because the cap counts every balerion
sweep job in the queue rather than only this family's -- so running both won't
quietly use 8 GPUs. That only holds in this direction: sweep.py counts only
its own jobs, so if you submit architecture runs while dataset runs are
queued, check the queue yourself first.
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
# Kept identical to sweep.py so both families request the same job.
PROJECT = "dg97"
QUEUE = "gpursaa"
# The architecture runs finish in 3-11 min against this; 12h is ample and
# queues far faster than a 48h request on a partition this contended.
# --resume means an overrun costs a requeue, not the run.
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
# GPU use: once a flavour runs out of work, the other spills into the spare
# slots rather than leaving them idle.
MAX_JOBS_PER_FLAVOUR = 2   # 2 uniform + 2 posterior, so both trends fill in together
MAX_TOTAL_JOBS = 4         # the 4 GPUs we're using
MAX_ATTEMPTS = 4           # give up resubmitting a run that keeps dying

SWEEP_DIR = Path(__file__).resolve().parent
REPO_DIR = SWEEP_DIR.parent
CONFIG = REPO_DIR / "config.toml"

# Per-run bookkeeping (job script, PBS log, attempt count, job id) lives in a
# .sweep/ subdirectory of that run's own output dir, not in the repo: a run's
# script and log belong with its results. The name is prefixed so run_status
# can tell our bookkeeping from actual sampler state -- see the "started"
# check there, which is what stops a submitted-but-never-run directory from
# being mistaken for a half-finished run.
SWEEP_SUBDIR = ".sweep"

# Only reg_dataset_size. posterior also has a
# reg_dataset_size_TrainingDataMoreThanUniformTrainingData tree from an
# earlier training pass; naming the directory exactly leaves it out.
MODELS_SUBDIR = "reg_dataset_size"

# Output sits under a family level. The architecture sweep's 24 runs already
# exist at <flavour>/<arch>/<repeat>, so it stayed flat to avoid a migration;
# this family starts from nothing and gets the tidier namespaced path.
# "dataset_size" is not a valid architecture name and sweep.py enumerates
# candidates from models_dir rather than from here, so they can't collide.
OUTPUT_SUBDIR = "dataset_size"

FLAVOURS = {"u": "uniform", "p": "posterior"}

# States meaning "this job is still alive, don't touch it": Running, Queued,
# Held, Begun, Suspended, Transiting, User-suspended, Moved. Deliberately not
# F/X (finished/exited) -- those fall through to the filesystem check, where
# result.json decides DONE vs INCOMPLETE.
LIVE_STATES = set("RQHBSTUM")

# Every balerion sweep job, either family: bal_u_xxxx / bal_p_xxxx from
# sweep.py, bal_us_xxxx / bal_ps_xxxx from here. Used for the shared cap, and
# why the job names below carry a family letter.
ANY_SWEEP_JOB = re.compile(r"^bal_([up])s?_[0-9a-f]{4}$")


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

def model_path(flavour, fraction):
    """The .keras for one training fraction.

    No <repeat> level: balerion's dataset-size test writes the model straight
    into the fraction's directory, where the architecture test nests it under
    a repeat index.
    """
    return (MODELS_DIR / flavour / MODELS_SUBDIR / fraction
            / f"model_reg_dataset_size_{fraction}.keras")


def output_dir(flavour, fraction):
    return INFERENCE_DIR / flavour / OUTPUT_SUBDIR / fraction


def job_name(flag, fraction):
    """PBS job name, e.g. bal_us_1c40 -- the 's' marks this family.

    Keyed on the fraction rather than its position in the sorted list, so
    training another fraction doesn't renumber jobs already queued. Cannot
    collide with sweep.py's bal_<flag>_<hash>, which carries no family letter.
    Short enough that qstat -f never wraps it onto a continuation line, which
    would break the parse in queued_jobs().
    """
    digest = hashlib.md5(fraction.encode()).hexdigest()[:4]
    return f"bal_{flag}s_{digest}"


def flag_of(flavour):
    return "u" if flavour == "uniform" else "p"


def list_fractions(flavour):
    """Fractions with a trained model, smallest first."""
    root = MODELS_DIR / flavour / MODELS_SUBDIR
    if not root.is_dir():
        raise SweepError(f"no dataset-size dir: {root}")
    fractions = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            float(child.name)
        except ValueError:
            continue  # tag.txt and friends
        if model_path(flavour, child.name).exists():
            fractions.append(child.name)
    # Numeric sort, so 0.005 precedes 0.01 and the cheap end runs first.
    return sorted(fractions, key=float)


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


def job_is_live(jobid):
    """Ask PBS about one job by id.

    Belt-and-braces against the `qstat -u` listing missing a job we submitted:
    that gap would read as INCOMPLETE and resubmit a job already in the queue.
    A non-zero exit means PBS doesn't know the id (finished and purged), which
    is a real answer; anything else we treat as still live rather than risk a
    duplicate.
    """
    proc = subprocess.run(["qstat", "-f", jobid], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("job_state = "):
            return stripped[len("job_state = "):]
    return None


# --- status ----------------------------------------------------------------

DONE, RUNNING, QUEUED, INCOMPLETE, STUCK, TODO = (
    "DONE", "RUNNING", "QUEUED", "INCOMPLETE", "STUCK", "TODO")
ELIGIBLE = (TODO, INCOMPLETE)
LIVE = (RUNNING, QUEUED)


def sweep_dir(out):
    return out / SWEEP_SUBDIR


def attempts_of(out):
    try:
        return int((sweep_dir(out) / "attempts").read_text().strip())
    except (OSError, ValueError):
        return 0


def jobid_of(out):
    """The job id try_submit last recorded for this run, if any."""
    try:
        return (sweep_dir(out) / "jobid").read_text().strip() or None
    except OSError:
        return None


def run_status(flavour, fraction, live_jobs):
    """DONE | RUNNING | QUEUED | INCOMPLETE | STUCK | TODO."""
    out = output_dir(flavour, fraction)
    # run_vectorized.py writes result.json only once sampler.run() returns,
    # so its presence is the completion marker.
    if (out / "result.json").exists():
        return DONE

    state = live_jobs.get(job_name(flag_of(flavour), fraction))
    if state is None:
        # The name wasn't in the listing. Before concluding nothing is queued
        # for this run -- which would make it eligible and risk a duplicate --
        # check the specific job id recorded at submit time.
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
        # (just our own .sweep/ bookkeeping) means "submitted, never ran" --
        # counting that as INCOMPLETE is what caused duplicate submissions.
        started = any(not child.name.startswith(SWEEP_SUBDIR)
                      for child in out.iterdir())
        if not started:
            return TODO
        # UltraNest resumes from its own state, so a run killed at the
        # walltime only needs resubmitting -- capped so one that dies on
        # startup every time doesn't cycle forever.
        return STUCK if attempts_of(out) >= MAX_ATTEMPTS else INCOMPLETE
    return TODO


def survey(live_jobs):
    """[(flag, flavour, fraction, jobname, status)] for every run."""
    rows = []
    for flag, flavour in FLAVOURS.items():
        for fraction in list_fractions(flavour):
            rows.append((flag, flavour, fraction, job_name(flag, fraction),
                         run_status(flavour, fraction, live_jobs)))
    return rows


def live_counts(live_jobs):
    """Live jobs per flavour, counting BOTH sweep families.

    The GPU budget is shared with the architecture sweep, so a dataset run
    must not be submitted into a slot an architecture run already holds.
    """
    counts = {flag: 0 for flag in FLAVOURS}
    for name in live_jobs:
        match = ANY_SWEEP_JOB.match(name)
        if match:
            counts[match.group(1)] += 1
    return counts


def column_width(rows):
    return max([len(f) for _, _, f, _, _ in rows] + [len("FRACTION")])


def cmd_status(_args):
    rows = survey(queued_jobs())
    width = column_width(rows)
    print(f"{'FLAVOUR':<10} {'JOB':<12} {'FRACTION':<{width}} {'STATUS':<11} OUTPUT")
    for _, flavour, fraction, name, status in rows:
        print(f"{flavour:<10} {name:<12} {fraction:<{width}} {status:<11} "
              f"{output_dir(flavour, fraction)}")

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

# Generated by dataset_size_sweep.py into this run's own output dir --
# regenerated on every submit, so edit that script rather than this file.
#   flavour:  {flavour}
#   fraction: {fraction} of the training set
#   attempt:  {attempt}

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


def write_job_script(flag, flavour, fraction, jobname):
    """Write this run's job script into its own output dir, return its path."""
    out = output_dir(flavour, fraction)
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
        flavour=flavour, fraction=fraction, attempt=attempt, flag=flag,
        model=model_path(flavour, fraction), out=out,
    ), encoding="utf-8")
    job.chmod(0o755)
    return job


def try_submit(flag, flavour, fraction, jobname, status, width, dry_run):
    """True if the slot was consumed."""
    if dry_run:
        # Report only. Job scripts live in the run's output dir, so writing
        # one would create that directory and leave the inference tree
        # littered with dirs for runs this didn't submit.
        print(f"  [dry-run] {jobname:<12} {flavour:<10} {fraction:<{width}} "
              f"({status})  -> {sweep_dir(output_dir(flavour, fraction))}/job.sh")
        return True

    job = write_job_script(flag, flavour, fraction, jobname)
    proc = subprocess.run(["qsub", str(job)], capture_output=True, text=True)
    jobid = proc.stdout.strip()
    if proc.returncode != 0 or not jobid:
        print(f"  FAILED to qsub {jobname} ({flavour} {fraction}): "
              f"{proc.stderr.strip()}", file=sys.stderr)
        return False

    out = output_dir(flavour, fraction)
    book = sweep_dir(out)
    book.mkdir(parents=True, exist_ok=True)
    (book / "attempts").write_text(f"{attempts_of(out) + 1}\n")
    # Recorded so run_status can tell "a job exists for this run" from "a run
    # started and died" when the queue listing comes back short.
    (book / "jobid").write_text(f"{jobid}\n")
    print(f"  submitted {jobname:<12} {flavour:<10} {fraction:<{width}} "
          f"({status}) {jobid}")
    return True


def cmd_submit(args):
    live_jobs = queued_jobs()
    rows = survey(live_jobs)
    width = column_width(rows)

    # Eligible work per flavour, cheapest fraction first, consumed by both
    # passes below.
    pending = {
        flag: [row for row in rows if row[0] == flag and row[4] in ELIGIBLE]
        for flag in FLAVOURS
    }
    live = live_counts(live_jobs)
    total_live = sum(live.values())

    for flag, flavour in FLAVOURS.items():
        print(f"{flavour + ':':<11}{live[flag]} in queue (both sweeps), "
              f"{len(pending[flag])} eligible")
    print(f"{'caps:':<11}{args.max_per_flavour} per flavour, "
          f"{args.max_total} total")

    def take(flag, limit):
        nonlocal total_live
        while (pending[flag] and live[flag] < limit
               and total_live < args.max_total):
            f, flavour, fraction, jobname, status = pending[flag].pop(0)
            if try_submit(f, flavour, fraction, jobname, status, width,
                          args.dry_run):
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
    )
    sub = parser.add_subparsers(dest="cmd")

    status = sub.add_parser("status", help="what is done / running / left to do")
    status.set_defaults(func=cmd_status)

    submit = sub.add_parser("submit", help="fill the free GPU slots")
    submit.add_argument("-n", "--dry-run", action="store_true",
                        help="write nothing, just report what would be sent")
    submit.add_argument("--max-total", type=int, default=MAX_TOTAL_JOBS,
                        help=f"GPUs to use at once (default {MAX_TOTAL_JOBS})")
    submit.add_argument("--max-per-flavour", type=int,
                        default=MAX_JOBS_PER_FLAVOUR,
                        help=f"cap per flavour (default {MAX_JOBS_PER_FLAVOUR})")
    submit.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    if args.cmd is None:
        args = parser.parse_args(["status"])

    try:
        args.func(args)
    except SweepError as exc:
        print(f"dataset_size_sweep: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
