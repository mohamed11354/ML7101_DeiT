from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable


DEFAULT_BATCH_SIZES = [32, 64, 128, 256]


@dataclass
class RunJob:
    batch_size: int
    gpu: str
    output_dir: Path
    log_path: Path
    command: list[str]
    process: subprocess.Popen | None = None
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    skipped: bool = False


@dataclass
class BatchSummary:
    batch_size: int
    gpu: str
    output_dir: str
    status: str
    best_val_top1: float
    best_epoch: int
    best_val_loss: float
    best_loss_epoch: int
    final_val_top1: float
    final_val_loss: float
    mean_train_throughput_img_s: float
    median_train_throughput_img_s: float
    total_train_time_s: float
    total_wall_time_s: float | None
    quality_target_top1: float | None = None
    epoch_to_quality: int | None = None
    time_to_quality_train_s: float | None = None
    goodput_to_quality_img_s: float | None = None
    selected: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch baseline DeiT training at several global batch sizes, one run per GPU, "
            "then select a critical batch size from validation quality and time-to-quality."
        )
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        "--gpus",
        type=str,
        nargs="+",
        default=None,
        help="GPU ids/UUIDs to assign to runs. Defaults to the first N visible CUDA devices.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--output-dir", type=str, default="./results/critical_batch_sweep")
    parser.add_argument("--sequential-script", type=str, default="deit_sequential.py")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--quality-ratio",
        type=float,
        default=0.95,
        help="Time-to-quality target as a fraction of the best validation top-1 found.",
    )
    parser.add_argument(
        "--close-top1-pp",
        type=float,
        default=0.5,
        help="Candidate runs must be within this many top-1 percentage points of the best run.",
    )
    parser.add_argument(
        "--time-tie-pct",
        type=float,
        default=5.0,
        help="Prefer the largest batch within this percent of the fastest time-to-quality.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse run directories that already contain metrics.csv instead of launching them.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to deit_sequential.py. Put them after '--', for example: "
            "-- --dataset cifar100 --data-root ./datasets/cifar100 --no-pretrained --amp"
        ),
    )
    return parser


def visible_gpu_ids() -> list[str]:
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def clean_training_args(training_args: list[str]) -> list[str]:
    if training_args and training_args[0] == "--":
        return training_args[1:]
    return training_args


def normalize_gpus(gpus: list[str] | None) -> list[str] | None:
    if gpus is None:
        return None
    normalized: list[str] = []
    for item in gpus:
        normalized.extend(part.strip() for part in item.split(",") if part.strip())
    return normalized


def command_for_run(
    *,
    python_executable: str,
    sequential_script: Path,
    training_args: list[str],
    batch_size: int,
    epochs: int,
    output_dir: Path,
) -> list[str]:
    # Put sweep-controlled args last so every launched run has the intended budget,
    # batch size, and isolated output directory even if forwarded args include them.
    return [
        python_executable,
        str(sequential_script),
        *training_args,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def prepare_jobs(args: argparse.Namespace) -> list[RunJob]:
    batch_sizes = list(args.batch_sizes)
    requested_gpus = normalize_gpus(args.gpus)
    gpus = requested_gpus or visible_gpu_ids()[: len(batch_sizes)]
    if len(gpus) < len(batch_sizes):
        raise RuntimeError(
            f"Need at least {len(batch_sizes)} GPUs for {len(batch_sizes)} simultaneous runs, "
            f"but got {len(gpus)}. Pass explicit ids with --gpus."
        )

    sequential_script = Path(args.sequential_script)
    if not sequential_script.exists():
        raise FileNotFoundError(f"Could not find sequential script: {sequential_script}")

    sweep_dir = Path(args.output_dir)
    training_args = clean_training_args(args.training_args)
    jobs: list[RunJob] = []
    for batch_size, gpu in zip(batch_sizes, gpus):
        run_dir = sweep_dir / f"bs_{batch_size}"
        log_path = run_dir / "train.log"
        command = command_for_run(
            python_executable=args.python,
            sequential_script=sequential_script,
            training_args=training_args,
            batch_size=batch_size,
            epochs=args.epochs,
            output_dir=run_dir,
        )
        jobs.append(
            RunJob(
                batch_size=batch_size,
                gpu=str(gpu),
                output_dir=run_dir,
                log_path=log_path,
                command=command,
                skipped=bool(args.skip_existing and (run_dir / "metrics.csv").exists()),
            )
        )

    write_json(
        sweep_dir / "sweep_config.json",
        {
            "batch_sizes": batch_sizes,
            "gpus": gpus[: len(batch_sizes)],
            "epochs": args.epochs,
            "quality_ratio": args.quality_ratio,
            "close_top1_pp": args.close_top1_pp,
            "time_tie_pct": args.time_tie_pct,
            "training_args": training_args,
            "sequential_script": str(sequential_script),
        },
    )
    return jobs


def launch_jobs(jobs: Iterable[RunJob], *, dry_run: bool) -> None:
    for job in jobs:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        command_text = shlex.join(job.command)
        (job.output_dir / "command.txt").write_text(
            f"CUDA_VISIBLE_DEVICES={job.gpu} {command_text}\n"
        )
        if job.skipped:
            print(f"[skip] batch_size={job.batch_size} gpu={job.gpu} output={job.output_dir}")
            continue
        print(f"[launch] batch_size={job.batch_size} gpu={job.gpu} log={job.log_path}")
        if dry_run:
            print(f"        CUDA_VISIBLE_DEVICES={job.gpu} {command_text}")
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = job.gpu
        env["PYTHONUNBUFFERED"] = "1"
        log_file = job.log_path.open("w")
        job.started_at = time.perf_counter()
        job.process = subprocess.Popen(
            job.command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        job.process._critical_batch_log_file = log_file  # type: ignore[attr-defined]


def wait_for_jobs(jobs: Iterable[RunJob], *, dry_run: bool) -> None:
    if dry_run:
        return
    for job in jobs:
        if job.skipped or job.process is None:
            continue
        job.returncode = job.process.wait()
        job.finished_at = time.perf_counter()
        log_file = getattr(job.process, "_critical_batch_log_file", None)
        if log_file is not None:
            log_file.close()
        status = "ok" if job.returncode == 0 else f"failed({job.returncode})"
        print(f"[done] batch_size={job.batch_size} gpu={job.gpu} status={status}")


def read_metrics(metrics_csv: Path) -> list[dict[str, float]]:
    with metrics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    if not rows:
        raise RuntimeError(f"No metrics rows found in {metrics_csv}")
    return rows


def cumulative_time_to_quality(
    rows: list[dict[str, float]],
    quality_target_top1: float,
) -> tuple[int | None, float | None, float | None]:
    elapsed_train_s = 0.0
    processed_images = 0.0
    for row in rows:
        epoch_time_s = float(row.get("epoch_time_s", 0.0))
        throughput = float(row.get("train_throughput_img_s", 0.0))
        elapsed_train_s += epoch_time_s
        processed_images += throughput * epoch_time_s
        if float(row["val_top1"]) >= quality_target_top1:
            goodput = processed_images / max(elapsed_train_s, 1e-8)
            return int(row["epoch"]), elapsed_train_s, goodput
    return None, None, None


def summarize_run(
    job: RunJob,
    *,
    status: str,
    quality_target_top1: float | None = None,
) -> BatchSummary:
    metrics_csv = job.output_dir / "metrics.csv"
    rows = read_metrics(metrics_csv)
    best_top1_row = max(rows, key=lambda row: row["val_top1"])
    best_loss_row = min(rows, key=lambda row: row["val_loss"])
    final_row = rows[-1]
    throughputs = [row["train_throughput_img_s"] for row in rows]
    total_train_time_s = sum(row["epoch_time_s"] for row in rows)
    wall_time_s = (
        None
        if job.started_at is None or job.finished_at is None
        else job.finished_at - job.started_at
    )
    epoch_to_quality = None
    time_to_quality_s = None
    goodput_to_quality = None
    if quality_target_top1 is not None:
        epoch_to_quality, time_to_quality_s, goodput_to_quality = cumulative_time_to_quality(
            rows,
            quality_target_top1,
        )

    return BatchSummary(
        batch_size=job.batch_size,
        gpu=job.gpu,
        output_dir=str(job.output_dir),
        status=status,
        best_val_top1=float(best_top1_row["val_top1"]),
        best_epoch=int(best_top1_row["epoch"]),
        best_val_loss=float(best_loss_row["val_loss"]),
        best_loss_epoch=int(best_loss_row["epoch"]),
        final_val_top1=float(final_row["val_top1"]),
        final_val_loss=float(final_row["val_loss"]),
        mean_train_throughput_img_s=float(mean(throughputs)),
        median_train_throughput_img_s=float(median(throughputs)),
        total_train_time_s=float(total_train_time_s),
        total_wall_time_s=None if wall_time_s is None else float(wall_time_s),
        quality_target_top1=quality_target_top1,
        epoch_to_quality=epoch_to_quality,
        time_to_quality_train_s=time_to_quality_s,
        goodput_to_quality_img_s=goodput_to_quality,
    )


def choose_batch(
    summaries: list[BatchSummary],
    *,
    quality_target_top1: float,
    close_top1_pp: float,
    time_tie_pct: float,
) -> BatchSummary:
    best_top1 = max(summary.best_val_top1 for summary in summaries)
    close_floor = best_top1 - close_top1_pp
    candidates = [
        summary
        for summary in summaries
        if summary.best_val_top1 >= close_floor and summary.time_to_quality_train_s is not None
    ]
    if not candidates:
        candidates = [
            summary
            for summary in summaries
            if summary.best_val_top1 >= quality_target_top1 and summary.time_to_quality_train_s is not None
        ]
    if not candidates:
        return max(summaries, key=lambda summary: summary.best_val_top1)

    fastest_time = min(summary.time_to_quality_train_s for summary in candidates if summary.time_to_quality_train_s)
    tie_limit = fastest_time * (1.0 + time_tie_pct / 100.0)
    near_fastest = [
        summary
        for summary in candidates
        if summary.time_to_quality_train_s is not None and summary.time_to_quality_train_s <= tie_limit
    ]
    return max(near_fastest, key=lambda summary: summary.batch_size)


def write_summary_files(
    sweep_dir: Path,
    summaries: list[BatchSummary],
    selected: BatchSummary,
    *,
    best_top1: float,
    quality_target_top1: float,
    args: argparse.Namespace,
) -> None:
    for summary in summaries:
        summary.selected = summary.batch_size == selected.batch_size

    payload = {
        "selected_batch_size": selected.batch_size,
        "selected_output_dir": selected.output_dir,
        "best_observed_val_top1": best_top1,
        "quality_target_top1": quality_target_top1,
        "selection_rule": (
            f"within {args.close_top1_pp} top-1 percentage points of the best run, "
            f"then fastest train time to {args.quality_ratio:.3f} * best top-1; "
            f"prefer largest batch within {args.time_tie_pct}% of fastest"
        ),
        "runs": [asdict(summary) for summary in summaries],
    }
    write_json(sweep_dir / "sweep_summary.json", payload)

    csv_path = sweep_dir / "sweep_summary.csv"
    rows = [asdict(summary) for summary in summaries]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries: list[BatchSummary], selected: BatchSummary, quality_target_top1: float) -> None:
    print()
    print(f"Quality target top-1: {quality_target_top1:.4f}")
    print("batch | best_top1 | best_epoch | epoch_to_target | train_s_to_target | median_img_s | selected")
    for summary in sorted(summaries, key=lambda item: item.batch_size):
        time_to_quality = (
            "NA"
            if summary.time_to_quality_train_s is None
            else f"{summary.time_to_quality_train_s:.1f}"
        )
        epoch_to_quality = "NA" if summary.epoch_to_quality is None else str(summary.epoch_to_quality)
        marker = "yes" if summary.selected else ""
        print(
            f"{summary.batch_size:>5} | "
            f"{summary.best_val_top1:>9.4f} | "
            f"{summary.best_epoch:>10} | "
            f"{epoch_to_quality:>15} | "
            f"{time_to_quality:>17} | "
            f"{summary.median_train_throughput_img_s:>12.2f} | "
            f"{marker}"
        )
    print()
    print(f"Selected critical batch size: {selected.batch_size}")


def main() -> None:
    args = build_parser().parse_args()
    jobs = prepare_jobs(args)
    launch_jobs(jobs, dry_run=args.dry_run)
    if args.dry_run:
        return

    wait_for_jobs(jobs, dry_run=False)
    failed = [job for job in jobs if not job.skipped and job.returncode != 0]
    if failed:
        for job in failed:
            print(f"[error] batch_size={job.batch_size} failed; inspect {job.log_path}")
        raise SystemExit(1)

    first_pass = [
        summarize_run(job, status="skipped" if job.skipped else "ok")
        for job in jobs
    ]
    best_top1 = max(summary.best_val_top1 for summary in first_pass)
    quality_target_top1 = best_top1 * args.quality_ratio
    summaries = [
        summarize_run(
            job,
            status="skipped" if job.skipped else "ok",
            quality_target_top1=quality_target_top1,
        )
        for job in jobs
    ]
    selected = choose_batch(
        summaries,
        quality_target_top1=quality_target_top1,
        close_top1_pp=args.close_top1_pp,
        time_tie_pct=args.time_tie_pct,
    )
    write_summary_files(
        Path(args.output_dir),
        summaries,
        selected,
        best_top1=best_top1,
        quality_target_top1=quality_target_top1,
        args=args,
    )
    print_summary(summaries, selected, quality_target_top1)
    print(f"Wrote summary files under {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
