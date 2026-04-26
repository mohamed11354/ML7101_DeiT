#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "epoch",
    "train_loss",
    "val_loss",
    "val_top1",
    "train_throughput_img_s",
    "epoch_time_s",
    "peak_mem_gb",
    "gpu_utilization",
}

PLOT_METRICS = {
    "val_loss": ("Validation loss", "Validation loss vs epoch", "val_loss_vs_epoch.png"),
    "train_loss": ("Training loss", "Training loss vs epoch", "train_loss_vs_epoch.png"),
    "val_top1": (
        "Validation top-1 accuracy",
        "Validation top-1 accuracy vs epoch",
        "val_top1_vs_epoch.png",
    ),
    "gpu_utilization_mean": (
        "Average GPU utilization (%)",
        "Average GPU utilization vs epoch",
        "gpu_utilization_vs_epoch.png",
    ),
    "peak_mem_gb": ("Peak memory (GB)", "Peak memory usage vs epoch", "peak_mem_vs_epoch.png"),
}


@dataclass
class RunData:
    batch_size: int
    path: Path
    df: pd.DataFrame


class MetricsError(Exception):
    """Raised when an input metrics file is malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze metrics_{batch_size}.csv files and compute checkpoint-based "
            "throughput, statistical efficiency, and goodput."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input CSV files named like metrics_{batch_size}.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_results"),
        help="Directory where plots and summary CSV will be written.",
    )
    parser.add_argument(
        "--objective-metric",
        choices=["train_loss", "val_loss"],
        default="train_loss",
        help=(
            "Loss metric used for statistical efficiency and goodput. "
            "Default: train_loss (closest to the optimization objective)."
        ),
    )
    parser.add_argument(
        "--se-window-epochs",
        type=int,
        default=1,
        help=(
            "Window size s, in epochs, used for SE_{t,s}. "
            "Default: 1 (progress over the most recent epoch)."
        ),
    )
    parser.add_argument(
        "--full-epoch",
        type=int,
        default=None,
        help=(
            "Explicit full checkpoint epoch. If omitted, the script picks the largest epoch "
            "shared by all runs; if that epoch is odd and --keep-odd-full-epoch is not set, "
            "it is nudged down by 1 so the half checkpoint is an integer."
        ),
    )
    parser.add_argument(
        "--half-epoch",
        type=int,
        default=None,
        help="Explicit half checkpoint epoch. If omitted, the script uses half of the selected full checkpoint.",
    )
    parser.add_argument(
        "--keep-odd-full-epoch",
        action="store_true",
        help="Do not auto-adjust an odd common max epoch down by 1.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling-average window for plotted curves. Default: 1 (no smoothing).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def infer_batch_size(path: Path) -> int:
    match = re.search(r"metrics_(\d+)\.csv$", path.name)
    if not match:
        raise MetricsError(
            f"Could not infer batch size from filename '{path.name}'. Expected pattern metrics_{{batch_size}}.csv"
        )
    return int(match.group(1))


def parse_gpu_utilization(value: object) -> float:
    """Return average GPU utilization for one epoch.

    Supported formats:
      - JSON dict string: '{"rank0/cuda:0": 59.6, ...}'
      - Python-literal dict string
      - numeric scalar
      - already-parsed dict
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, dict):
        vals = [float(v) for v in value.values()]
        return float(np.mean(vals)) if vals else np.nan

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.nan

    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                return float(text)
            except ValueError as exc:
                raise MetricsError(f"Could not parse gpu_utilization value: {text!r}") from exc

    if isinstance(parsed, dict):
        vals = [float(v) for v in parsed.values()]
        return float(np.mean(vals)) if vals else np.nan
    if isinstance(parsed, (int, float, np.integer, np.floating)):
        return float(parsed)

    raise MetricsError(f"Unsupported gpu_utilization format: {text!r}")


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_run(path_str: str) -> RunData:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    batch_size = infer_batch_size(path)
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise MetricsError(f"File {path.name} is missing required columns: {missing_str}")

    ensure_numeric(
        df,
        [
            "epoch",
            "train_loss",
            "train_top1",
            "val_loss",
            "val_top1",
            "val_top5",
            "train_throughput_img_s",
            "epoch_time_s",
            "peak_mem_gb",
            "lr",
        ],
    )

    if df["epoch"].isna().any():
        raise MetricsError(f"File {path.name} contains non-numeric values in the epoch column.")

    df = df.copy()
    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)
    df["gpu_utilization_mean"] = df["gpu_utilization"].apply(parse_gpu_utilization)
    df["images_this_epoch"] = df["train_throughput_img_s"] * df["epoch_time_s"]
    df["cumulative_time_s"] = pd.to_numeric(df["epoch_time_s"], errors="coerce").cumsum()
    df["cumulative_images"] = pd.to_numeric(df["images_this_epoch"], errors="coerce").cumsum()
    return RunData(batch_size=batch_size, path=path, df=df)


def safe_rolling(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def plot_metric_curves(
    runs: Sequence[RunData],
    metric: str,
    ylabel: str,
    title: str,
    outpath: Path,
    smooth: int,
    dpi: int,
) -> None:
    plt.figure(figsize=(10, 6))
    for run in sorted(runs, key=lambda r: r.batch_size):
        df = run.df
        x = pd.to_numeric(df["epoch"], errors="coerce")
        y = pd.to_numeric(df[metric], errors="coerce")
        y = safe_rolling(y, smooth)
        plt.plot(x, y, label=f"BS={run.batch_size}", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def shared_epochs(runs: Sequence[RunData]) -> list[int]:
    epoch_sets = [set(run.df["epoch"].astype(int).tolist()) for run in runs]
    if not epoch_sets:
        return []
    shared = set.intersection(*epoch_sets)
    return sorted(shared)


def choose_checkpoints(
    shared_epoch_values: Sequence[int],
    full_epoch: int | None,
    half_epoch: int | None,
    keep_odd_full_epoch: bool,
) -> tuple[int, int]:
    if not shared_epoch_values:
        raise MetricsError("No common epochs exist across the provided runs.")

    shared_set = set(shared_epoch_values)

    if full_epoch is None:
        full_epoch_value = int(max(shared_epoch_values))
        if (
            not keep_odd_full_epoch
            and full_epoch_value > 1
            and full_epoch_value % 2 == 1
            and (full_epoch_value - 1) in shared_set
        ):
            full_epoch_value -= 1
    else:
        if full_epoch not in shared_set:
            raise MetricsError(
                f"Requested --full-epoch {full_epoch} is not shared by all runs. "
                f"Available shared epochs span {min(shared_epoch_values)}..{max(shared_epoch_values)}."
            )
        full_epoch_value = int(full_epoch)

    if half_epoch is None:
        half_target = full_epoch_value // 2
        candidates = [epoch for epoch in shared_epoch_values if epoch <= half_target]
        if not candidates:
            raise MetricsError(
                f"Could not determine a valid half checkpoint for full epoch {full_epoch_value}."
            )
        half_epoch_value = int(max(candidates))
    else:
        if half_epoch not in shared_set:
            raise MetricsError(
                f"Requested --half-epoch {half_epoch} is not shared by all runs. "
                f"Available shared epochs span {min(shared_epoch_values)}..{max(shared_epoch_values)}."
            )
        half_epoch_value = int(half_epoch)

    if half_epoch_value >= full_epoch_value:
        raise MetricsError(
            f"Half checkpoint ({half_epoch_value}) must be smaller than full checkpoint ({full_epoch_value})."
        )

    return half_epoch_value, full_epoch_value


def get_epoch_index(df: pd.DataFrame, epoch: int) -> int:
    matches = np.flatnonzero(df["epoch"].to_numpy(dtype=int) == int(epoch))
    if len(matches) == 0:
        raise MetricsError(f"Epoch {epoch} not found in run.")
    return int(matches[-1])


def numeric_scalar(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def checkpoint_metrics(
    df: pd.DataFrame,
    epoch: int,
    objective_metric: str,
    se_window_epochs: int,
) -> dict[str, float | int]:
    idx = get_epoch_index(df, epoch)

    time_taken_s = numeric_scalar(df.loc[idx, "cumulative_time_s"])
    cumulative_images = numeric_scalar(df.loc[idx, "cumulative_images"])
    throughput = cumulative_images / time_taken_s if time_taken_s > 0 else np.nan
    loss_value = numeric_scalar(df.loc[idx, objective_metric])

    result: dict[str, float | int] = {
        "epoch": int(epoch),
        "time_taken_s": time_taken_s,
        "loss": loss_value,
        "throughput_img_s": throughput,
        "se_progress_per_img": np.nan,
        "goodput_progress_per_s": np.nan,
        "se_window_start_epoch": np.nan,
    }

    prev_idx = idx - int(se_window_epochs)
    if prev_idx < 0:
        return result

    loss_prev = numeric_scalar(df.loc[prev_idx, objective_metric])
    cumulative_images_prev = numeric_scalar(df.loc[prev_idx, "cumulative_images"])
    images_window = cumulative_images - cumulative_images_prev
    progress = loss_prev - loss_value

    if images_window > 0 and not np.isnan(progress):
        se_value = progress / images_window
        goodput_value = throughput * se_value if not np.isnan(throughput) else np.nan
        result["se_progress_per_img"] = se_value
        result["goodput_progress_per_s"] = goodput_value
        result["se_window_start_epoch"] = int(df.loc[prev_idx, "epoch"])

    return result


def build_summary(
    runs: Sequence[RunData],
    objective_metric: str,
    half_epoch: int,
    full_epoch: int,
    se_window_epochs: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda r: r.batch_size):
        half_metrics = checkpoint_metrics(
            run.df,
            epoch=half_epoch,
            objective_metric=objective_metric,
            se_window_epochs=se_window_epochs,
        )
        full_metrics = checkpoint_metrics(
            run.df,
            epoch=full_epoch,
            objective_metric=objective_metric,
            se_window_epochs=se_window_epochs,
        )

        row: dict[str, object] = {
            "batch_size": run.batch_size,
            "file": str(run.path),
            f"time_taken_epoch_{half_epoch}_s": half_metrics["time_taken_s"],
            f"loss_epoch_{half_epoch}": half_metrics["loss"],
            f"throughput_epoch_{half_epoch}_img_s": half_metrics["throughput_img_s"],
            f"statistical_efficiency_epoch_{half_epoch}_progress_per_img": half_metrics[
                "se_progress_per_img"
            ],
            f"goodput_epoch_{half_epoch}_progress_per_s": half_metrics[
                "goodput_progress_per_s"
            ],
            f"time_taken_epoch_{full_epoch}_s": full_metrics["time_taken_s"],
            f"loss_epoch_{full_epoch}": full_metrics["loss"],
            f"throughput_epoch_{full_epoch}_img_s": full_metrics["throughput_img_s"],
            f"statistical_efficiency_epoch_{full_epoch}_progress_per_img": full_metrics[
                "se_progress_per_img"
            ],
            f"goodput_epoch_{full_epoch}_progress_per_s": full_metrics[
                "goodput_progress_per_s"
            ],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_two_checkpoint_summary(
    summary_df: pd.DataFrame,
    half_epoch: int,
    full_epoch: int,
    half_column: str,
    full_column: str,
    ylabel: str,
    title: str,
    outpath: Path,
    dpi: int,
) -> None:
    data = summary_df.sort_values("batch_size")
    x = pd.to_numeric(data["batch_size"], errors="coerce")
    y_half = pd.to_numeric(data[half_column], errors="coerce")
    y_full = pd.to_numeric(data[full_column], errors="coerce")

    plt.figure(figsize=(9, 5.5))
    plt.plot(x, y_half, marker="o", linewidth=2, label=f"Epoch {half_epoch}")
    plt.plot(x, y_full, marker="o", linewidth=2, label=f"Epoch {full_epoch}")
    plt.xlabel("Batch size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def write_readme(
    outdir: Path,
    objective_metric: str,
    se_window_epochs: int,
    half_epoch: int,
    full_epoch: int,
) -> None:
    lines = [
        "ML710 metrics analysis summary",
        "============================",
        "",
        f"Objective metric for SE and goodput: {objective_metric}",
        f"SE window (epochs): {se_window_epochs}",
        f"Half checkpoint epoch: {half_epoch}",
        f"Full checkpoint epoch: {full_epoch}",
        "",
        "Definitions used in this script:",
        "- throughput_t = cumulative_images_to_t / cumulative_time_to_t",
        "- SE_{t,s} = (L_{t-s} - L_t) / images_processed_between_(t-s,_t]",
        "- goodput_{t,s} = throughput_t * SE_{t,s}",
        "",
        "Interpretation:",
        "- Positive SE means the loss decreased over the chosen window.",
        "- Negative SE means the loss increased over the chosen window.",
        "- Because the logs are epoch-level, epochs are treated as coarse-grained iterations.",
        "",
        "Generated files:",
        "- summary_metrics.csv",
        "- val_loss_vs_epoch.png",
        "- train_loss_vs_epoch.png",
        "- val_top1_vs_epoch.png",
        "- gpu_utilization_vs_epoch.png",
        "- peak_mem_vs_epoch.png",
        "- time_taken_vs_batch_size.png",
        "- throughput_vs_batch_size.png",
        "- statistical_efficiency_vs_batch_size.png",
        "- goodput_vs_batch_size.png",
    ]
    (outdir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smooth < 1:
        raise MetricsError("--smooth must be >= 1")
    if args.se_window_epochs < 1:
        raise MetricsError("--se-window-epochs must be >= 1")

    runs = [load_run(path) for path in args.files]
    batch_sizes = [run.batch_size for run in runs]
    if len(batch_sizes) != len(set(batch_sizes)):
        duplicates = sorted({b for b in batch_sizes if batch_sizes.count(b) > 1})
        dup_str = ", ".join(map(str, duplicates))
        raise MetricsError(f"Duplicate batch sizes detected: {dup_str}")

    shared_epoch_values = shared_epochs(runs)
    half_epoch, full_epoch = choose_checkpoints(
        shared_epoch_values=shared_epoch_values,
        full_epoch=args.full_epoch,
        half_epoch=args.half_epoch,
        keep_odd_full_epoch=args.keep_odd_full_epoch,
    )

    if args.se_window_epochs >= half_epoch:
        raise MetricsError(
            "--se-window-epochs must be smaller than the half checkpoint epoch so that SE can be computed there. "
            f"Current half checkpoint: {half_epoch}, requested window: {args.se_window_epochs}."
        )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for metric, (ylabel, title, filename) in PLOT_METRICS.items():
        plot_metric_curves(
            runs=runs,
            metric=metric,
            ylabel=ylabel,
            title=title,
            outpath=outdir / filename,
            smooth=args.smooth,
            dpi=args.dpi,
        )

    summary_df = build_summary(
        runs=runs,
        objective_metric=args.objective_metric,
        half_epoch=half_epoch,
        full_epoch=full_epoch,
        se_window_epochs=args.se_window_epochs,
    ).sort_values("batch_size")
    summary_df.to_csv(outdir / "summary_metrics.csv", index=False)

    plot_two_checkpoint_summary(
        summary_df=summary_df,
        half_epoch=half_epoch,
        full_epoch=full_epoch,
        half_column=f"time_taken_epoch_{half_epoch}_s",
        full_column=f"time_taken_epoch_{full_epoch}_s",
        ylabel="Time taken (s)",
        title="Time taken vs batch size",
        outpath=outdir / "time_taken_vs_batch_size.png",
        dpi=args.dpi,
    )
    plot_two_checkpoint_summary(
        summary_df=summary_df,
        half_epoch=half_epoch,
        full_epoch=full_epoch,
        half_column=f"throughput_epoch_{half_epoch}_img_s",
        full_column=f"throughput_epoch_{full_epoch}_img_s",
        ylabel="Throughput (img/s)",
        title="Throughput vs batch size",
        outpath=outdir / "throughput_vs_batch_size.png",
        dpi=args.dpi,
    )
    plot_two_checkpoint_summary(
        summary_df=summary_df,
        half_epoch=half_epoch,
        full_epoch=full_epoch,
        half_column=f"statistical_efficiency_epoch_{half_epoch}_progress_per_img",
        full_column=f"statistical_efficiency_epoch_{full_epoch}_progress_per_img",
        ylabel="Statistical efficiency (loss progress/img)",
        title="Statistical efficiency vs batch size",
        outpath=outdir / "statistical_efficiency_vs_batch_size.png",
        dpi=args.dpi,
    )
    plot_two_checkpoint_summary(
        summary_df=summary_df,
        half_epoch=half_epoch,
        full_epoch=full_epoch,
        half_column=f"goodput_epoch_{half_epoch}_progress_per_s",
        full_column=f"goodput_epoch_{full_epoch}_progress_per_s",
        ylabel="Goodput (loss progress/s)",
        title="Goodput vs batch size",
        outpath=outdir / "goodput_vs_batch_size.png",
        dpi=args.dpi,
    )

    write_readme(
        outdir=outdir,
        objective_metric=args.objective_metric,
        se_window_epochs=args.se_window_epochs,
        half_epoch=half_epoch,
        full_epoch=full_epoch,
    )

    display_cols = [
        "batch_size",
        f"time_taken_epoch_{half_epoch}_s",
        f"loss_epoch_{half_epoch}",
        f"throughput_epoch_{half_epoch}_img_s",
        f"statistical_efficiency_epoch_{half_epoch}_progress_per_img",
        f"goodput_epoch_{half_epoch}_progress_per_s",
        f"time_taken_epoch_{full_epoch}_s",
        f"loss_epoch_{full_epoch}",
        f"throughput_epoch_{full_epoch}_img_s",
        f"statistical_efficiency_epoch_{full_epoch}_progress_per_img",
        f"goodput_epoch_{full_epoch}_progress_per_s",
    ]

    print(f"Objective metric: {args.objective_metric}")
    print(f"SE window (epochs): {args.se_window_epochs}")
    print(f"Half checkpoint epoch: {half_epoch}")
    print(f"Full checkpoint epoch: {full_epoch}")
    print("\nSummary:")
    print(summary_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.6g}"))
    print(f"\nSaved outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except MetricsError as exc:
        raise SystemExit(f"Error: {exc}")
