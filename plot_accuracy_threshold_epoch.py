from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ThresholdPoint:
    batch_size: int
    metrics_csv: Path
    first_epoch: int | None
    best_metric: float
    max_epoch: int

    @property
    def reached(self) -> bool:
        return self.first_epoch is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot batch size vs the first epoch where validation accuracy reaches a threshold."
        )
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("results/critical_batch_sweep"),
        help="Directory containing bs_<batch_size>/metrics.csv files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=65.0,
        help="Accuracy threshold to detect. Default: 65.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val_top1",
        help="Metric column to compare against the threshold. Default: val_top1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to <sweep-dir>/epoch_to_accuracy_<threshold>.svg.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <sweep-dir>/epoch_to_accuracy_<threshold>.csv.",
    )
    return parser.parse_args()


def infer_batch_size(path: Path) -> int:
    for part in [path.parent.name, path.name]:
        match = re.search(r"(?:bs_|batch_?|metrics_)(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer batch size from {path}")


def find_metric_files(sweep_dir: Path) -> list[Path]:
    files = sorted(sweep_dir.glob("bs_*/metrics.csv"), key=infer_batch_size)
    if not files:
        raise FileNotFoundError(f"No metrics.csv files found under {sweep_dir}/bs_*")
    return files


def load_threshold_point(metrics_csv: Path, *, metric: str, threshold: float) -> ThresholdPoint:
    first_epoch: int | None = None
    best_metric = -math.inf
    max_epoch = 0

    with metrics_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(f"{metrics_csv} does not contain metric '{metric}'. Available: {available}")
        if "epoch" not in reader.fieldnames:
            raise ValueError(f"{metrics_csv} does not contain an 'epoch' column.")

        for row in reader:
            epoch = int(float(row["epoch"]))
            value = float(row[metric])
            max_epoch = max(max_epoch, epoch)
            best_metric = max(best_metric, value)
            if first_epoch is None and value >= threshold:
                first_epoch = epoch

    if max_epoch == 0:
        raise ValueError(f"{metrics_csv} has no data rows.")

    return ThresholdPoint(
        batch_size=infer_batch_size(metrics_csv),
        metrics_csv=metrics_csv,
        first_epoch=first_epoch,
        best_metric=best_metric,
        max_epoch=max_epoch,
    )


def nice_epoch_max(max_epoch: int) -> int:
    if max_epoch <= 10:
        return max_epoch
    return int(math.ceil(max_epoch / 10.0) * 10)


def default_stem(threshold: float) -> str:
    if threshold.is_integer():
        return f"epoch_to_accuracy_{int(threshold)}"
    return f"epoch_to_accuracy_{str(threshold).replace('.', '_')}"


def write_summary_csv(path: Path, points: list[ThresholdPoint], *, metric: str, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "batch_size",
            "metric",
            "threshold",
            "reached",
            "first_epoch",
            "best_metric",
            "max_epoch",
            "metrics_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "batch_size": point.batch_size,
                    "metric": metric,
                    "threshold": threshold,
                    "reached": point.reached,
                    "first_epoch": "" if point.first_epoch is None else point.first_epoch,
                    "best_metric": f"{point.best_metric:.6f}",
                    "max_epoch": point.max_epoch,
                    "metrics_csv": point.metrics_csv,
                }
            )


def svg_text(x: float, y: float, text: str, *, size: int = 13, anchor: str = "middle", weight: str = "400") -> str:
    safe_text = html.escape(text)
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="#222">{safe_text}</text>'
    )


def render_svg(points: list[ThresholdPoint], *, metric: str, threshold: float) -> str:
    width = 900
    height = 560
    margin_left = 82
    margin_right = 36
    margin_top = 72
    margin_bottom = 86
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_epoch = nice_epoch_max(max(point.max_epoch for point in points))
    min_epoch = 0
    denom = max(max_epoch - min_epoch, 1)

    def x_for(index: int) -> float:
        if len(points) == 1:
            return margin_left + plot_width / 2
        return margin_left + index * plot_width / (len(points) - 1)

    def y_for(epoch: int) -> float:
        return margin_top + plot_height - ((epoch - min_epoch) / denom) * plot_height

    tick_count = 5
    y_ticks = [round(max_epoch * i / tick_count) for i in range(tick_count + 1)]
    y_ticks = sorted(set(y_ticks))

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        svg_text(
            width / 2,
            30,
            f"First Epoch Reaching {threshold:g}% Accuracy",
            size=22,
            weight="700",
        ),
        svg_text(width / 2, 54, f"Metric: {metric}", size=13),
    ]

    for tick in y_ticks:
        y = y_for(tick)
        elements.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(svg_text(margin_left - 12, y + 4, str(tick), size=12, anchor="end"))

    elements.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222" stroke-width="1.5"/>',
            f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#222" stroke-width="1.5"/>',
            svg_text(width / 2, height - 26, "Batch size", size=14, weight="700"),
            (
                f'<text x="24" y="{height / 2:.2f}" text-anchor="middle" font-size="14" '
                'font-weight="700" fill="#222" transform="rotate(-90 24 '
                f'{height / 2:.2f})">First epoch reaching threshold</text>'
            ),
        ]
    )

    reached_points = []
    for index, point in enumerate(points):
        x = x_for(index)
        elements.append(
            f'<line x1="{x:.2f}" y1="{height - margin_bottom}" x2="{x:.2f}" y2="{height - margin_bottom + 6}" stroke="#222" stroke-width="1"/>'
        )
        elements.append(svg_text(x, height - margin_bottom + 25, str(point.batch_size), size=12))
        if point.first_epoch is not None:
            reached_points.append((x, y_for(point.first_epoch), point))

    if reached_points:
        path_data = " ".join(
            f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
            for index, (x, y, _) in enumerate(reached_points)
        )
        elements.append(f'<path d="{path_data}" fill="none" stroke="#2563eb" stroke-width="3"/>')

    for x, y, point in reached_points:
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#2563eb"/>')
        elements.append(svg_text(x, y - 12, f"E{point.first_epoch}", size=12, weight="700"))

    for index, point in enumerate(points):
        if point.first_epoch is not None:
            continue
        x = x_for(index)
        y = y_for(point.max_epoch)
        elements.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#fff" stroke="#dc2626" stroke-width="2"/>'
        )
        elements.append(svg_text(x, y - 12, f">{point.max_epoch}", size=12, weight="700"))

    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_svg(path: Path, svg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)


def main() -> None:
    args = parse_args()
    metric_files = find_metric_files(args.sweep_dir)
    points = [
        load_threshold_point(path, metric=args.metric, threshold=args.threshold)
        for path in metric_files
    ]
    points.sort(key=lambda point: point.batch_size)

    stem = default_stem(args.threshold)
    output = args.output or (args.sweep_dir / f"{stem}.svg")
    summary_csv = args.summary_csv or (args.sweep_dir / f"{stem}.csv")

    write_summary_csv(summary_csv, points, metric=args.metric, threshold=args.threshold)
    write_svg(output, render_svg(points, metric=args.metric, threshold=args.threshold))

    print(f"Wrote graph: {output.resolve()}")
    print(f"Wrote summary: {summary_csv.resolve()}")
    for point in points:
        reached = "not reached" if point.first_epoch is None else f"epoch {point.first_epoch}"
        print(f"batch_size={point.batch_size}: {reached} (best {args.metric}={point.best_metric:.2f})")


if __name__ == "__main__":
    main()
