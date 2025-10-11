#!/usr/bin/env python3
"""Plot SIMD benchmark summaries and attention breakdown figures."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Row = Dict[str, str]
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def _read_rows(csv_path: Path, required: Iterable[str]) -> List[Row]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [col for col in required if col not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing expected columns: {missing}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV appears empty: {csv_path}")
        return rows


def _parse_float(value: str, *, context: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Unable to parse '{value}' as float ({context})") from exc


# --- Benchmark overview ----------------------------------------------------

def plot_benchmarks(csv_path: Path, output_path: Path, dpi: int) -> None:
    rows = _read_rows(csv_path, required=("suite", "label", "speedup"))

    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for row in rows:
        suite = row["suite"]
        if suite in {
            "src/03_Examples/05_attention_block",
            "03_Examples/05_attention_block",
            "src/03_Examples/05_mha_block",
            "03_Examples/05_mha_block",
        }:
            # attention gets its own figure
            continue
        grouped.setdefault(suite, []).append(
            (row["label"], _parse_float(row["speedup"], context=f"suite={suite}, label={row['label']}") )
        )

    if not grouped:
        raise ValueError("No benchmark data to plot (all rows filtered?)")

    suites = sorted(grouped.keys())
    values = [sp for suite in suites for _, sp in grouped[suite]]
    median = sorted(values)[len(values) // 2]
    mean = sum(values) / len(values)

    count = len(suites)
    cols = math.ceil(math.sqrt(count))
    rows_count = math.ceil(count / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(4 * cols, 3 * rows_count), squeeze=False)
    fig.suptitle(f"SIMD Microbenchmark Speedups (median {median:.2f}×, mean {mean:.2f}×)", fontsize=14)

    for ax in axes.flat[count:]:
        ax.axis("off")

    for idx, suite in enumerate(suites):
        ax = axes.flat[idx]
        labels = [label for label, _ in grouped[suite]]
        speedups = [value for _, value in grouped[suite]]
        colors = ["#d62728" if sp < 1.0 else "#1f77b4" for sp in speedups]
        ypos = list(range(len(labels)))
        ax.barh(ypos, speedups, color=colors)
        ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Speedup (scalar / SIMD)")
        ax.set_title(suite, fontsize=10)
        ax.set_xlim(left=0)
        for y, sp in zip(ypos, speedups):
            ax.text(sp + 0.05, y, f"{sp:.2f}×", va="center", ha="left", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# --- Attention breakdown ---------------------------------------------------

def plot_attention(csv_path: Path, output_path: Path, dpi: int) -> None:
    rows = _read_rows(
        csv_path,
        required=(
            "component",
            "scalar_total_us",
            "simd_total_us",
            "speedup",
            "time_saved_us",
            "contribution_pct",
        ),
    )

    overall = next((row for row in rows if row["component"] == "overall"), None)
    if not overall:
        raise ValueError("attention_components.csv must contain an 'overall' row")

    components = [row for row in rows if row["component"] != "overall"]
    if not components:
        raise ValueError("attention_components.csv has no component rows")

    names = [row["component"] for row in components]
    speedups = [_parse_float(row["speedup"], context=row["component"]) for row in components]
    contributions = [
        _parse_float(row["contribution_pct"], context=row["component"])
        for row in components
    ]
    total_scalar = _parse_float(overall["scalar_total_us"], context="overall scalar_total_us")
    total_simd = _parse_float(overall["simd_total_us"], context="overall simd_total_us")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("SIMD Attention Block Breakdown", fontsize=14)

    ax_speed, ax_total, ax_contrib = axes

    ax_speed.bar(names, speedups, color="#1f77b4")
    ax_speed.set_ylabel("Speedup (scalar / SIMD)")
    ax_speed.set_title("Component Speedups")
    ax_speed.tick_params(axis="x", rotation=45)
    for label in ax_speed.get_xticklabels():
        label.set_horizontalalignment("right")
    for idx, val in enumerate(speedups):
        ax_speed.text(idx, val + 0.05, f"{val:.2f}×", ha="center", va="bottom", fontsize=8)

    ax_total.bar(["scalar", "simd"], [total_scalar, total_simd], color=["#d62728", "#2ca02c"])
    ax_total.set_ylabel("Microseconds")
    ax_total.set_title("End-to-End Latency")
    ax_total.set_ylim(0, max(total_scalar, total_simd) * 1.15)
    ax_total.text(0, total_scalar + 10, f"{total_scalar:.0f} μs", ha="center", va="bottom", fontsize=9)
    ax_total.text(1, total_simd + 10, f"{total_simd:.0f} μs", ha="center", va="bottom", fontsize=9)

    ax_contrib.barh(names, contributions, color="#9467bd")
    ax_contrib.set_xlabel("% of Total Speedup")
    ax_contrib.set_title("Contribution Share")
    for y, val in enumerate(contributions):
        ax_contrib.text(val + 0.5, y, f"{val:.1f}%", va="center", fontsize=8)
    ax_contrib.set_xlim(0, max(contributions + [10]) * 1.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# --- CLI -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks-csv",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "benchmark_results.csv",
        help="Path to benchmark results CSV.",
    )
    parser.add_argument(
        "--benchmarks-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "benchmark_speedups.png",
        help="Output path for the benchmark overview plot.",
    )
    parser.add_argument(
        "--attention-csv",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "attention_components.csv",
        help="Path to attention components CSV.",
    )
    parser.add_argument(
        "--attention-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "attention_speedups.png",
        help="Output path for the attention breakdown plot.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI")
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Skip plotting the attention breakdown (benchmark overview still generated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_benchmarks(args.benchmarks_csv, args.benchmarks_output, args.dpi)
    if not args.skip_attention:
        plot_attention(args.attention_csv, args.attention_output, args.dpi)


if __name__ == "__main__":  # pragma: no cover
    main()
