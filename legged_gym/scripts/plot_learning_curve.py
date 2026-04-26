#!/usr/bin/env python3
"""Plot learning curves from progress.csv logs (mean reward with ±std shaded band)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# TrueType text in PDF so labels scale cleanly in viewers and LaTeX.
mpl.rcParams["pdf.fonttype"] = 42


BIN_WIDTH = 50 * 4096 * 24


def load_progress_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    env_steps = data[:, 0]
    mean_reward = data[:, 1]
    std_reward = data[:, 2]
    return env_steps, mean_reward, std_reward


def subsample_progress(
    env_steps: np.ndarray,
    mean_reward: np.ndarray,
    std_reward: np.ndarray,
    bin_width: float = BIN_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the CSV row whose env_steps is closest to each bin center."""
    max_step = env_steps.max()
    centers = np.arange(bin_width / 2, max_step + bin_width, bin_width)
    picks = np.unique(np.abs(env_steps[:, None] - centers[None, :]).argmin(axis=0))
    return env_steps[picks], mean_reward[picks], std_reward[picks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: data/learning_curve.pdf next to this script)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI when saving raster formats (.png, .jpg, …); ignored for vector PDF/SVG",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo_root = root.parents[1]
    random_csv = repo_root / "data/logs/flat_go2/Mar21_01-10-37_random/progress.csv"
    sac_csv = repo_root / "data/logs/flat_go2_sac/Mar21_23-22-12_/progress.csv"
    ppo_csv = repo_root / "data/logs/flat_go2/Mar21_19-24-54_ppo/progress.csv"
    transformer_csv = repo_root / "data/logs/flat_go2_transformer/Apr13_12-58-53_/progress.csv"

    series = [
        ("PPO", ppo_csv, "C0"),
        ("SAC", sac_csv, "C1"),
        # ("MODEL_BASE", csv, "C2"),
        # ("PPO + Multi Critic", csv, "C3"),
        ("PPO + Transformer", transformer_csv, "C4"),
        ("Random policy", random_csv, "C5"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ppo_steps, _, _ = load_progress_csv(ppo_csv)
    x_max = ppo_steps.max()

    for label, csv_path, color in series:
        steps, mean_r, std_r = load_progress_csv(csv_path)
        order = np.argsort(steps)
        steps, mean_r, std_r = steps[order], mean_r[order], std_r[order]
        mask = steps <= x_max
        steps, mean_r, std_r = steps[mask], mean_r[mask], std_r[mask]
        b_steps, b_mean, b_std = subsample_progress(steps, mean_r, std_r)
        low = b_mean - b_std
        high = b_mean + b_std
        ax.fill_between(
            b_steps,
            low,
            high,
            color=color,
            alpha=0.25,
            linewidth=0,
            label=label,
        )
        ax.plot(b_steps, b_mean, color=color, linewidth=1.2)

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Total environment steps")
    ax.set_ylabel("Mean reward")
    ax.set_title("Flat-terrain Go2: Learning curves (mean reward vs environment steps)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = args.out if args.out is not None else root / "learning_curve.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    save_kw: dict = {}
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
        save_kw["dpi"] = args.dpi
    fig.savefig(out, **save_kw)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
