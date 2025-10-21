"""Plotting utilities for the coastline fractal workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

GeometryLike = LineString | MultiLineString


__all__ = ["report_plots", "plot_geometry", "plot_loglog"]


def _ensure_output_dir(path: Path | None) -> Path:
    """Ensure the output directory exists and return it.

    Args:
        path: The directory path, or None for current working directory.

    Returns:
        The ensured directory path.
    """
    if path is None:
        return Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_plots(
    geometry: GeometryLike,
    aggregated: pd.DataFrame,
    fit_stats: Dict[str, float],
    window_indices: tuple[int, int],
    residuals: np.ndarray,
    sensitivity: pd.DataFrame,
    output_dir: Path | None = None,
) -> Dict[str, Path]:
    """Create report figures.

    Args:
        geometry: Coastline geometry.
        aggregated: Aggregated counts.
        fit_stats: Fit stats.
        window_indices: (start, end) indices.
        residuals: Fit residuals.
        sensitivity: Sensitivity results.
        output_dir: Save dir, default cwd.

    Returns:
        Dict of figure names to paths.
    """

    output_path = _ensure_output_dir(output_dir)
    figures = {}

    # Figure 1: map view
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    geom = unary_union(geometry) if isinstance(geometry, (list, tuple)) else geometry
    if isinstance(geom, LineString):
        xs, ys = geom.xy
        ax1.plot(xs, ys, color="steelblue", linewidth=1.5)
    else:
        for line in geom.geoms:
            xs, ys = line.xy
            ax1.plot(xs, ys, color="steelblue", linewidth=1.0)
    ax1.set_title("Figure 1. Coastline in study CRS")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    figures["figure1"] = output_path / "figure1_map.png"
    fig1.savefig(figures["figure1"], dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: log-log plot with fit and residuals inset
    fig2, (ax2, ax2_resid) = plt.subplots(
        2, 1, figsize=(6, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax2.errorbar(
        aggregated["log_inv_eps"],
        aggregated["log_mean_N"],
        yerr=aggregated["std_N"],
        fmt="o",
        color="black",
        label="Mean log N",
    )
    x_band = aggregated.iloc[window_indices[0] : window_indices[1] + 1][
        "log_inv_eps"
    ].to_numpy()
    y_fit = fit_stats["D"] * x_band + fit_stats["intercept"]
    ax2.plot(x_band, y_fit, color="crimson", label=f"Fit D={fit_stats['D']:.3f}")
    ax2.fill_between(
        x_band,
        (fit_stats["CI_low"] * x_band + fit_stats["intercept"]),
        (fit_stats["CI_high"] * x_band + fit_stats["intercept"]),
        color="crimson",
        alpha=0.2,
        label="95% CI",
    )
    ax2.set_ylabel("log N(ε)")
    ax2.set_title("Figure 2. Box counting results")
    ax2.legend()

    ax2_resid.axhline(0, color="gray", linestyle="--")
    ax2_resid.plot(x_band, residuals, marker="o", linestyle="-", color="black")
    ax2_resid.set_xlabel("log(1/ε)")
    ax2_resid.set_ylabel("Residuals")

    figures["figure2"] = output_path / "figure2_loglog.png"
    fig2.savefig(figures["figure2"], dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Figure 4: sensitivity plot
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    rotations = sorted(sensitivity["rotation"].unique())
    for rotation in rotations:
        subset = sensitivity[sensitivity["rotation"] == rotation]
        yerr = np.vstack(
            [
                (subset["mean_D"] - subset["CI_low"]).to_numpy(),
                (subset["CI_high"] - subset["mean_D"]).to_numpy(),
            ]
        )
        ax4.errorbar(
            subset["offset_x"],
            subset["mean_D"],
            yerr=yerr,
            fmt="o",
            label=f"rot={rotation:.1f}°",
        )
    ax4.set_xlabel("Offset fraction (x)")
    ax4.set_ylabel("Estimated D")
    ax4.set_title("Figure 4. Sensitivity across grid realisations")
    ax4.legend(title="Rotation")
    figures["figure4"] = output_path / "figure4_sensitivity.png"
    fig4.savefig(figures["figure4"], dpi=300, bbox_inches="tight")
    plt.close(fig4)

    return figures


def plot_geometry(geometry: GeometryLike, title: str = "Geometry", save_path: Path | None = None):
    """Plot geometry.

    Args:
        geometry: Geometry to plot.
        title: Plot title.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    geom = unary_union(geometry) if isinstance(geometry, (list, tuple)) else geometry
    if isinstance(geom, LineString):
        xs, ys = geom.xy
        ax.plot(xs, ys, color="steelblue", linewidth=1.5)
    else:
        for line in geom.geoms:
            xs, ys = line.xy
            ax.plot(xs, ys, color="steelblue", linewidth=1.0)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_loglog(aggregated: pd.DataFrame, title: str = "Log-Log Plot", save_path: Path | None = None):
    """Plot log-log: log(1/ε) vs log N(ε).

    Args:
        aggregated: Aggregated data.
        title: Plot title.
        save_path: Optional save path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        aggregated["log_inv_eps"],
        aggregated["log_mean_N"],
        color="black",
        label="Data points",
    )
    ax.set_xlabel("log(1/ε)")
    ax.set_ylabel("log N(ε)")
    ax.set_title(title)
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
