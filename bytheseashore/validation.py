r"""Validation helpers for the coastline fractal workflow.

The routines in this module generate synthetic geometries with known fractal
properties and run the full analysis pipeline. They provide numerical checks
against theoretical expectations described in Mandelbrot (1967) and Falconer
(2014).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from .boxcount import (
    aggregate_counts,
    boxcount_series,
    choose_linear_window,
    fit_and_ci,
    sensitivity_analysis,
    summarize_sensitivity,
)

GeometryLike = LineString | MultiLineString


__all__ = [
    "generate_straight_line",
    "generate_noise_curve",
    "run_sanity_checks",
    "simplification_study",
]


def generate_straight_line(length: float = 1000.0, segments: int = 2) -> LineString:
    r"""Generate straight line, D=1.

    Args:
        length: Total length.
        segments: Number of segments.

    Returns:
        LineString.
    """
    xs = np.linspace(0, length, segments + 1)
    ys = np.zeros_like(xs)
    return LineString(np.column_stack([xs, ys]))


def generate_noise_curve(
    length: float = 1000.0, segments: int = 1024, noise: float = 100.0
) -> LineString:
    r"""Generate noisy curve, D≈1.5.

    Gaussian random walk, H≈0.5.

    Args:
        length: Total length.
        segments: Number of segments.
        noise: Noise amplitude.

    Returns:
        LineString.
    """
    xs = np.linspace(0, length, segments)
    rng = np.random.default_rng(1234)
    ys = rng.standard_normal(segments).cumsum()
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9) * noise
    return LineString(np.column_stack([xs, ys]))


def run_sanity_checks(
    geometry: GeometryLike,
    eps_list: Sequence[float],
    offsets: Iterable[tuple[float, float]],
    rotations: Iterable[float],
) -> pd.DataFrame:
    r"""Run full workflow, return summary.

    Box counts, aggregate, fit, sensitivity.

    Args:
        geometry: Geometry.
        eps_list: Scales ε.
        offsets: (o_x, o_y).
        rotations: Angles θ.

    Returns:
        DataFrame with D estimates.
    """
    df = boxcount_series(geometry, eps_list, offsets, rotations)
    aggregated = aggregate_counts(df)
    window = choose_linear_window(aggregated)
    stats = fit_and_ci(aggregated.iloc[window.start : window.stop])
    per_grid = sensitivity_analysis(df, window)
    summary = summarize_sensitivity(per_grid)
    summary["D"] = stats["D"]
    summary["R2"] = stats["R2"]
    summary["CI_low_fit"] = stats["CI_low"]
    summary["CI_high_fit"] = stats["CI_high"]
    return summary


def simplification_study(
    geometry: GeometryLike,
    tolerances: Sequence[float],
    eps_list: Sequence[float],
    offsets: Iterable[tuple[float, float]],
    rotations: Iterable[float],
) -> pd.DataFrame:
    r"""Study simplification effect on D.

    Douglas-Peucker simplification, D vs tolerance.

    Args:
        geometry: Original geometry.
        tolerances: Simplification tolerances.
        eps_list: Scales ε.
        offsets: (o_x, o_y).
        rotations: Angles θ.

    Returns:
        DataFrame: tolerance, D, CI, R2, levels.
    """
    rows = []
    for tol in tolerances:
        simplified = geometry.simplify(tol)
        if isinstance(simplified, LineString):
            geom_for_analysis = simplified
        elif isinstance(simplified, MultiLineString):
            geom_for_analysis = simplified
        else:
            geom_for_analysis = (
                MultiLineString([simplified.boundary])
                if hasattr(simplified, "boundary")
                else geometry
            )
        df = boxcount_series(geom_for_analysis, eps_list, offsets, rotations)
        aggregated = aggregate_counts(df)
        window = choose_linear_window(aggregated)
        stats = fit_and_ci(aggregated.iloc[window.start : window.stop])
        rows.append(
            {
                "tolerance": tol,
                "D": stats["D"],
                "CI_low": stats["CI_low"],
                "CI_high": stats["CI_high"],
                "R2": stats["R2"],
                "levels": window.stop - window.start,
            }
        )
    return pd.DataFrame(rows)
