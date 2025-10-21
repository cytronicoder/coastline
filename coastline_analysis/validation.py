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
    r"""Create a straight line geometry for sanity checks.

    The resulting polyline has theoretical box-counting dimension \(D = 1\).
    Regardless of ``segments`` the geometry lies on the ``x``-axis and therefore
    should produce perfectly linear log–log counts up to numerical precision.

    Args:
        length: Total length of the line.
        segments: Number of segments (resolution).

    Returns:
        A straight :class:`shapely.geometry.LineString` geometry.
    """

    xs = np.linspace(0, length, segments + 1)
    ys = np.zeros_like(xs)
    return LineString(np.column_stack([xs, ys]))


def generate_noise_curve(
    length: float = 1000.0, segments: int = 1024, noise: float = 100.0
) -> LineString:
    r"""Generate a fractal-like noisy coastline surrogate.

    A Gaussian random walk is generated along the ``y`` direction and scaled to
    the provided ``noise`` amplitude. Such curves resemble fractional Brownian
    motion with Hurst exponent \(H \approx 0.5\), giving an expected box-counting
    dimension \(D = 2 - H \approx 1.5\).

    Args:
        length: Total length of the curve.
        segments: Number of segments for discretization.
        noise: Amplitude of the noise.

    Returns:
        A noisy :class:`shapely.geometry.LineString` geometry.
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
    r"""Run the primary workflow on a geometry and return summary stats.

    The function executes the workflow components in order: raw box counts,
    aggregation, linear-window selection, parametric fit, and grid sensitivity.
    The output summarises both the global slope \(D\) and bootstrap sensitivity
    measures so that empirical results can be compared against theoretical
    expectations (e.g., \(D = 1\) for straight lines).

    Args:
        geometry: The geometry to analyze.
        eps_list: Sequence of scales.
        offsets: Iterable of offset fractions.
        rotations: Iterable of rotation angles.

    Returns:
        Summary DataFrame with fractal dimension estimates.
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
    r"""Assess how Douglas–Peucker simplification affects fractal estimates.

    Simplification removes vertices whose orthogonal distance to the original
    curve is smaller than ``tolerance``. Because box counts at scales below
    ``tolerance`` decrease, the observed slope \(D\) typically drops toward the
    Euclidean value 1. This helper quantifies that trend across a range of
    tolerances.

    Args:
        geometry: The original geometry.
        tolerances: Sequence of simplification tolerances.
        eps_list: Sequence of scales.
        offsets: Iterable of offset fractions.
        rotations: Iterable of rotation angles.

    Returns:
        DataFrame with fractal dimensions for each tolerance.
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
