"""Validation helpers for the coastline fractal workflow."""

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


def generate_straight_line(length: float = 1000.0, segments: int = 2) -> LineString:
    """Create a straight line geometry for sanity checks."""

    xs = np.linspace(0, length, segments + 1)
    ys = np.zeros_like(xs)
    return LineString(np.column_stack([xs, ys]))


def generate_noise_curve(length: float = 1000.0, segments: int = 1024, noise: float = 100.0) -> LineString:
    """Generate a fractal-like noisy coastline surrogate."""

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
    """Run the primary workflow on a geometry and return summary stats."""

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
    """Assess how Douglas–Peucker simplification affects fractal estimates."""

    from shapely.geometry import MultiLineString as MLS

    rows = []
    for tol in tolerances:
        simplified = geometry.simplify(tol)
        if isinstance(simplified, LineString):
            geom_for_analysis = simplified
        elif isinstance(simplified, MultiLineString):
            geom_for_analysis = simplified
        else:
            geom_for_analysis = MLS([simplified.boundary]) if hasattr(simplified, "boundary") else geometry
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


__all__ = [
    "generate_straight_line",
    "generate_noise_curve",
    "run_sanity_checks",
    "simplification_study",
]
