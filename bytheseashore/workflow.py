"""High-level workflow orchestration for coastline fractal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .boxcount import (
    aggregate_counts,
    boxcount_series,
    choose_linear_window,
    fit_and_ci,
    sensitivity_analysis,
    summarize_sensitivity,
)
from .io import load_coastline
from .plots import report_plots
from .validation import simplification_study


__all__ = ["run_workflow"]


def run_workflow(
    coastline_path: Path,
    eps_list: Sequence[float],
    offsets: Iterable[tuple[float, float]],
    rotations: Iterable[float],
    simplification_tolerances: Sequence[float] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run end-to-end fractal analysis.

    Args:
        coastline_path: Path to data.
        eps_list: Scales ε.
        offsets: (o_x, o_y).
        rotations: Angles θ.
        simplification_tolerances: Optional tolerances.
        output_dir: Output dir.

    Returns:
        Dict with geometry, data, fits, figures, tables.
    """
    geometry = load_coastline(coastline_path)
    box_counts = boxcount_series(geometry, eps_list, offsets, rotations)
    aggregated = aggregate_counts(box_counts)
    window = choose_linear_window(aggregated)
    band = aggregated.iloc[window.start : window.stop]
    fit_stats = fit_and_ci(band)
    sensitivity = summarize_sensitivity(sensitivity_analysis(box_counts, window))

    figures = report_plots(
        geometry=geometry,
        aggregated=aggregated,
        fit_stats=fit_stats,
        window_indices=(window.start, window.stop - 1),
        residuals=fit_stats["residuals"],
        sensitivity=sensitivity,
        output_dir=output_dir,
    )

    table1 = pd.DataFrame(
        [
            {
                "D": fit_stats["D"],
                "CI_low": fit_stats["CI_low"],
                "CI_high": fit_stats["CI_high"],
                "R2": fit_stats["R2"],
                "levels": window.stop - window.start,
                "n_grids": box_counts["grid_id"].nunique(),
            }
        ]
    )

    table2 = None
    if simplification_tolerances:
        table2 = simplification_study(
            geometry, simplification_tolerances, eps_list, offsets, rotations
        )

    return {
        "geometry": geometry,
        "box_counts": box_counts,
        "aggregated": aggregated,
        "window": window,
        "fit": fit_stats,
        "sensitivity": sensitivity,
        "figures": figures,
        "table1": table1,
        "table2": table2,
    }
