"""Utilities for fractal analysis of coastline geometries."""

from .io import load_coastline
from .raster import rasterize_to_binary
from .grid import GridSpec, make_grids, count_boxes_vector
from .boxcount import (
    grid_count_vector,
    boxcount_series,
    aggregate_counts,
    choose_linear_window,
    fit_and_ci,
    sensitivity_analysis,
    summarize_sensitivity,
)
from .plots import report_plots, plot_geometry, plot_loglog
from .validation import (
    generate_straight_line,
    generate_noise_curve,
    simplification_study,
    run_sanity_checks,
)
from .workflow import run_workflow

__all__ = [
    "load_coastline",
    "rasterize_to_binary",
    "GridSpec",
    "make_grids",
    "count_boxes_vector",
    "grid_count_vector",
    "boxcount_series",
    "aggregate_counts",
    "choose_linear_window",
    "fit_and_ci",
    "sensitivity_analysis",
    "summarize_sensitivity",
    "report_plots",
    "plot_geometry",
    "plot_loglog",
    "generate_straight_line",
    "generate_noise_curve",
    "simplification_study",
    "run_sanity_checks",
    "run_workflow",
]
