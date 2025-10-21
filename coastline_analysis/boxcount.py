"""Vector-based box-counting implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from shapely import affinity
from shapely.geometry import LineString, MultiLineString

from .grid import count_boxes_vector, make_grids

GeometryLike = LineString | MultiLineString


__all__ = [
    "grid_count_vector",
    "boxcount_series",
    "aggregate_counts",
    "choose_linear_window",
    "fit_and_ci",
    "sensitivity_analysis",
    "summarize_sensitivity",
]


@dataclass
class LinearWindow:
    """Represents a linear window in the log-log box count data.

    Attributes:
        start: Index of the first scale in the window.
        stop: Index of the last scale in the window.
        r2: R2 statistic for the linear fit.
        slope: Slope of the linear fit.
        intercept: Intercept of the linear fit.
        residuals: Residuals from the linear fit.
    """

    start: int
    stop: int
    r2: float
    slope: float
    intercept: float
    residuals: np.ndarray


def grid_count_vector(
    geometry: GeometryLike,
    eps: float,
    offsets: Iterable[Tuple[float, float]],
    rotations: Iterable[float],
) -> List[Dict[str, object]]:
    """Count intersected boxes for a geometry at a single scale.

    Args:
        geometry: The input geometry to analyze.
        eps: The scale (box size) for counting.
        offsets: Iterable of (x, y) offset fractions for grid positioning.
        rotations: Iterable of rotation angles in degrees.

    Returns:
        A list of dictionaries with box count records for each grid realization.
    """

    records: List[Dict[str, object]] = []
    for rotation in rotations:
        rotated = affinity.rotate(
            geometry, -rotation, origin="center", use_radians=False
        )
        grids = make_grids(rotated.bounds, eps, offsets, [rotation])
        for grid in grids:
            count = count_boxes_vector(rotated, grid)
            records.append(
                {
                    "eps": eps,
                    "rotation": rotation,
                    "offset_x": grid.offset[0],
                    "offset_y": grid.offset[1],
                    "grid_id": f"rot{rotation:.2f}_off{grid.offset[0]:.2f}_{grid.offset[1]:.2f}",
                    "N": count,
                }
            )
    return records


def boxcount_series(
    geometry: GeometryLike,
    eps_list: Sequence[float],
    offsets: Iterable[Tuple[float, float]],
    rotations: Iterable[float],
) -> pd.DataFrame:
    """Compute box counts for a set of scales.

    Args:
        geometry: The input geometry to analyze.
        eps_list: Sequence of scales (box sizes) to compute counts for.
        offsets: Iterable of (x, y) offset fractions for grid positioning.
        rotations: Iterable of rotation angles in degrees.

    Returns:
        A DataFrame with box counts, including log-transformed columns.
    """

    records: List[Dict[str, object]] = []
    for eps in eps_list:
        records.extend(grid_count_vector(geometry, eps, offsets, rotations))
    df = pd.DataFrame.from_records(records)
    df["log_inv_eps"] = np.log(1.0 / df["eps"].values)
    df["log_N"] = np.log(df["N"].clip(lower=1))
    return df


def aggregate_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate box counts across grid realisations for each scale.

    Args:
        df: DataFrame with individual box count records.

    Returns:
        Aggregated DataFrame with mean and std counts per scale.
    """

    grouped = (
        df.groupby("eps")
        .agg(
            log_inv_eps=("log_inv_eps", "first"),
            mean_N=("N", "mean"),
            std_N=("N", "std"),
            n_grids=("grid_id", "nunique"),
        )
        .reset_index()
        .sort_values("eps", ascending=False)
    )
    grouped["log_mean_N"] = np.log(grouped["mean_N"].clip(lower=1))
    grouped["std_N"] = grouped["std_N"].fillna(0.0)
    return grouped


def choose_linear_window(df: pd.DataFrame, min_levels: int = 4) -> LinearWindow:
    """Select the most linear contiguous region in the log-log curve.

    Args:
        df: Aggregated DataFrame with log-transformed counts.
        min_levels: Minimum number of scales for a window.

    Returns:
        The best LinearWindow with highest R2.

    Raises:
        ValueError: If not enough scales to form a linear window.
    """

    best = LinearWindow(
        start=0,
        stop=min_levels,
        r2=-np.inf,
        slope=np.nan,
        intercept=np.nan,
        residuals=np.array([]),
    )
    x = df["log_inv_eps"].to_numpy()
    y = df["log_mean_N"].to_numpy()
    n = len(df)
    if n < min_levels:
        raise ValueError("Not enough scales to form a linear window")

    for start in range(0, n - min_levels + 1):
        for stop in range(start + min_levels, n + 1):
            xi = x[start:stop]
            yi = y[start:stop]
            A = np.vstack([xi, np.ones_like(xi)]).T
            slope, intercept = np.linalg.lstsq(A, yi, rcond=None)[0]
            predictions = slope * xi + intercept
            residuals = yi - predictions
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((yi - yi.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            resid_std = residuals.std(ddof=1) if residuals.size > 1 else 0.0
            if resid_std > 0 and np.max(np.abs(residuals)) > 2.5 * resid_std:
                continue
            if r2 > best.r2:
                best = LinearWindow(start, stop, r2, slope, intercept, residuals)
    return best


def fit_and_ci(df_band: pd.DataFrame) -> Dict[str, float]:
    """Fit the log-log relationship and compute statistics.

    Args:
        df_band: DataFrame subset for the linear window.

    Returns:
        Dictionary with fit parameters, R2, confidence intervals, and residuals.
    """

    x = df_band["log_inv_eps"].to_numpy()
    y = df_band["log_mean_N"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    predictions = slope * x + intercept
    residuals = y - predictions
    dof = max(len(x) - 2, 1)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se_slope = np.sqrt(cov[0, 0])
    tcrit = stats.t.ppf(0.975, dof)
    ci_low = slope - tcrit * se_slope
    ci_high = slope + tcrit * se_slope
    return {
        "D": slope,
        "intercept": intercept,
        "R2": r2,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "residuals": residuals,
    }


def sensitivity_analysis(
    df: pd.DataFrame,
    window: LinearWindow,
) -> pd.DataFrame:
    """Compute fractal dimension per grid realisation within the chosen window.

    Args:
        df: DataFrame with individual box counts.
        window: The selected LinearWindow.

    Returns:
        DataFrame with fractal dimensions for each grid realization.
    """

    eps_window = (
        df["eps"]
        .drop_duplicates()
        .sort_values(ascending=False)
        .iloc[window.start : window.stop]
        .to_list()
    )
    window_df = df[df["eps"].isin(eps_window)].copy()
    results = []
    for grid_id, group in window_df.groupby("grid_id"):
        group = group.sort_values("eps", ascending=False)
        x = group["log_inv_eps"].to_numpy()
        y = group["log_N"].to_numpy()
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        results.append(
            {
                "grid_id": grid_id,
                "rotation": group["rotation"].iloc[0],
                "offset_x": group["offset_x"].iloc[0],
                "offset_y": group["offset_y"].iloc[0],
                "D": slope,
            }
        )
    return pd.DataFrame(results)


def summarize_sensitivity(
    results: pd.DataFrame, n_bootstrap: int = 1000
) -> pd.DataFrame:
    """Summarise sensitivity of D across grid realisations with bootstrap CI.

    Args:
        results: DataFrame from sensitivity_analysis.
        n_bootstrap: Number of bootstrap samples for CI.

    Returns:
        Summary DataFrame with mean D, std, and bootstrap CI per grid type.

    Raises:
        ValueError: If sensitivity results are empty.
    """

    if results.empty:
        raise ValueError("Sensitivity results are empty")

    grouped = (
        results.groupby(["rotation", "offset_x", "offset_y"])["D"]
        .apply(list)
        .reset_index()
    )
    summary_rows = []
    rng = np.random.default_rng(12345)
    for _, row in grouped.iterrows():
        values = np.array(row["D"])
        boots = []
        for _ in range(n_bootstrap):
            resampled = rng.choice(values, size=len(values), replace=True)
            boots.append(resampled.mean())
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        summary_rows.append(
            {
                "rotation": row["rotation"],
                "offset_x": row["offset_x"],
                "offset_y": row["offset_y"],
                "mean_D": values.mean(),
                "std_D": values.std(ddof=1) if len(values) > 1 else 0.0,
                "CI_low": ci_low,
                "CI_high": ci_high,
            }
        )
    return pd.DataFrame(summary_rows)
