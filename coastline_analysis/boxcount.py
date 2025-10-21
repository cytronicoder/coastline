r"""Vector-based box-counting implementations.

The routines in this module implement the discrete box-counting algorithm for
planar geometries. They follow the methodology described in the accompanying
documentation and expose intermediate values such as log-transformed counts,
linear regression residuals, and bootstrap sensitivities.
"""

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
    r"""Represent a linear window in the log-log box-count data.

    The discrete scales \(\varepsilon_i\) are sorted from coarse to fine. A
    window selects a contiguous subset `start <= i < stop` over which
    \(\log N(\varepsilon_i)\) is well approximated by a line in
    \(\log(1/\varepsilon_i)\). The slope and intercept correspond to the
    parameters of the least-squares fit \(y = D x + b\) and the coefficient of
    determination ``r2`` quantifies goodness of fit.

    Attributes:
        start: Index of the first scale in the window (inclusive).
        stop: Index of the last scale (exclusive).
        r2: Coefficient of determination \(R^2\) for the linear fit.
        slope: Estimated fractal dimension \(D\).
        intercept: Estimated intercept \(b\) in the log-log model.
        residuals: Residual vector \(r_i = y_i - (Dx_i + b)\).
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
    r"""Count boxes N(ε) at scale ε across rotated/offset grids.

    For each θ, (o_x, o_y), rotate geometry by -θ, generate grid, count intersections N(ε, θ, o_x, o_y).

    Args:
        geometry: Input geometry.
        eps: Box size ε.
        offsets: Offset fractions (o_x, o_y) ∈ [0,1).
        rotations: Angles θ in degrees.

    Returns:
        List of dicts with ε, θ, o_x, o_y, grid_id, N.
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
    r"""Compute box counts N(ε) for multiple scales ε.

    Calls grid_count_vector for each ε, adds log columns: log_inv_eps = log(1/ε), log_N = log(max(N,1)).

    Args:
        geometry: Input geometry.
        eps_list: Box sizes ε (coarse to fine).
        offsets: Offset fractions (o_x, o_y).
        rotations: Angles θ in degrees.

    Returns:
        DataFrame with rows per grid/scale: eps, θ, o_x, o_y, N, log_inv_eps, log_N.
    """

    records: List[Dict[str, object]] = []
    for eps in eps_list:
        records.extend(grid_count_vector(geometry, eps, offsets, rotations))
    df = pd.DataFrame.from_records(records)
    df["log_inv_eps"] = np.log(1.0 / df["eps"].values)
    df["log_N"] = np.log(df["N"].clip(lower=1))
    return df


def aggregate_counts(df: pd.DataFrame) -> pd.DataFrame:
    r"""Aggregate counts across grids for each ε.

    Computes mean_N = mean(N), std_N = std(N), log_mean_N = log(max(mean_N,1)).

    Args:
        df: DataFrame from boxcount_series.

    Returns:
        DataFrame with one row per ε: eps, log_inv_eps, mean_N, std_N, n_grids, log_mean_N.
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
    r"""Select linear window in log-log plot with max R².

    Fits y = D x + b on subsets, chooses window with highest R² where max|r_i| ≤ 2.5 σ_r.

    Args:
        df: Aggregated DataFrame.
        min_levels: Min scales in window (default 4).

    Returns:
        LinearWindow with best fit.

    Raises:
        ValueError: If < min_levels scales.
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
    r"""Fit log-log: y = D x + b, compute CI.

    OLS fit, R², 95% CI via t-distribution.

    Args:
        df_band: DataFrame for linear window.

    Returns:
        Dict: D, intercept, R2, CI_low, CI_high, residuals.
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
    r"""Compute D per grid in window.

    Fits log-log per grid_id in window, returns slopes D_j.

    Args:
        df: DataFrame from boxcount_series.
        window: LinearWindow.

    Returns:
        DataFrame: grid_id, rotation, offset_x, offset_y, D.
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
        slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
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
    r"""Bootstrap CI for D per (θ, o_x, o_y).

    Samples D_j with replacement, computes mean, std, 95% CI.

    Args:
        results: DataFrame from sensitivity_analysis.
        n_bootstrap: Bootstrap samples (default 1000).

    Returns:
        DataFrame: rotation, offset_x, offset_y, mean_D, std_D, CI_low, CI_high.

    Raises:
        ValueError: If results empty.
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
