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
    r"""Represent a linear window in the log–log box-count data.

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
        intercept: Estimated intercept \(b\) in the log–log model.
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
    r"""Count intersected boxes for a geometry at a single scale.

    Each grid realisation is defined by a rotation ``theta`` and offset
    fractions ``(o_x, o_y)``. The grid is rotated about the geometry centroid and
    translated so that a lattice of spacing ``eps`` covers the geometry bounds.
    The number of intersected boxes \(N(\varepsilon, \theta, o_x, o_y)\) is
    recorded for every combination, producing the raw counts used by subsequent
    aggregation.

    Args:
        geometry: Input coastline geometry.
        eps: Grid spacing \(\varepsilon\).
        offsets: Iterable of offset fractions \((o_x, o_y)\) with
            \(0 \le o_x, o_y < 1\).
        rotations: Iterable of rotation angles in degrees.

    Returns:
        A list of dictionaries with one record per grid realisation containing
        the rotation, offsets, and integer count ``N``.
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
    r"""Compute box counts for a set of scales.

    For every \(\varepsilon \in \texttt{eps_list}\) the function calls
    :func:`grid_count_vector` and stores the resulting counts in a tidy
    :class:`pandas.DataFrame`. Two logarithmic helper columns are added:

    * ``log_inv_eps`` = \(\log(1/\varepsilon)\), i.e., the abscissa for the
      regression.
    * ``log_N`` = \(\log N\) with ``N`` clipped below by 1 to avoid taking the
      logarithm of zero when coarse grids miss the geometry entirely.

    Args:
        geometry: Input coastline geometry.
        eps_list: Sequence of box sizes (from coarse to fine).
        offsets: Iterable of offset fractions.
        rotations: Iterable of rotation angles in degrees.

    Returns:
        DataFrame with one row per grid realisation and scale containing the raw
        count and the log-transformed auxiliaries.
    """

    records: List[Dict[str, object]] = []
    for eps in eps_list:
        records.extend(grid_count_vector(geometry, eps, offsets, rotations))
    df = pd.DataFrame.from_records(records)
    df["log_inv_eps"] = np.log(1.0 / df["eps"].values)
    df["log_N"] = np.log(df["N"].clip(lower=1))
    return df


def aggregate_counts(df: pd.DataFrame) -> pd.DataFrame:
    r"""Aggregate box counts across grid realisations for each scale.

    The aggregation step computes summary statistics required by the regression:

    * ``mean_N`` is the sample mean \(\bar{N}(\varepsilon)\) across rotations
      and offsets.
    * ``std_N`` stores the sample standard deviation to diagnose grid
      sensitivity before bootstrapping.
    * ``log_mean_N`` evaluates \(\log \bar{N}(\varepsilon)\) after clipping the
      mean at 1. This keeps the log–log points finite without discarding data
      where only one cell intersects the geometry.

    Args:
        df: DataFrame with individual box-count records as produced by
            :func:`boxcount_series`.

    Returns:
        Aggregated DataFrame with one row per \(\varepsilon\) sorted from coarse
        to fine scales.
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
    r"""Select the most linear contiguous region in the log–log curve.

    Each candidate window ``[start, stop)`` is evaluated by fitting
    \(y = D x + b\) on the subset of ``log_inv_eps`` (``x``) and ``log_mean_N``
    (``y``). The window with maximum \(R^2\) that also satisfies the residual
    outlier test ``max|r_i| <= 2.5 * sigma_r`` is returned. This mirrors common
    practice in fractal dimension estimation where curvature at extreme scales is
    trimmed before fitting (Falconer, 2014).

    Args:
        df: Aggregated DataFrame containing log-transformed columns.
        min_levels: Minimum number of scales \(m\) allowed in a window. Values
            below 4 tend to overfit and are therefore disallowed by default.

    Returns:
        The :class:`LinearWindow` with highest \(R^2\) among admissible windows.

    Raises:
        ValueError: If the dataset has fewer than ``min_levels`` scales.
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
    r"""Fit the log–log relationship and compute inferential statistics.

    Given a window of \(n\) points, the function solves the OLS system
    ``A @ [slope, intercept] = y`` with
    ``A = [[x_i, 1] for x_i in log_inv_eps]``. Residuals \(r_i\) are used to
    compute the sum of squared residuals \(\mathrm{SSR}\) and the coefficient of
    determination \(R^2\). The variance estimate ``sigma2`` equals
    \(\mathrm{SSR} / (n-2)\) and yields the standard error of the slope, from
    which a 95 % confidence interval is derived via the Student
    \(t_{0.975,\,n-2}\) quantile.

    Args:
        df_band: DataFrame subset corresponding to the selected linear window.

    Returns:
        Dictionary containing the slope (``D``), intercept, ``R2``, lower and
        upper confidence limits, and the residual vector.
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
    r"""Compute fractal dimension per grid realisation within the chosen window.

    The regression is repeated for every ``grid_id`` using only the scales in the
    ``window``. This yields slopes \(D_j\) quantifying how grid orientation and
    placement influence the estimated dimension. These slopes underpin the
    bootstrap sensitivity summary and correspond directly to the data visualised
    in Figure 4 of the workflow report.

    Args:
        df: DataFrame with individual box counts (output of
            :func:`boxcount_series`).
        window: The selected :class:`LinearWindow` describing the admissible
            scales.

    Returns:
        DataFrame with one row per grid realisation and the associated slope.
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
    r"""Summarise sensitivity of ``D`` across grid realisations via bootstrapping.

    Within each group of common rotation and offsets the slopes ``D`` are sampled
    with replacement ``n_bootstrap`` times. The bootstrap distribution of the
    mean approximates the sampling distribution of \(\bar{D}\) and its 2.5th and
    97.5th percentiles form a non-parametric 95 % confidence interval. This is a
    direct implementation of Efron's bootstrap applied to the per-grid slopes.

    Args:
        results: DataFrame returned by :func:`sensitivity_analysis`.
        n_bootstrap: Number of bootstrap replicates ``B``.

    Returns:
        Summary DataFrame with the mean slope, standard deviation, and bootstrap
        confidence limits for each (rotation, offset) combination.

    Raises:
        ValueError: If ``results`` is empty.
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
