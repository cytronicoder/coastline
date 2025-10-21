# Mathematical Notes for Coastline Fractal Analysis

This document summarizes the mathematical foundations implemented in the
`coastline_analysis` package. Equations follow the notation used in Mandelbrot's
treatment of fractal coastlines and in common box-counting literature.

## 1. Box-counting algorithm

* **Key idea.** Cover the geometry \(\Gamma\) with a lattice of axis-aligned
  squares (boxes) whose edge length is \(\varepsilon\). Count the number of
  boxes \(N(\varepsilon)\) intersected by the geometry.
* **Grid generation.** For each scale we rotate the lattice by an angle
  \(\theta\) and translate it by offsets \((o_x, o_y)\) expressed as fractions
  of \(\varepsilon\). The grid origin \((x_0, y_0)\) is computed as
  \[
  x_0 = \bigl\lfloor \frac{\min x - o_x\varepsilon}{\varepsilon} \bigr\rfloor \varepsilon + o_x\varepsilon,
  \quad
  n_x = \left\lceil \frac{\max x - x_0}{\varepsilon} \right\rceil + 1
  \]
  with an analogous expression for \(y\). This ensures the rotated and offset
  grid fully covers the bounding box of \(\Gamma\).
* **Intersection test.** Each grid cell is represented by a Shapely polygon, and
  intersections are evaluated using prepared geometries for efficiency. The
  resulting count is an integer \(N(\varepsilon, \theta, o_x, o_y)\).
* **Clipping rule.** To avoid \(\log 0\), counts are clipped below by 1 prior
  to logarithmic transformation. This matches the implementation in
  `boxcount_series` where `df["N"].clip(lower=1)` is used.

## 2. Fractal dimension estimation

* **Box-counting dimension.** The theoretical fractal (Hausdorff) dimension is
  approximated by the slope of the line in the log–log space:
  \[
  D \approx -\lim_{\varepsilon \to 0} \frac{\log N(\varepsilon)}{\log \varepsilon}.
  \]
  In discrete settings we regress the empirical values
  \((x_i, y_i) = (\log(1/\varepsilon_i), \log \bar{N}(\varepsilon_i))\) where
  \(\bar{N}\) denotes the mean count across grid realizations.
* **Least-squares fit.** The linear model \(y_i = D x_i + b + \varepsilon_i\) is
  solved via ordinary least squares (OLS) by minimizing
  \(\sum_i (y_i - Dx_i - b)^2\). Implementation uses
  `np.linalg.lstsq(A, y)` with \(A = [x_i, 1]\), confirming a standard OLS fit.
* **Coefficient of determination.** The goodness of fit is quantified by
  \[
  R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2},
  \]
  where \(\hat{y}_i = Dx_i + b\) and \(\bar{y}\) is the sample mean. This is
  computed both when choosing the linear window and when reporting final
  statistics.

### 2.1 Linear window selection

* **Objective.** Identify contiguous scales where the log–log relation is most
  linear while satisfying a minimum window size \(m\).
* **Search strategy.** Every candidate window \([s, t)\) with \(t - s \ge m\) is
  fit via OLS. The selected window maximizes \(R^2\) subject to a residual
  outlier rejection step: residual standard deviation \(\sigma_r\) is computed
  and windows with \(\max |r_i| > 2.5\sigma_r\) are discarded.
* **Edge cases.** If the dataset contains fewer than \(m\) scales a `ValueError`
  is raised. Residual variance of zero causes \(R^2 = 0\) to avoid division by
  zero, reflecting flat data.

### 2.2 Confidence intervals

* **Parameter covariance.** The OLS estimator has covariance matrix
  \(\sigma^2 (A^\top A)^{-1}\) with \(\sigma^2 = \mathrm{SSR} / (n-2)\).
* **95% interval for \(D\).** Using the Student \(t\)-distribution with
  \(\nu = n-2\) degrees of freedom, the confidence interval is
  \[
  D \pm t_{0.975,\,\nu} \cdot \mathrm{SE}(D),
  \]
  where \(\mathrm{SE}(D) = \sqrt{(A^\top A)^{-1}_{00}}\).
  This matches `fit_and_ci`, which computes `tcrit = stats.t.ppf(0.975, dof)`.
* **Interpretation.** The interval measures uncertainty from sampling different
  scales; it does not capture geometric variability across grid realizations.

## 3. Sensitivity analysis

* **Per-grid slopes.** For each combination of rotation and offset, the slope of
  the log–log counts is recomputed within the chosen linear window.
* **Bootstrap confidence bands.** Let \(D_j\) be the slope for grid realisation
  \(j\) within a cell of constant rotation/offset. The bootstrap procedure draws
  \(B\) resamples with replacement and records the mean. The 2.5th and 97.5th
  percentiles provide a non-parametric confidence interval on the mean
  \(\mathbb{E}[D_j]\), addressing grid-to-grid sensitivity.
* **Random seed.** The generator is seeded (`rng = np.random.default_rng(12345)`)
  to make reported bootstrap intervals reproducible.

## 4. Validation geometries

* **Straight line.** A polyline of zero curvature should have a theoretical
  fractal dimension \(D = 1\). The observed estimates validate that the method
  does not spuriously inflate the dimension when the geometry is Euclidean.
* **Noisy curve.** The cumulative-sum noise curve emulates a fractional Brownian
  coastline with expected \(D \approx 1.5\) (cf. Mandelbrot 1967). The workflow
  recovers a slope in this range, providing a qualitative check of correctness.

## 5. Practical considerations

* **Scale selection.** Too narrow or wide a scale range can bias the slope. The
  automated window search is still sensitive to resolution choices; users should
  inspect the log–log plot for curvature.
* **Rotations and offsets.** Rotating the grid mitigates alignment artefacts.
  For anisotropic coastlines, include multiple angles (e.g., every 15°).
* **Handling zeros.** If a scale yields no intersected boxes, the count is
  clipped to 1 before taking logs. This corresponds to assuming a minimum single
  box coverage rather than discarding the scale; users may prefer to omit such
  scales entirely.
* **Geometric simplification.** Douglas–Peucker smoothing changes small-scale
  structure and therefore the estimated \(D\). The `simplification_study`
  routine quantifies this effect across tolerances.

## References

1. Benoit B. Mandelbrot, *How Long Is the Coast of Britain? Statistical
   Self-Similarity and Fractional Dimension*, Science 156(3775), 1967.
2. Kenneth Falconer, *Fractal Geometry: Mathematical Foundations and
   Applications*, Wiley, 2014.
