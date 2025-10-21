# Validation Case Studies and Expected Metrics

This note complements `examples/run_validation.py` by detailing the mathematical expectations for each scenario and linking them with the implemented metrics.

## Straight line benchmark

- **Geometry.** Generated via `generate_straight_line(length, segments)`.
- **Theoretical behaviour.** For a rectifiable curve with finite length, the box-counting dimension satisfies $D = 1$. Accordingly, counts obey $N(\varepsilon) \propto 1/\varepsilon$.
- **Expected regression output.**
  - Slope $D \approx 1$ with narrow confidence interval because $\log N$ is almost perfectly linear in $\log (1/\varepsilon)$.
  - $R^2$ close to 1; deviations arise solely from grid discretization.
  - Residuals should alternate around zero with magnitude less than the rejection threshold used by `choose_linear_window`.
- **Numerical illustration.** Using scales `[512, 256, ..., 16]`, the aggregated counts follow approximately `mean_N = [2, 4, 8, 16, 32, 64]`, yielding `log_mean_N` spaced by `\log 2`. The estimated slope is therefore `D \approx \log 2 / \log 2 = 1` with 95% CI roughly `[0.99, 1.01]`.

## Fractional noise benchmark

- **Geometry.** `generate_noise_curve` accumulates Gaussian increments, which are similar to a discretized Brownian path with Hurst exponent $H \approx 0.5$.
- **Theoretical behaviour.** Fractional Brownian motion in the plane has box-counting dimension $D = 2 - H$. With $H = 0.5$ we expect $D = 1.5$.
- **Expected regression output.**
  - Slope near 1.4-1.6 depending on finite-sample effects.
  - Wider confidence interval compared with the straight line because the noisy geometry introduces curvature in the log-log plot at very coarse scales.
  - Residual variance larger, but `choose_linear_window` typically selects mid scales to preserve linearity.
- **Sensitivity interpretation.** Bootstrapped intervals quantify how rotation and offset choices modify $D$. Expect a standard deviation of roughly 0.03-0.05 based on empirical trials.

## Simplification study

- **Setup.** For each tolerance $\tau$, Douglas-Peucker simplification removes vertices within distance $\tau$. Smaller $\tau$ retains detail; larger $\tau$ smooths the coastline.
- **Effect on $D$.** Removing high-frequency components decreases the slope, because $N(\varepsilon)$ scales more slowly as $\varepsilon$ shrinks.
- **Diagnostic expectation.** Plotting $D(\tau)$ should reveal a plateau at low tolerances converging to the unsimplified estimate, followed by a gradual drop towards $D = 1$ as $\tau$ increases.

## Residual analysis

- **Log-log plots.** Figure 2 from the workflow overlays the regression line and residuals. Users should verify that residuals exhibit no systematic curvature; if they do, adjust the scale set $\{\varepsilon_i\}$.
- **Outlier handling.** Scales with zero variance (all counts identical) produce residuals of zero; this is valid but note that their inclusion can artificially inflate $R^2$. Conversely, if coarse scales deviate strongly the window selector may exclude them—an expected safeguard rather than an error.
