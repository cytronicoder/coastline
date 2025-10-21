# Coastline Fractal Analysis Toolkit

This repository provides a reproducible Python workflow to estimate the fractal dimension of coastline geometries using a vector-based box-counting approach. The implementation follows the classic formulation from Mandelbrot's "How Long Is the Coast of Britain?" and augments it with automated diagnostics and validation utilities.

## Mathematical overview

- **Box-counting.** For each box size $\varepsilon$ we count the number of grid cells $N(\varepsilon, \theta, o_x, o_y)$ intersected by the coastline under multiple rotations $\theta$ and offsets $(o_x, o_y)$. The mean count $\bar{N}(\varepsilon)$ approximates the scaling law $N(\varepsilon) \propto \varepsilon^{-D}$.
- **Linear regression.** The fractal dimension $D$ is estimated by fitting the line $\log \bar{N}(\varepsilon) = D \log(1/\varepsilon) + b$ via ordinary least squares. Confidence intervals use the $t$-distribution with $\nu = n-2$ degrees of freedom.
- **Window selection.** A contiguous set of scales maximizing $R^2$ while rejecting residual outliers identifies the most linear portion of the log–log curve. This mitigates curvature at extreme scales.
- **Sensitivity analysis.** Grid-specific slopes $D_j$ are re-estimated within the chosen window. Bootstrap resampling (default 1000 draws) of these slopes yields a non-parametric confidence band for the mean fractal dimension.

Comprehensive derivations and worked examples are provided in [`docs/mathematical_notes.md`](docs/mathematical_notes.md) and [`docs/validation_expectations.md`](docs/validation_expectations.md).

## Installation

```bash
pip install -e .
```

The package depends on `geopandas`, `shapely>=2.0`, `numpy`, `pandas`, `matplotlib`, and `scipy`.

## Quick start

```python
from pathlib import Path

from coastline_analysis import run_workflow

coastline_path = Path("data/singapore_coast.geojson")
resolution_levels = [5000, 2500, 1250, 625, 312.5, 156.25]
offsets = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
rotations = [0.0, 15.0, 30.0, 45.0, 60.0]

results = run_workflow(
    coastline_path,
    eps_list=resolution_levels,
    offsets=offsets,
    rotations=rotations,
    simplification_tolerances=[5, 25, 100],
)

print(results["table1"])
```

The call produces figures, summary tables, and intermediate results suitable for reporting.

### Interpreting the output

1. **Regression summary.** `table1` contains $D$, $R^2$, the linear window indices, and the 95% confidence interval $D \pm t_{0.975,\,\nu}\mathrm{SE}(D)$.
2. **Log–log plot.** Figure 2 overlays the fitted line and residuals. Inspect the residual subplot for curvature; if present, adjust the scale list.
3. **Sensitivity grid.** Figure 4 reports the bootstrap mean and interval of $D_j$ across offsets and rotations, highlighting numerical stability.

## Validation utilities

Use the validation helpers to verify the workflow against geometries with known behaviour:

```python
from coastline_analysis import (
    generate_straight_line,
    generate_noise_curve,
    run_sanity_checks,
)

line = generate_straight_line()
noise_curve = generate_noise_curve()
summary_line = run_sanity_checks(line, resolution_levels, offsets, rotations)
summary_noise = run_sanity_checks(noise_curve, resolution_levels, offsets, rotations)
```

- For the straight line benchmark expect $D \approx 1$ and $R^2$ close to 1.
- For the noise surrogate expect $D$ in the 1.4–1.6 range with wider confidence intervals, as described in [`docs/validation_expectations.md`](docs/validation_expectations.md).

## Figures and tables

The plotting module generates the following deliverables:

1. **Figure 1** – coastline map in SVY21 (or the data CRS).
2. **Figure 2** – log–log box-counting plot with fit line, 95% CI, and residuals.
3. **Figure 4** – sensitivity of fractal dimension across grid offsets and rotations.

Table helpers are returned by `run_workflow`:

- **Table 1** – estimated fractal dimension with CI, coefficient of determination, scale window, and number of grid realisations.
- **Table 2** – (optional) simplification tolerance versus estimated fractal dimension.

## Development

Run a simple import test to ensure the package compiles:

```bash
python -m compileall coastline_analysis
```

Contributions are welcome!
