# Coastline Fractal Analysis Toolkit

This package estimates the fractal dimension $D$ of coastline geometries using vector-based box-counting, following Mandelbrot's formulation. It automates diagnostics, validation, and sensitivity analysis for reproducible results.

## Mathematical Foundation

- **Box-Counting Algorithm**: For scale $\varepsilon$, count boxes $N(\varepsilon)$ intersecting the geometry. Under rotations $\theta$ and offsets $(o_x, o_y)$, $N(\varepsilon) \propto \varepsilon^{-D}$, where $D$ is the fractal dimension.
- **Estimation of $D$**: Fit $\log N(\varepsilon) = -D \log \varepsilon + c$ via OLS. Select the linear window maximizing $R^2$ to avoid edge effects.
- **Confidence Intervals**: Use $t$-distribution with $\nu = n-2$ df for slope CI: $D \pm t_{0.975} \cdot \mathrm{SE}(D)$.
- **Sensitivity**: Bootstrap $D_j$ across grids for stability assessment.

For derivations, see [`docs/mathematical_notes.md`](docs/mathematical_notes.md).

## Installation

```bash
pip install -e .
```

Dependencies: `geopandas`, `shapely>=2.0`, `numpy`, `pandas`, `matplotlib`, `scipy`.

## Quick Start

```python
from coastline_analysis import run_workflow

results = run_workflow(
    Path("coastline.geojson"),
    eps_list=[5000, 2500, 1250, 625, 312.5],
    offsets=[(0,0), (0.25,0.25)],
    rotations=[0, 15, 30, 45],
)

print(results["table1"])  # D, R², CI
```

Outputs: figures (map, log-log, sensitivity), tables, and intermediates.

### Output Interpretation

- **Table 1**: $D$, $R^2$, 95% CI, window size, grid count.
- **Figure 2**: Log-log plot with fit; check residuals for linearity.
- **Figure 4**: $D$ variation across grids; wide bands indicate instability.

## Validation

Test with known geometries:

```python
from coastline_analysis import generate_straight_line, run_sanity_checks

line = generate_straight_line()
summary = run_sanity_checks(line, [512,256,128], [(0,0)], [0])
# Expect D ≈ 1, R² ≈ 1
```

See [`docs/validation_expectations.md`](docs/validation_expectations.md) for details.

## Development

```bash
python -m compileall coastline_analysis
```

Contributions welcome!
