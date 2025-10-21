# Coastline Fractal Analysis Toolkit

This repository provides a reproducible Python workflow to estimate the fractal dimension of coastline geometries using a vector-based box-counting approach. The package implements all stages requested in the specification:

* data ingestion from GeoJSON (or any GeoPandas readable dataset),
* vector grid generation with configurable offsets and rotations,
* log–log regression with automated scale-window selection,
* sensitivity analysis with bootstrap confidence intervals,
* validation utilities for sanity checks and simplification studies, and
* publication-ready plots and summary tables.

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

## Validation utilities

Use the validation helpers to verify the workflow:

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

## Figures and tables

The plotting module generates the following deliverables:

1. **Figure 1** – coastline map in SVY21 (or the data CRS).
2. **Figure 2** – log–log box-counting plot with fit line, 95% CI, and residuals.
3. **Figure 4** – sensitivity of fractal dimension across grid offsets and rotations.

Table helpers are returned by `run_workflow`:

* **Table 1** – estimated fractal dimension with CI, coefficient of determination, scale window, and number of grid realisations.
* **Table 2** – (optional) simplification tolerance versus estimated fractal dimension.

## Development

Run a simple import test to ensure the package compiles:

```bash
python -m compileall coastline_analysis
```

Contributions are welcome!
