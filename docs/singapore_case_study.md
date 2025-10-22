# Case study: Singapore coastline fractal analysis

This walkthrough demonstrates how to apply the package to Singapore's shoreline
using the box-counting methodology.

## 1. Data preparation

1. Obtain a high-resolution coastline outline. Suitable sources include the
   Singapore Land Authority's open data portal or Natural Earth. Export the
   geometry as a GeoJSON or shapefile in a projected coordinate reference system
   (CRS) that uses metres (for instance EPSG:3414).
2. If the source is raster imagery, vectorise or trace the coastline so that the
   land–sea boundary is represented by polylines. Simplify colours beforehand to
   avoid ambiguity during tracing.
3. Optionally produce additional polygon layers delineating subregions of
   interest, such as the East Coast Park reclamation belt or mangrove reserves in
   Sungei Buloh.

## 2. Running the workflow

```python
from pathlib import Path

from geopandas import read_file
from shapely.geometry import box

from bytheseashore import run_workflow

coastline_path = Path("data/singapore_coastline.geojson")

# Create subregion polygons highlighting contrasting shorelines.
subregions = {
    "east_coast": box(43300, 28000, 86000, 47000),
    "north_mangroves": read_file("data/north_mangroves.geojson").unary_union,
    "tuas": read_file("data/tuas_breakwaters.geojson").unary_union,
}

results = run_workflow(
    coastline_path,
    eps_list=[5000, 2500, 1250, 625, 312.5, 156.25, 78.125],
    offsets=[(0, 0), (0.25, 0.25), (0.5, 0.5)],
    rotations=[0, 15, 30, 45],
    subregions=subregions,
)
```

The returned dictionary contains the standard top-level results along with a
``subregions`` entry. Each subregion dictionary mirrors the main output and
includes the clipped geometry, box-count data frame, summary tables, and figure
paths. When ``output_dir`` is supplied to ``run_workflow`` each subregion will
receive its own subfolder (``subregion_<name>``) containing the exported plots.

## 3. Inspecting results

Plot ``log(N(\varepsilon))`` against ``log(1/\varepsilon)`` and confirm that the
linear window used for the regression spans multiple scales without curvature.
Check the sensitivity table for each rotation/offset combination; wider error
bars indicate alignment effects that may warrant additional offsets.

Comparing the tables highlights how engineered shorelines (Tuas, Changi) often
produce lower fractal dimensions than natural mangrove fringes. Refer to the
primary literature for benchmark values: Mandelbrot reported \(D \approx 1.25\)
for Britain, whereas smoother coastlines such as South Africa tend to lie closer
to 1.1.

## 4. Robustness checks

* Re-run the workflow with rotated grids or different offset seeds to assess
  stability.
* Exclude the finest scales where digitisation noise or image resolution may
  dominate; the ``subregions`` output is well-suited for experimenting with
  alternative ``eps_list`` configurations.
* Document the CRS, data sources, and preprocessing steps to maintain
  reproducibility for academic submissions.
