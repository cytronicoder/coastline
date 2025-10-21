### bytheseashore

> she sells seashells by the seashore,
>
> while waves erase the dreams she wore.
>
> -unknown

This package estimates the fractal dimension $D$ of coastline geometries using vector-based box-counting, following Mandelbrot's formulation.

For scale $\varepsilon$, we count the number of boxes $N(\varepsilon)$ intersecting the geometry. Under rotations $\theta$ and offsets $(o_x, o_y)$, we expect $N(\varepsilon) \propto \varepsilon^{-D}$, where $D$ is the fractal dimension.

We then fit $\log N(\varepsilon) = -D \log \varepsilon + c$ via OLS. We select the linear window maximizing $R^2$ to avoid edge effects.

Subsequently, we use a $t$-distribution with $\nu = n-2$ degrees of freedom to compute a 95% confidence interval for the slope $D$: $D \pm t_{0.975} \cdot \mathrm{SE}(D)$. We then bootstrap $D_j$ across grids for stability assessment.

For derivations and implementation details, see [`docs/mathematical_notes.md`](docs/mathematical_notes.md).

To install the package, run:

```bash
pip install -e .
```

Here's a neat script to get you started:

```python
from bytheseashore import run_workflow

results = run_workflow(
    Path("coastline.geojson"),
    eps_list=[5000, 2500, 1250, 625, 312.5],
    offsets=[(0,0), (0.25,0.25)],
    rotations=[0, 15, 30, 45],
)

print(results["table1"])  # D, R², CI
```

The package generates several outputs. It will produce a big table summarizing $D$, $R^2$, confidence intervals, and other diagnostics, along with figures for visual assessment.

You can test with known geometries:

```python
from bytheseashore import generate_straight_line, run_sanity_checks

line = generate_straight_line()
summary = run_sanity_checks(line, [512,256,128], [(0,0)], [0])
# Expect D ≈ 1, R² ≈ 1
```

See [`docs/validation_expectations.md`](docs/validation_expectations.md) for details.

If you want to develop this further, run:

```bash
python -m compileall bytheseashore
```

Unfortunately, contributions are not being accepted at this time. This is my DP Maths AA HL IA project, and contributions would technically violate the academic honesty policy.
