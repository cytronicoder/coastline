r"""Grid generation utilities for box counting.

The functions here translate mathematical definitions of box-counting lattices
into concrete Shapely geometries. Each grid is fully described by its spacing,
rotation, and offset so that higher-level routines can reproduce the exact box
placements used during analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

from shapely.geometry.base import BaseGeometry


__all__ = ["GridSpec", "make_grids", "count_boxes_vector"]


@dataclass(frozen=True)
class GridSpec:
    r"""Grid spec for box counting.

    Attributes:
        eps: Box size ε.
        rotation: Angle θ in degrees.
        offset: Fractions (o_x, o_y) ∈ [0,1).
        origin: SW corner coords after offset.
        shape: Cells (n_x, n_y).
        bounds: Unrotated extent (minx, miny, maxx, maxy).
    """

    eps: float
    rotation: float
    offset: Tuple[float, float]
    origin: Tuple[float, float]
    shape: Tuple[int, int]
    bounds: Tuple[float, float, float, float]

    def cells(self):
        r"""Yield shapely boxes for each cell in the grid.

        This generator expresses the lattice \(\{x_0 + i\varepsilon\} \times
        \{y_0 + j\varepsilon\}\) as individual axis-aligned boxes prior to any
        rotation. The caller is responsible for applying geometric transformations
        if a rotated frame is required.
        """
        minx, miny = self.origin
        nx, ny = self.shape
        eps = self.eps
        for ix in range(nx):
            x0 = minx + ix * eps
            for iy in range(ny):
                y0 = miny + iy * eps
                yield box(x0, y0, x0 + eps, y0 + eps)


def _grid_origin(
    min_coord: float, max_coord: float, eps: float, offset: float
) -> Tuple[float, int]:
    r"""Calculate the grid origin and number of cells along one axis.

    The origin is obtained by sliding the lattice by ``offset * eps`` and then
    snapping to the nearest multiple of ``eps`` not exceeding ``min_coord``. This
    ensures coverage regardless of floating-point round-off and mirrors the
    theoretical expression \(x_0 = \lfloor (\min x - o_x\varepsilon)/\varepsilon
    \rfloor \varepsilon + o_x\varepsilon\).

    Args:
        min_coord: Minimum coordinate of the extent.
        max_coord: Maximum coordinate of the extent.
        eps: Grid cell size.
        offset: Offset fraction of ``eps``.

    Returns:
        A tuple ``(origin, count)`` where ``count`` equals the number of steps of
        length ``eps`` necessary to reach ``max_coord``.
    """
    start = floor((min_coord - offset * eps) / eps) * eps + offset * eps
    count = int(np.ceil((max_coord - start) / eps)) + 1
    return start, count


def make_grids(
    extent: Sequence[float],
    eps: float,
    offsets: Iterable[Tuple[float, float]],
    rotations: Iterable[float],
) -> List[GridSpec]:
    r"""Create GridSpec for each (θ, o_x, o_y).

    Generates unrotated grids covering extent.

    Args:
        extent: (minx, miny, maxx, maxy).
        eps: Box size ε.
        offsets: (o_x, o_y) fractions.
        rotations: Angles θ in degrees.

    Returns:
        List of GridSpec.
    """

    minx, miny, maxx, maxy = extent
    grids: List[GridSpec] = []
    for rotation in rotations:
        for ox, oy in offsets:
            origin_x, nx = _grid_origin(minx, maxx, eps, ox)
            origin_y, ny = _grid_origin(miny, maxy, eps, oy)
            grids.append(
                GridSpec(
                    eps=eps,
                    rotation=rotation,
                    offset=(ox, oy),
                    origin=(origin_x, origin_y),
                    shape=(nx, ny),
                    bounds=(minx, miny, maxx, maxy),
                )
            )
    return grids


def count_boxes_vector(geometry: BaseGeometry, grid: GridSpec) -> int:
    r"""Count intersected cells N(ε, θ, o_x, o_y).

    Uses prepared geometry for efficient intersects.

    Args:
        geometry: Rotated geometry.
        grid: GridSpec.

    Returns:
        Int N.
    """

    prepared = prep(geometry)
    count = 0
    for cell in grid.cells():
        if prepared.intersects(cell):
            count += 1
    return count
