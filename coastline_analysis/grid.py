"""Grid generation utilities for box counting."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely.geometry import box
from shapely.prepared import prep

from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class GridSpec:
    """Specification for a square grid used in box counting."""

    eps: float
    rotation: float
    offset: Tuple[float, float]
    origin: Tuple[float, float]
    shape: Tuple[int, int]
    bounds: Tuple[float, float, float, float]

    def cells(self):
        """Yield shapely boxes for each cell in the grid."""
        minx, miny = self.origin
        nx, ny = self.shape
        eps = self.eps
        for ix in range(nx):
            x0 = minx + ix * eps
            for iy in range(ny):
                y0 = miny + iy * eps
                yield box(x0, y0, x0 + eps, y0 + eps)


def _grid_origin(min_coord: float, max_coord: float, eps: float, offset: float) -> Tuple[float, int]:
    start = floor((min_coord - offset * eps) / eps) * eps + offset * eps
    count = int(np.ceil((max_coord - start) / eps)) + 1
    return start, count


def make_grids(extent: Sequence[float], eps: float, offsets: Iterable[Tuple[float, float]], rotations: Iterable[float]) -> List[GridSpec]:
    """Create grid specifications covering an extent for all offsets/rotations."""

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
    """Count the number of grid cells intersected by a geometry."""

    prepared = prep(geometry)
    count = 0
    for cell in grid.cells():
        if prepared.intersects(cell):
            count += 1
    return count


__all__ = ["GridSpec", "make_grids", "count_boxes_vector"]
