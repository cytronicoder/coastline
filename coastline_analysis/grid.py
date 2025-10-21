"""Grid generation utilities for box counting."""

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
    """Specification for a square grid used in box counting.

    Attributes:
        eps: Size of each grid cell.
        rotation: Rotation angle of the grid in degrees.
        offset: Tuple of (x_offset, y_offset) as fractions of eps.
        origin: Tuple of (minx, miny) coordinates of the grid origin.
        shape: Tuple of (num_cells_x, num_cells_y) defining grid dimensions.
        bounds: Tuple of (minx, miny, maxx, maxy) defining the grid extent.
    """

    eps: float
    rotation: float
    offset: Tuple[float, float]
    origin: Tuple[float, float]
    shape: Tuple[int, int]
    bounds: Tuple[float, float, float, float]

    def cells(self):
        """Yield shapely boxes for each cell in the grid.

        Yields:
            Shapely box geometries for each grid cell.
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
    """Calculate the grid origin and number of cells along one axis.

    Args:
        min_coord: Minimum coordinate of the extent.
        max_coord: Maximum coordinate of the extent.
        eps: Grid cell size.
        offset: Offset fraction of eps.

    Returns:
        A tuple of (origin coordinate, number of cells).
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
    """Create grid specifications covering an extent for all offsets/rotations.

    Args:
        extent: Bounding box (minx, miny, maxx, maxy).
        eps: Grid cell size.
        offsets: Iterable of (x, y) offset fractions.
        rotations: Iterable of rotation angles in degrees.

    Returns:
        List of GridSpec objects for each combination.
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
    """Count the number of grid cells intersected by a geometry.

    Args:
        geometry: The geometry to check intersections with.
        grid: The GridSpec defining the grid.

    Returns:
        The count of intersected cells.
    """

    prepared = prep(geometry)
    count = 0
    for cell in grid.cells():
        if prepared.intersects(cell):
            count += 1
    return count
