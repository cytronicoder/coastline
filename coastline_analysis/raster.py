"""Rasterisation helper functions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform

GeometryLike = LineString | MultiLineString


__all__ = ["rasterize_to_binary"]


def rasterize_to_binary(
    geometry: GeometryLike, resolution: Tuple[int, int] = (2048, 2048)
) -> np.ndarray:
    """Rasterize geometry to binary array.

    Args:
        geometry: LineString or MultiLineString.
        resolution: (width, height) pixels.

    Returns:
        Binary np.ndarray, 1 where intersects.

    Raises:
        TypeError: If not LineString/MultiLineString.
    """

    if not isinstance(geometry, (LineString, MultiLineString)):
        raise TypeError("Geometry must be a LineString or MultiLineString")

    width, height = resolution
    minx, miny, maxx, maxy = geometry.bounds
    scale_x = width / (maxx - minx) if maxx > minx else 1.0
    scale_y = height / (maxy - miny) if maxy > miny else 1.0

    def normalise(x, y, z=None):  # pragma: no cover - simple wrapper for transform
        _ = z  # unused
        return (x - minx) * scale_x, (maxy - y) * scale_y

    normed = transform(normalise, geometry)
    canvas = np.zeros((height, width), dtype=np.uint8)

    def draw_line(coords):
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            num = max(abs(x1 - x0), abs(y1 - y0))
            if num == 0:
                canvas[int(round(y0)) % height, int(round(x0)) % width] = 1
                continue
            xs = np.linspace(x0, x1, int(num) + 1)
            ys = np.linspace(y0, y1, int(num) + 1)
            canvas[
                np.clip(np.round(ys).astype(int), 0, height - 1),
                np.clip(np.round(xs).astype(int), 0, width - 1),
            ] = 1

    if isinstance(normed, LineString):
        draw_line(np.array(normed.coords))
    else:
        for line in normed.geoms:
            draw_line(np.array(line.coords))

    return canvas
