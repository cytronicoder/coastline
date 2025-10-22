"""Input helpers for coastline fractal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import geopandas as gpd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    LinearRing,
    MultiLineString,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

GeometryLike = Union[LineString, MultiLineString]


__all__ = ["load_coastline", "clip_coastline"]


def _collect_lines(geometry: BaseGeometry) -> List[LineString]:
    """Collect line components from a geometry."""

    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, LinearRing):
        return [LineString(geometry)]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        lines: List[LineString] = []
        for geom in geometry.geoms:
            lines.extend(_collect_lines(geom))
        return lines
    if hasattr(geometry, "boundary"):
        return _collect_lines(geometry.boundary)
    return []


def _to_lines(geometry: BaseGeometry) -> GeometryLike:
    """Extract boundary lines from geometry for box-counting.

    For fractal dimension estimation, we need 1D boundaries. Polygons are converted to their exterior rings.

    Args:
        geometry: Input geometry (polygon, etc.).

    Returns:
        Boundary as LineString or MultiLineString.

    Raises:
        ValueError: If no line geometry extracted.
        TypeError: If unsupported type.
    """
    lines = _collect_lines(geometry)
    if not lines:
        raise ValueError("No line geometry could be extracted from the boundary")
    if len(lines) == 1:
        return lines[0]
    return unary_union(lines)


def load_coastline(filepath: Union[str, Path]) -> GeometryLike:
    """Load and merge coastline geometries from vector file.

    Reads GeoPandas-compatible file, unions geometries, extracts boundaries for box-counting.

    Args:
        filepath: Path to vector file (e.g., GeoJSON).

    Returns:
        Merged boundary lines as LineString or MultiLineString.

    Raises:
        ValueError: If no geometries in file.
    """

    gdf = gpd.read_file(filepath)
    if gdf.empty:
        raise ValueError("The provided file contains no geometries")

    geometry = unary_union(gdf.geometry)
    if isinstance(geometry, GeometryCollection):
        lines = [
            geom
            for geom in geometry.geoms
            if isinstance(geom, (LineString, MultiLineString, LinearRing))
        ]
        if not lines:
            boundaries = [
                geom.boundary for geom in geometry.geoms if hasattr(geom, "boundary")
            ]
            geometry = unary_union(boundaries)
        else:
            geometry = unary_union(lines)

    return _to_lines(geometry)


def clip_coastline(coastline: GeometryLike, region: BaseGeometry) -> GeometryLike:
    """Clip a coastline geometry to a region of interest.

    Args:
        coastline: LineString or MultiLineString representing the coastline.
        region: Geometry describing the region of interest (e.g., polygon).

    Returns:
        The portion of the coastline within ``region``.

    Raises:
        ValueError: If the intersection does not contain any line components.
    """

    clipped = coastline.intersection(region)
    return _to_lines(clipped)
