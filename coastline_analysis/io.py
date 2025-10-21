"""Input helpers for coastline fractal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import unary_union

GeometryLike = Union[LineString, MultiLineString]


__all__ = ["load_coastline"]


def _to_lines(geometry) -> GeometryLike:
    """Convert any polygonal geometry into its boundary lines.

    Args:
        geometry: The input geometry to convert.

    Returns:
        The boundary lines as LineString or MultiLineString.

    Raises:
        ValueError: If no line geometry can be extracted.
        TypeError: If the geometry type is unsupported.
    """
    if isinstance(geometry, (LineString, MultiLineString)):
        return geometry
    if hasattr(geometry, "boundary"):
        boundary = geometry.boundary
        if isinstance(boundary, (LineString, MultiLineString)):
            return boundary
        if isinstance(boundary, GeometryCollection):
            lines = [geom for geom in boundary.geoms if isinstance(geom, LineString)]
            if not lines:
                raise ValueError(
                    "No line geometry could be extracted from the boundary"
                )
            return MultiLineString(lines)
    raise TypeError(f"Unsupported geometry type: {geometry.geom_type}")


def load_coastline(filepath: Union[str, Path]) -> GeometryLike:
    """Load a coastline from a GeoJSON or vector dataset.

    Args:
        filepath: Path to the GeoJSON file. Any format readable by GeoPandas works.

    Returns:
        The merged coastline geometry in the dataset's CRS as LineString or MultiLineString.

    Raises:
        ValueError: If the provided file contains no geometries.
    """

    gdf = gpd.read_file(filepath)
    if gdf.empty:
        raise ValueError("The provided file contains no geometries")

    geometry = unary_union(gdf.geometry)
    if isinstance(geometry, GeometryCollection):
        lines = [
            geom
            for geom in geometry.geoms
            if isinstance(geom, (LineString, MultiLineString))
        ]
        if not lines:
            lines = [
                geom.boundary for geom in geometry.geoms if hasattr(geom, "boundary")
            ]
        geometry = unary_union(lines)

    return _to_lines(geometry)
