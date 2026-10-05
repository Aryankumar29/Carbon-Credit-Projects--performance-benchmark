"""Project-plot grid and sampling (VM0047 v1.1, Appendix 1, Step 1). Needs the [geo] extra."""
from __future__ import annotations

import numpy as np


def _metric(gdf):
    return gdf.to_crs(gdf.estimate_utm_crs()) if gdf.crs is not None and gdf.crs.is_geographic else gdf


def make_grid(area, size_m: float = 30.0, min_inside: float = 0.75):
    """Non-overlapping square plots covering `area` (GeoDataFrame); keeps cells with at
    least `min_inside` of their area inside the boundary. Returns a metric-CRS GeoDataFrame
    with plot_id, inside_frac and projected centroid x/y. 30 m (0.09 ha) is the minimum size."""
    import geopandas as gpd
    from shapely import box

    if size_m < 30:
        raise ValueError("plots must be at least 30 x 30 m (0.09 ha)")
    area = _metric(area)
    union = area.geometry.union_all()
    minx, miny, maxx, maxy = union.bounds
    xs = np.arange(minx, maxx + size_m, size_m)
    ys = np.arange(miny, maxy + size_m, size_m)
    cells = gpd.GeoDataFrame(geometry=[box(x, y, x + size_m, y + size_m) for x in xs for y in ys], crs=area.crs)
    cells = cells[cells.intersects(union)].copy()
    cells["inside_frac"] = cells.geometry.intersection(union).area / cells.geometry.area
    cells = cells[cells.inside_frac >= min_inside].reset_index(drop=True)
    cells["plot_id"] = [f"P{i:06d}" for i in range(len(cells))]
    cells["x"], cells["y"] = cells.geometry.centroid.x, cells.geometry.centroid.y
    return cells


def sample_plots(grid, n: int = 30, method: str = "random", seed: int = 0):
    """Random or systematic sample of at least 30 plots."""
    if n < 30:
        raise ValueError("the methodology requires n >= 30 project plots")
    if len(grid) < n:
        raise ValueError(f"grid has only {len(grid)} plots")
    if method == "random":
        return grid.sample(n, random_state=seed).reset_index(drop=True)
    if method == "systematic":
        idx = np.linspace(0, len(grid) - 1, n).round().astype(int)
        return grid.iloc[idx].reset_index(drop=True)
    raise ValueError("method must be 'random' or 'systematic'")
