"""Turn a scattered GPS track into a closed boundary and preview it. Needs the [utils] extra.

Coordinates are projected to the correct UTM zone for the data (the original notebooks
hardcoded UTM 33N and swapped lat/lon, which distorted shapes outside Europe).
"""
from __future__ import annotations

import numpy as np


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def to_meters(lons, lats):
    """WGS84 lon/lat -> metres in the UTM zone of the data's centroid. Returns (x, y, epsg)."""
    from pyproj import Transformer

    lons, lats = np.asarray(lons, float), np.asarray(lats, float)
    epsg = utm_epsg(float(lons.mean()), float(lats.mean()))
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lons, lats)
    return np.asarray(x), np.asarray(y), epsg


def connect_first_last(x, y):
    return np.append(x, x[0]), np.append(y, y[0])


def convex_hull(x, y):
    from scipy.spatial import ConvexHull

    pts = np.column_stack((x, y))
    v = ConvexHull(pts).vertices
    return np.append(pts[v, 0], pts[v[0], 0]), np.append(pts[v, 1], pts[v[0], 1])


def alpha_shape(x, y, max_radius_m: float | None = None):
    """Concave boundary: union of Delaunay triangles whose circumradius < max_radius_m.

    Default radius is 2.5 x the median triangle edge length. Falls back to the convex hull
    if nothing survives. If several pieces remain, the largest is returned.
    """
    from scipy.spatial import Delaunay
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    pts = np.column_stack((x, y))
    tri = Delaunay(pts)
    a, b, c = (pts[tri.simplices[:, i]] for i in range(3))
    la, lb, lc = (np.linalg.norm(u - v, axis=1) for u, v in ((b, c), (c, a), (a, b)))
    s = (la + lb + lc) / 2
    area = np.sqrt(np.clip(s * (s - la) * (s - lb) * (s - lc), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        circum = la * lb * lc / (4 * area)
    if max_radius_m is None:
        max_radius_m = 2.5 * float(np.median(np.concatenate([la, lb, lc])))
    keep = np.isfinite(circum) & (circum < max_radius_m)
    if not keep.any():
        return convex_hull(x, y)
    merged = unary_union([Polygon(t) for t in pts[tri.simplices[keep]]])
    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)
    bx, by = merged.exterior.xy
    return np.asarray(bx), np.asarray(by)


def smooth_closed(x, y, points_per_vertex: int = 3):
    """Periodic cubic-spline smoothing of the closed track."""
    from scipy.interpolate import CubicSpline

    xc, yc = connect_first_last(x, y)
    t = np.linspace(0, 1, len(xc))
    ti = np.linspace(0, 1, len(x) * points_per_vertex)
    return CubicSpline(t, xc, bc_type="periodic")(ti), CubicSpline(t, yc, bc_type="periodic")(ti)


METHODS = {
    "connect": connect_first_last,
    "convex_hull": convex_hull,
    "alpha_shape": alpha_shape,
    "spline": smooth_closed,
}


def polygon_area_m2(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def plot_methods(points, out_png: str, methods=METHODS):
    """Save one figure with the track and each closing method (areas in the titles)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lons, lats = zip(*points)
    x, y, epsg = to_meters(lons, lats)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 4.5), squeeze=False)
    for ax, (name, fn) in zip(axes[0], methods.items()):
        mx, my = fn(x, y)
        ax.plot(x, y, "bo", ms=3)
        ax.fill(mx, my, alpha=0.3)
        ax.set_title(f"{name}\n{polygon_area_m2(mx, my) / 1e4:.3f} ha")
        ax.set_aspect("equal")
        ax.set_xlabel(f"x (m, EPSG:{epsg})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main(argv=None):
    import argparse

    from .gps_to_kml import dedupe_consecutive, read_points

    ap = argparse.ArgumentParser(description="Preview GPS track closing methods as a PNG")
    ap.add_argument("input")
    ap.add_argument("output_png")
    a = ap.parse_args(argv)
    plot_methods(dedupe_consecutive(read_points(a.input)), a.output_png)
    print("wrote", a.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
