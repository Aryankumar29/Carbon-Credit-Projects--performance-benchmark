"""Batch KML -> Shapefile conversion with geometry repair. Needs the [utils] extra.

    python -m vm0047_pb.utils.kml_to_shp INPUT_DIR OUTPUT_DIR [--bbox-fallback]
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import linemerge, polygonize, unary_union

log = logging.getLogger("kml_to_shp")


def _bbox_polygon(geom):
    try:
        minx, miny, maxx, maxy = geom.bounds
        return Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    except Exception:
        return None


def _close_ring(coords):
    coords = [c[:2] for c in coords]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def fix_geometry(geom, bbox_fallback: bool = False, point_buffer: float = 0.0001):
    """Return a valid Polygon/MultiPolygon, or None if the geometry cannot be repaired.

    LineStrings are closed into rings, MultiLineStrings merged/polygonized, invalid polygons
    repaired with buffer(0). Points become tiny buffers (point_buffer, in CRS units).
    With bbox_fallback=True unrecoverable geometries are replaced by their bounding box;
    this changes the boundary and is OFF by default.
    """
    if geom is None or geom.is_empty:
        return None
    fallback = (lambda g: _bbox_polygon(g)) if bbox_fallback else (lambda g: None)
    try:
        t = geom.geom_type
        if t in ("Polygon", "MultiPolygon"):
            if geom.is_valid:
                return geom
            fixed = geom.buffer(0)
            if fixed.is_valid and not fixed.is_empty:
                return fixed
            return fallback(geom)
        if t == "Point":
            return geom.buffer(point_buffer)
        if t == "LineString":
            if len(geom.coords) < 3:
                return fallback(geom)
            return Polygon(_close_ring(geom.coords)).buffer(0)
        if t == "MultiLineString":
            merged = linemerge(geom)
            if merged.geom_type == "LineString" and len(merged.coords) >= 3:
                return Polygon(_close_ring(merged.coords)).buffer(0)
            polys = list(polygonize(geom))
            if polys:
                return polys[0] if len(polys) == 1 else MultiPolygon(polys)
            return fallback(geom)
        if t == "GeometryCollection":
            parts = [fix_geometry(g, bbox_fallback, point_buffer) for g in geom.geoms]
            parts = [p for p in parts if p is not None]
            return unary_union(parts) if parts else fallback(geom)
        log.warning("unsupported geometry type %s", t)
        return fallback(geom)
    except Exception as exc:
        log.error("error fixing geometry: %s", exc)
        return None


def truncate_column_names(gdf):
    """Shapefile field names are limited to 10 characters; keep them unique."""
    mapping: dict[str, str] = {}
    taken = {c for c in gdf.columns if len(c) <= 10}
    for col in gdf.columns:
        if len(col) > 10:
            new, i = col[:10], 1
            while new in taken:
                new = f"{col[:8]}{i:02d}"
                i += 1
            taken.add(new)
            mapping[col] = new
    if mapping:
        log.info("truncated column names: %s", mapping)
    return gdf.rename(columns=mapping)


def read_kml(path) -> gpd.GeoDataFrame:
    """Read every layer (KML folder) of a KML file into one GeoDataFrame."""
    try:
        import pyogrio

        layers = [name for name, _ in pyogrio.list_layers(path)]
        frames = [gpd.read_file(path, layer=n) for n in layers]
        frames = [f for f in frames if not f.empty]
        if frames:
            return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    except Exception as exc:
        log.debug("layer-wise read failed (%s); trying default read", exc)
    return gpd.read_file(path)


def kml_to_shapefile(kml_path, output_path, bbox_fallback: bool = False):
    """Convert one KML to a polygon Shapefile. Returns the output path, or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gdf = read_kml(kml_path)
        if gdf.empty:
            log.warning("empty KML: %s", kml_path)
            return None
        crs = gdf.crs
        gdf["geometry"] = gdf.geometry.apply(lambda g: fix_geometry(g, bbox_fallback))
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf["geometry"] = gdf.geometry.make_valid()
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]) & gdf.geometry.is_valid]
        if gdf.empty:
            log.warning("no valid polygons in %s", kml_path)
            return None
        gdf = truncate_column_names(gdf.set_crs(crs, allow_override=True) if crs else gdf)
        gdf.to_file(output_path)
        return output_path
    except Exception as exc:
        log.error("error processing %s: %s", kml_path, exc)
        return None


def convert_kml_folder(kml_folder, output_folder, bbox_fallback: bool = False):
    """Convert every *.kml in a folder. Returns (n_ok, [failed file names])."""
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(kml_folder).glob("*.kml"))
    if not files:
        log.warning("no KML files found in %s", kml_folder)
        return 0, []
    try:
        from tqdm import tqdm

        files_iter = tqdm(files, desc="Converting KML files")
    except ImportError:
        files_iter = files
    ok, failed = 0, []
    for f in files_iter:
        if kml_to_shapefile(f, out / f"{f.stem}.shp", bbox_fallback):
            ok += 1
        else:
            failed.append(f.name)
    log.info("conversion complete: %d ok, %d failed", ok, len(failed))
    return ok, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--bbox-fallback", action="store_true",
                    help="replace unrepairable geometries with their bounding box (alters boundaries)")
    ap.add_argument("--log-file", default=None)
    a = ap.parse_args(argv)
    handlers = [logging.StreamHandler()] + ([logging.FileHandler(a.log_file)] if a.log_file else [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers)
    ok, failed = convert_kml_folder(a.input_dir, a.output_dir, a.bbox_fallback)
    for name in failed:
        print("FAILED:", name)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
