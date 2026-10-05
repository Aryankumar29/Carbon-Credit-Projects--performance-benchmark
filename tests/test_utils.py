import zipfile

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("simplekml")
pytest.importorskip("pyproj")
pytest.importorskip("matplotlib")
import geopandas as gpd
import simplekml
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from vm0047_pb.utils import gps_shapes as gs
from vm0047_pb.utils import gps_to_kml as g2k
from vm0047_pb.utils import kml_to_shp as k2s

LAT0, LON0 = 26.3068, 89.3516  # synthetic location, rectangle ~100 m x 50 m


def _track(n_side=10):
    """Rectangle outline sampled like a walked GPS track, in lon/lat."""
    dlat, dlon = 50 / 111_000, 100 / (111_000 * np.cos(np.radians(LAT0)))
    t = np.linspace(0, 1, n_side, endpoint=False)
    ring = (
        [(LON0 + dlon * s, LAT0) for s in t]
        + [(LON0 + dlon, LAT0 + dlat * s) for s in t]
        + [(LON0 + dlon * (1 - s), LAT0 + dlat) for s in t]
        + [(LON0, LAT0 + dlat * (1 - s)) for s in t]
    )
    return ring


def _write_txt(path, pts, dup_first=False):
    lines = ["type,date time,latitude,longitude,accuracy(m),name,desc"]
    for lon, lat in ([pts[0]] if dup_first else []) + pts:
        lines.append(f"T,2024-01-01 00:00:00,{lat:.8f},{lon:.8f},3.5,,")
    lines.append("T,2024-01-01 00:00:01,,,3.5,,")  # blank row must be skipped
    path.write_text("\n".join(lines), encoding="utf-8")


def test_read_points_skips_blank_and_validates(tmp_path):
    f = tmp_path / "a.txt"
    _write_txt(f, _track())
    assert len(g2k.read_points(f)) == 40
    bad = tmp_path / "bad.txt"
    bad.write_text("latitude,longitude\n95,10\n")
    with pytest.raises(ValueError):
        g2k.read_points(bad)
    with pytest.raises(ValueError):
        g2k.read_points(b"foo,bar\n1,2\n")


def test_txt_to_kml_closed_polygon_roundtrip(tmp_path):
    f = tmp_path / "plot1.txt"
    _write_txt(f, _track(), dup_first=True)
    out = tmp_path / "plot1.kml"
    g2k.txt_to_kml(f, out)
    gdf = k2s.read_kml(out)
    assert len(gdf) == 1 and gdf.geometry.iloc[0].geom_type == "Polygon"
    x, y, _ = gs.to_meters(*zip(*_track()))
    assert gdf.to_crs(gdf.estimate_utm_crs()).area.iloc[0] == pytest.approx(gs.polygon_area_m2(x, y), rel=0.02)


def test_batch_zip_isolates_bad_file(tmp_path):
    src = tmp_path / "in.zip"
    with zipfile.ZipFile(src, "w") as z:
        good = tmp_path / "g.txt"
        _write_txt(good, _track())
        z.write(good, "good.txt")
        z.writestr("broken.txt", "latitude,longitude\n1,1\n2,2\n")  # < 3 points
    ok, errors = g2k.batch_to_kml(src, tmp_path / "out.zip")
    assert ok == 1 and list(errors) == ["broken.txt"]
    assert zipfile.ZipFile(tmp_path / "out.zip").namelist() == ["good.kml"]


def test_utm_zone_and_lat_lon_order():
    assert gs.utm_epsg(LON0, LAT0) == 32645
    assert gs.utm_epsg(-63, -10) == 32720
    x, y, epsg = gs.to_meters([LON0, LON0 + 0.001], [LAT0, LAT0])
    assert epsg == 32645 and abs((x[1] - x[0]) - 100) < 2 and abs(y[1] - y[0]) < 5  # east-west is x (small dy = UTM grid convergence)


def test_shape_methods_on_concave_track():
    # L-shaped boundary (area 3 units^2 of a 2x2 square); convex hull overestimates it
    ring = np.array([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)], float) * 50
    pts = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        pts += [a + (b - a) * s for s in np.linspace(0, 1, 8, endpoint=False)]
    x, y = np.array(pts).T
    hull = gs.polygon_area_m2(*gs.convex_hull(x, y))
    alpha = gs.polygon_area_m2(*gs.alpha_shape(x, y, max_radius_m=40))
    assert hull == pytest.approx(3.5 * 2500, rel=0.01)
    assert alpha < hull
    assert gs.polygon_area_m2(*gs.connect_first_last(x, y)) == pytest.approx(3 * 2500, rel=0.01)


def test_plot_methods_writes_png(tmp_path):
    out = gs.plot_methods(_track(), str(tmp_path / "m.png"))
    assert (tmp_path / "m.png").stat().st_size > 1000 and out.endswith("m.png")


def test_fix_geometry_cases():
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    fixed = k2s.fix_geometry(bowtie)
    assert fixed.is_valid and fixed.geom_type in ("Polygon", "MultiPolygon")
    ring = k2s.fix_geometry(LineString([(0, 0), (1, 0), (1, 1), (0, 1)]))
    assert ring.geom_type == "Polygon" and ring.area == pytest.approx(1)
    segs = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 0)]])
    assert k2s.fix_geometry(segs).geom_type == "Polygon"
    assert k2s.fix_geometry(Point(0, 0)).area > 0
    assert k2s.fix_geometry(LineString([(0, 0), (1, 1)])) is None
    assert k2s.fix_geometry(LineString([(0, 0), (1, 1)]), bbox_fallback=True).area == pytest.approx(1.0)


def test_truncate_column_names_unique():
    gdf = gpd.GeoDataFrame({"description_a": [1], "description_b": [2], "ok": [3]}, geometry=[Point(0, 0)])
    cols = list(k2s.truncate_column_names(gdf).columns)
    assert len(set(cols)) == len(cols) and all(len(c) <= 10 for c in cols)


def test_kml_folder_to_shapefiles_reads_all_layers(tmp_path):
    kin, kout = tmp_path / "kml", tmp_path / "shp"
    kin.mkdir()
    kml = simplekml.Kml()
    for i, off in enumerate((0.0, 0.01)):  # two KML folders -> two layers
        folder = kml.newfolder(name=f"f{i}")
        folder.newpolygon(name=f"p{i}", outerboundaryis=[(LON0 + off, LAT0), (LON0 + off + 0.001, LAT0), (LON0 + off + 0.001, LAT0 + 0.001), (LON0 + off, LAT0)])
    kml.save(str(kin / "two.kml"))
    (kin / "empty.kml").write_text("<kml xmlns='http://www.opengis.net/kml/2.2'><Document/></kml>")
    ok, failed = k2s.convert_kml_folder(kin, kout)
    assert ok == 1 and failed == ["empty.kml"]
    out = gpd.read_file(kout / "two.shp")
    assert len(out) == 2 and out.crs.to_epsg() == 4326
