import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
import numpy as np
from shapely import box

from vm0047_pb.donor import delineate_donor_pool, tile_donor_pool
from vm0047_pb.plots import make_grid, sample_plots
from vm0047_pb.si import ndvi, zonal_mean

CRS = "EPSG:32645"  # metric CRS so no reprojection surprises


def _area():
    return gpd.GeoDataFrame(geometry=[box(0, 0, 600, 600)], crs=CRS)


def test_grid_min_inside_and_size():
    g = make_grid(_area(), 30, 0.75)
    assert len(g) == 400
    assert (g.inside_frac >= 0.75).all()
    assert np.allclose(g.geometry.area, 900)
    with pytest.raises(ValueError):
        make_grid(_area(), 20)


def test_sample_requires_30():
    g = make_grid(_area())
    assert len(sample_plots(g, 30)) == 30
    assert len(sample_plots(g, 40, "systematic")) == 40
    with pytest.raises(ValueError):
        sample_plots(g, 10)


def test_donor_pool_filters():
    proj = _area()
    juris = gpd.GeoDataFrame(geometry=[box(-50_000, -50_000, 50_000, 50_000)], crs=CRS)
    drop = gpd.GeoDataFrame(geometry=[box(1_000, -50_000, 50_000, 50_000)], crs=CRS)
    pool = delineate_donor_pool(proj, jurisdiction=juris, registered_projects=drop, radius_km=100)
    minx, miny, maxx, maxy = pool.total_bounds
    assert minx == -50_000 and maxx == 1_000
    assert pool.geometry.iloc[0].intersection(proj.geometry.iloc[0]).area == 0
    tiles = tile_donor_pool(pool, make_grid(proj))
    assert len(tiles) > 0 and np.allclose(tiles.geometry.area, 900)


def test_zonal_mean_and_ndvi(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    arr = np.full((60, 60), 0.5, dtype="float32")
    arr[:, 30:] = 0.1
    path = tmp_path / "si.tif"
    with rasterio.open(path, "w", driver="GTiff", height=60, width=60, count=1, dtype="float32",
                       crs=CRS, transform=from_origin(0, 600, 10, 10)) as dst:
        dst.write(arr, 1)
    plots = gpd.GeoDataFrame({"plot_id": ["L", "R"]}, geometry=[box(0, 300, 290, 600), box(310, 300, 600, 600)], crs=CRS)
    m = zonal_mean(str(path), plots)
    assert m["L"] == pytest.approx(0.5) and m["R"] == pytest.approx(0.1)
    assert ndvi(np.array([0.1]), np.array([0.5]))[0] == pytest.approx(0.6667, abs=1e-3)
