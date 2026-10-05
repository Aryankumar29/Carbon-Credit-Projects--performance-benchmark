"""Stocking index (SI) extraction. Needs the [geo] extra for raster work.

The methodology's SI must come from a Verra-vetted data service provider or an
equivalent dataset that meets its criteria (<=30 m, annual, fixed season, biomass-correlated).
NDVI is provided here as a transparent proxy for development, teaching and screening.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red, nir = red.astype("float32"), nir.astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((nir + red) == 0, np.nan, (nir - red) / (nir + red))


def zonal_mean(raster_path: str, plots, band: int = 1, nodata=None) -> pd.Series:
    """Mean raster value inside each plot polygon (plots reprojected to the raster CRS)."""
    import rasterio
    from rasterio.mask import mask

    out = {}
    with rasterio.open(raster_path) as src:
        g = plots.to_crs(src.crs)
        nd = nodata if nodata is not None else src.nodata
        for pid, geom in zip(plots["plot_id"], g.geometry):
            try:
                arr, _ = mask(src, [geom], crop=True, indexes=band, filled=False)
            except ValueError:  # polygon outside raster
                out[pid] = np.nan
                continue
            a = np.ma.masked_invalid(np.ma.asarray(arr, dtype="float64"))
            if nd is not None:
                a = np.ma.masked_equal(a, nd)
            out[pid] = float(a.mean()) if a.count() else np.nan
    return pd.Series(out, name=raster_path)


def si_table(rasters_by_t: dict[int, str], plots, **kw) -> pd.DataFrame:
    """Wide SI table: index plot_id, one column per time key (e.g. -8, -4, 0 or 0..T)."""
    return pd.DataFrame({t: zonal_mean(p, plots, **kw) for t, p in rasters_by_t.items()})
