# utils - data preparation helpers

Small tools for getting plot boundaries into the shapes the benchmark pipeline needs.
They grew out of field-data workflows (GPS logger tracks -> KML -> Shapefile) and were
rewritten as tested modules. Install with `pip install -e ".[utils]"`.

| Module | CLI | Purpose |
|---|---|---|
| `kml_to_shp.py` | `vm0047-kml2shp IN_DIR OUT_DIR` | Batch KML -> polygon Shapefile with geometry repair |
| `gps_to_kml.py` | `vm0047-gps2kml IN OUT` | GPS point logs (`latitude`, `longitude` columns) -> KML polygon |
| `gps_shapes.py` | `vm0047-gpsshapes IN OUT.png` | Compare ways of closing a GPS track into a boundary, with areas |

## kml_to_shp

```bash
vm0047-kml2shp kml_in/ shp_out/ --log-file conversion.log
```

- Reads **every layer** (KML folder) of each file, not only the first.
- `fix_geometry` repairs what field exports commonly contain: open lines become closed rings,
  `MultiLineString` is merged or polygonized, invalid polygons are repaired with `buffer(0)`,
  collections are unioned, points become small buffers.
- Field names are truncated to Shapefile's 10-character limit, kept unique.
- Unrepairable geometries are **skipped and reported**. `--bbox-fallback` replaces them with their
  bounding box instead; this changes the boundary, so it is off by default.
- Exit code is non-zero if any file failed; failed names are printed.
- Output keeps the KML CRS (EPSG:4326).

## gps_to_kml

```bash
vm0047-gps2kml track.txt plot.kml      # one file
vm0047-gps2kml tracks/ kml_out/        # folder -> folder
vm0047-gps2kml tracks.zip kml.zip      # zip -> zip
```

Input is any CSV/TXT with `latitude` and `longitude` columns (extra columns such as `accuracy(m)`
are ignored). Empty rows are skipped, out-of-range values raise, consecutive identical fixes are
dropped (`--keep-duplicates` to keep them), and the ring is closed. One bad file does not stop a batch.
It needs at least 3 points.

## gps_shapes

```bash
vm0047-gpsshapes track.txt methods.png
```

Methods: `connect` (join last point to first), `convex_hull`, `alpha_shape` (concave boundary from
Delaunay triangles, default circumradius 2.5 x median edge), `spline` (periodic cubic smoothing).
Each panel title shows the area in hectares, so you can see how much the choice matters.
Coordinates are projected to the UTM zone of the data. Use `connect` for a track walked around a
boundary; hull/alpha for scattered points.

## Differences from the original notebooks

- UTM zone was hardcoded to 33N (Europe) and latitude/longitude were passed in swapped order, so
  shapes and areas were distorted. Now the right zone is derived from the data.
- The "alpha shape" function returned a convex hull. It is now a real alpha shape.
- Bounding-box replacement of broken geometries is opt-in rather than silent.
- Multi-layer KML files are fully read.

## Privacy

GPS logger exports often carry names and notes in `name`/`desc` columns, and plot boundaries are
personal land data. Only `latitude`/`longitude` are read here, but never commit real tracks, KMLs or
Shapefiles (see the repo `.gitignore`). Tests use synthetic coordinates only.
