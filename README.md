# vm0047-pb

Open-source implementation of the **dynamic performance benchmark** from Verra's VM0047
(Afforestation, Reforestation and Revegetation) methodology: plot sampling, donor-pool
delineation, control-plot matching, match validation and the benchmark (PB) calculation.

> Unofficial. Not endorsed by or affiliated with Verra. Results are not validated for registry
> submission; check every step against the current VM0047 document and with your validation body.

## What it does

```
project area --> 30 m plots --> sample (n>=30) --+
donor pool filters (100 km, ecoregion, ...) -----+--> match controls (Mahalanobis, k-NN, no reuse)
                                                      --> SDM <= 0.25? else widen radius / lower k
SI time series (project + matched controls) --> weighted slopes --> Z-test --> PB
```

Step-by-step mapping to Appendix 1, including interpretation choices: [docs/method_mapping.md](docs/method_mapping.md).

## Install

```bash
pip install -e .            # matching + benchmark (numpy, pandas, scipy)
pip install -e ".[geo]"     # + grid, donor pool, raster SI (geopandas, shapely, rasterio)
```

## Quick start

```bash
python examples/run_example.py    # synthetic data, expects PB ~ 0.4
```

```python
from vm0047_pb import run_matching, to_long_series, benchmark_over_time

match = run_matching(project_df, control_df, ["SI_t-8", "SI_t-4", "SI_t0"], k=5)
assert match.valid                       # all SDM <= 0.25
series = to_long_series(project_si, control_si, match.matches)   # wide tables, columns t = 0, 1, 2...
print(benchmark_over_time(series))       # PB per monitoring year
```

CLI: `vm0047-pb match --project p.csv --control c.csv --covariates SI_t-8 SI_t-4 SI_t0 -k 5`
then `vm0047-pb benchmark --project-si ... --control-si ... --matches out/matches.csv`.

## Data-preparation utilities

`src/vm0047_pb/utils/` has KML to Shapefile conversion with geometry repair and GPS track to KML/polygon helpers
(`vm0047-kml2shp`, `vm0047-gps2kml`, `vm0047-gpsshapes`). Details: [src/vm0047_pb/utils/README.md](src/vm0047_pb/utils/README.md).
Install with `pip install -e ".[utils]"`.

## Stocking index

The methodology expects SI from a Verra-vetted data service provider or an equivalent dataset
(pixels <= 30 m, annual, fixed season, correlated with biomass). Bring your own SI table; `vm0047_pb.si`
can also compute NDVI zonal means as a proxy for development and screening.

## Tests

```bash
pip install -e ".[geo,dev]" && pytest
```

Tests use synthetic data only. Never commit plot boundaries, rasters or credentials (see `.gitignore`).

## License

MIT
