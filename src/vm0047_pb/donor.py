"""Donor pool delineation (VM0047 v1.1, Appendix 1, Step 2 / Table A1). Needs the [geo] extra.

Every layer is a GeoDataFrame supplied by the user from official sources. Time-variant
layers must be current as of t=0 +/- 5 years and no coarser than 30 m (the caller's duty).
"""
from __future__ import annotations


def delineate_donor_pool(
    project_area,
    jurisdiction=None,
    ecoregion=None,
    tenure=None,
    tenure_col: str | None = None,
    policy_same=None,
    policy_different=None,
    registered_projects=None,
    radius_km: float = 100.0,
):
    """Return a single-row GeoDataFrame (metric CRS) with the eligible control area.

    jurisdiction : polygon(s) of the jurisdiction containing the project (kept).
    ecoregion    : polygon(s) of the project's ecoregion (kept).
    tenure       : land-tenure layer; with `tenure_col`, only classes present in the project
                   area are kept.
    policy_same  : polygons subject to the same tree-planting incentive programme (kept).
    policy_different : polygons whose programme status differs (removed).
    registered_projects : registered AFOLU project boundaries (removed, optional).
    radius_km    : maximum distance from the project area.
    """
    import geopandas as gpd

    crs = project_area.estimate_utm_crs() if project_area.crs.is_geographic else project_area.crs
    proj = project_area.to_crs(crs)
    pool = proj.geometry.union_all().buffer(radius_km * 1000)

    def g(layer):
        return layer.to_crs(crs)

    for keep in (jurisdiction, ecoregion, policy_same):
        if keep is not None:
            pool = pool.intersection(g(keep).geometry.union_all())
    if tenure is not None:
        t = g(tenure)
        if tenure_col:
            present = set(gpd.sjoin(t, proj[["geometry"]], predicate="intersects")[tenure_col])
            t = t[t[tenure_col].isin(present)]
        pool = pool.intersection(t.geometry.union_all())
    for drop in (policy_different, registered_projects):
        if drop is not None:
            pool = pool.difference(g(drop).geometry.union_all())
    pool = pool.difference(proj.geometry.union_all())
    return gpd.GeoDataFrame(geometry=[pool], crs=crs)


def tile_donor_pool(pool, project_plots, min_inside: float = 0.75):
    """Cut the pool into non-overlapping units sized like the mean project plot."""
    from .plots import make_grid

    size = float(project_plots.geometry.area.mean() ** 0.5)
    return make_grid(pool, size_m=max(size, 30.0), min_inside=min_inside)
