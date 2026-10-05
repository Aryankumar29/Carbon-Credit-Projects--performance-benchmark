"""Control-plot matching (VM0047 v1.1, Appendix 1, Steps 2-3 and 4)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

BIG = 1e12


def euclidean_matrix(project_x: np.ndarray, control_x: np.ndarray) -> np.ndarray:
    diff = project_x[:, None, :] - control_x[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def mahalanobis_matrix(project_x: np.ndarray, control_x: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of every project plot to every control plot.

    The covariance matrix is estimated on the control (donor) population.
    """
    cov = np.atleast_2d(np.cov(control_x, rowvar=False))
    inv = np.linalg.pinv(cov)
    diff = project_x[:, None, :] - control_x[None, :, :]
    d2 = np.einsum("pci,ij,pcj->pc", diff, inv, diff)
    return np.sqrt(np.clip(d2, 0, None))


def knn_match(dist: np.ndarray, k: int, method: str = "optimal") -> np.ndarray:
    """k nearest controls per project plot, without replacement.

    Returns an (n_project, k) array of control column indices, sorted by distance.

    method="optimal": globally minimises total distance (Hungarian algorithm on
    k copies of each project plot), as the methodology asks for "optimal matching".
    method="greedy": project plots served in row order (order dependent).
    """
    n_p, n_c = dist.shape
    if n_c < n_p * k:
        raise ValueError(f"need at least n_project*k={n_p * k} control plots, got {n_c}")
    if method == "optimal":
        rows, cols = linear_sum_assignment(np.repeat(dist, k, axis=0))
        out = cols.reshape(n_p, k)
    elif method == "greedy":
        used: set[int] = set()
        out = np.empty((n_p, k), dtype=int)
        for i in range(n_p):
            picks = [j for j in np.argsort(dist[i]) if j not in used][:k]
            out[i] = picks
            used.update(picks)
    else:
        raise ValueError("method must be 'optimal' or 'greedy'")
    order = np.argsort(np.take_along_axis(dist, out, axis=1), axis=1)
    return np.take_along_axis(out, order, axis=1)


def a1_weights(distances: np.ndarray) -> np.ndarray:
    """Equation A1: W_ij = exp(-MD_ij) / sum_j exp(-MD_ij), per project plot (rows)."""
    d = np.atleast_2d(np.asarray(distances, dtype=float))
    e = np.exp(-(d - d.min(axis=1, keepdims=True)))
    return e / e.sum(axis=1, keepdims=True)


def standardized_difference(
    project_x: np.ndarray, control_x: np.ndarray, matches: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Equation A2, per covariate.

    Control mean is the mean over project plots of the weighted sum of matched
    control covariates; control variance is taken over the distinct matched controls.
    """
    wsum = np.einsum("nk,nkp->np", weights, control_x[matches])
    used = np.unique(matches)
    var_p = project_x.var(axis=0, ddof=1)
    var_c = control_x[used].var(axis=0, ddof=1)
    return np.abs(project_x.mean(axis=0) - wsum.mean(axis=0)) / np.sqrt((var_p + var_c) / 2)


@dataclass
class MatchResult:
    matches: pd.DataFrame  # project_id, control_id, rank, distance, weight
    sdm: pd.Series
    k: int
    radius_km: float | None
    valid: bool
    threshold: float = 0.25
    attempts: list[dict] = field(default_factory=list)


def _attempt(project_x, control_x, dist, k, method):
    idx = knn_match(dist, k, method)
    d = np.take_along_axis(dist, idx, axis=1)
    if (d >= BIG).any():
        return None
    w = a1_weights(d)
    return idx, d, w, standardized_difference(project_x, control_x, idx, w)


def run_matching(
    project: pd.DataFrame,
    control: pd.DataFrame,
    covariates: list[str],
    k: int = 5,
    id_col: str = "plot_id",
    metric: str = "mahalanobis",
    method: str = "optimal",
    xy_cols: tuple[str, str] | None = ("x", "y"),
    radius_km: float = 100.0,
    radius_step_km: float = 100.0,
    max_radius_km: float = 500.0,
    sdm_threshold: float = 0.25,
) -> MatchResult:
    """Match, validate (SDM <= threshold) and, if invalid, widen the donor radius in
    steps and then reduce k, as prescribed in Step 3 of the methodology.

    `project` / `control` need an id column, the covariate columns and, for the
    radius test, projected centroid coordinates (metres) in `xy_cols`. Pass
    xy_cols=None to skip the distance filter.
    """
    px = project[covariates].to_numpy(float)
    cx = control[covariates].to_numpy(float)
    base = (mahalanobis_matrix if metric == "mahalanobis" else euclidean_matrix)(px, cx)
    use_xy = xy_cols is not None and all(c in project and c in control for c in xy_cols)
    if use_xy:
        geo = euclidean_matrix(project[list(xy_cols)].to_numpy(float), control[list(xy_cols)].to_numpy(float)) / 1000.0
        radii = list(np.arange(radius_km, max_radius_km + 1e-9, radius_step_km))
    else:
        geo, radii = None, [None]

    attempts, best = [], None
    for kk in range(k, 0, -1):
        for r in radii:
            dist = base if r is None else np.where(geo <= r, base, BIG)
            try:
                res = _attempt(px, cx, dist, kk, method)
            except ValueError:
                res = None
            if res is None:
                attempts.append({"k": kk, "radius_km": r, "feasible": False, "max_sdm": None})
                continue
            idx, d, w, sdm = res
            attempts.append({"k": kk, "radius_km": r, "feasible": True, "max_sdm": float(sdm.max())})
            best = (idx, d, w, sdm, kk, r)
            if (sdm <= sdm_threshold).all():
                return _pack(project, control, id_col, covariates, *best, True, sdm_threshold, attempts)
    if best is None:
        raise ValueError("no feasible matching; add control plots or raise max_radius_km")
    return _pack(project, control, id_col, covariates, *best, False, sdm_threshold, attempts)


def _pack(project, control, id_col, covariates, idx, d, w, sdm, k, r, valid, thr, attempts):
    pid = project[id_col].to_numpy()
    cid = control[id_col].to_numpy()
    rows = [
        (pid[i], cid[idx[i, j]], j + 1, d[i, j], w[i, j])
        for i in range(idx.shape[0])
        for j in range(idx.shape[1])
    ]
    df = pd.DataFrame(rows, columns=["project_id", "control_id", "rank", "distance", "weight"])
    return MatchResult(df, pd.Series(sdm, index=covariates, name="SDM"), k, r, valid, thr, attempts)


def replace_invalid_controls(
    matches: pd.DataFrame,
    invalid_ids: list,
    project: pd.DataFrame,
    control: pd.DataFrame,
    covariates: list[str],
    id_col: str = "plot_id",
    metric: str = "mahalanobis",
) -> pd.DataFrame:
    """Step 4: drop controls that became invalid, refill each affected match set back
    to k from unused donor-pool plots (nearest first), and renormalise the A1 weights."""
    invalid = set(invalid_ids)
    kept = matches[~matches.control_id.isin(invalid)].copy()
    k = int(matches.groupby("project_id").size().max())
    pool = control[~control[id_col].isin(invalid)].reset_index(drop=True)
    pcols = pool[id_col].to_numpy()
    dist = (mahalanobis_matrix if metric == "mahalanobis" else euclidean_matrix)(
        project[covariates].to_numpy(float), pool[covariates].to_numpy(float)
    )
    prow = {p: i for i, p in enumerate(project[id_col])}
    used = set(kept.control_id)
    new_rows = []
    for pid, grp in matches.groupby("project_id", sort=False):
        need = k - int((~grp.control_id.isin(invalid)).sum())
        if need <= 0:
            continue
        for j in np.argsort(dist[prow[pid]]):
            if pcols[j] in used:
                continue
            new_rows.append((pid, pcols[j], 0, dist[prow[pid], j], np.nan))
            used.add(pcols[j])
            need -= 1
            if need == 0:
                break
    out = pd.concat([kept, pd.DataFrame(new_rows, columns=kept.columns)], ignore_index=True)
    out = out.sort_values(["project_id", "distance"], kind="stable")
    out["rank"] = out.groupby("project_id").cumcount() + 1
    out["weight"] = out.groupby("project_id")["distance"].transform(
        lambda s: pd.Series(a1_weights(s.to_numpy())[0], index=s.index)
    )
    return out.reset_index(drop=True)
