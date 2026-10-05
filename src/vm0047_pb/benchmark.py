"""Performance benchmark (VM0047 v1.1, Appendix 1, Steps 5-6; Equations A3-A6)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

MIN_PROJECT_PLOTS = 30
Z_CRIT = 1.96


def wls_slope(t: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted least squares slope, its standard error and two-sided p-value.

    Scale-invariant in the weights (same convention as statsmodels WLS).
    """
    t, y, w = (np.asarray(a, float) for a in (t, y, w))
    n = len(t)
    if n < 3 or np.ptp(t) == 0:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(n), t])
    xtwx = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(xtwx, X.T @ (w * y))
    res = y - X @ beta
    dof = n - 2
    sigma2 = float((w * res**2).sum() / dof)
    se = float(np.sqrt(sigma2 * np.linalg.inv(xtwx)[1, 1]))
    p = float(2 * stats.t.sf(abs(beta[1] / se), dof)) if se > 0 else 0.0
    return float(beta[1]), se, p


def to_long_series(
    project_si: pd.DataFrame, control_si: pd.DataFrame, matches: pd.DataFrame
) -> pd.DataFrame:
    """Build the regression dataset.

    project_si / control_si: index = plot id, columns = years since project start
    (t = 0, 1, 2, ...), values = stocking index. `matches` is MatchResult.matches.

    A project plot enters at time t only if it and *all* of its matched controls have
    an SI value at t (cloud-free), as required in Step 5. Control weights are the A1
    weights; project plots have weight 1.
    """
    rows = []
    for t in project_si.columns:
        for pid, grp in matches.groupby("project_id", sort=False):
            if pid not in project_si.index or pd.isna(project_si.at[pid, t]):
                continue
            cs = control_si.reindex(grp.control_id)[t]
            if cs.isna().any():
                continue
            rows.append((pid, pid, "project", t, project_si.at[pid, t], 1.0))
            for cid, w, v in zip(grp.control_id, grp.weight, cs):
                rows.append((pid, cid, "control", t, v, w))
    return pd.DataFrame(rows, columns=["set_id", "plot_id", "group", "t", "si", "weight"])


@dataclass
class BenchmarkResult:
    t: int
    n_project_plots: int
    delta_si_project: float
    se_project: float
    delta_si_control: float
    se_control: float
    p_control: float
    z: float
    significant: bool
    pb: float
    meets_min_plots: bool
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["warnings"] = "; ".join(self.warnings)
        return d


def performance_benchmark(series: pd.DataFrame, t_end: int) -> BenchmarkResult:
    """Steps 5-6 for the monitoring interval ending at year `t_end` (accumulated from t=0)."""
    warn: list[str] = []
    s = series[(series.t >= 0) & (series.t <= t_end)]
    proj, ctrl = s[s.group == "project"], s[s.group == "control"]
    if s.t.nunique() < 3:
        warn.append("fewer than 3 time steps; slopes not estimable")
    n_plots = proj[proj.t == t_end].set_id.nunique()
    meets = n_plots >= MIN_PROJECT_PLOTS
    if not meets:
        warn.append(f"only {n_plots} project plots at t={t_end}; minimum is {MIN_PROJECT_PLOTS}")

    # Weights per A3/A4 differ from these only by a common constant (1 / sum n_rs,t),
    # which cancels in the slope and its standard error.
    b_p, se_p, _ = wls_slope(proj.t, proj.si, proj.weight)
    b_c, se_c, p_c = wls_slope(ctrl.t, ctrl.si, ctrl.weight)
    z = (b_p - b_c) / np.sqrt(se_p**2 + se_c**2) if np.isfinite(se_p + se_c) and (se_p + se_c) > 0 else np.nan
    significant = bool(abs(z) >= Z_CRIT) if np.isfinite(z) else False

    if not significant:
        pb = 1.0
    else:
        dc = 0.0 if (not np.isfinite(b_c) or b_c < 0 or p_c > 0.05) else b_c
        if not np.isfinite(b_p) or b_p <= 0:
            pb = np.nan
            warn.append("project SI slope <= 0; benchmark undefined")
        else:
            pb = dc / b_p
    return BenchmarkResult(t_end, n_plots, b_p, se_p, b_c, se_c, p_c, z, significant, pb, meets, warn)


def benchmark_over_time(series: pd.DataFrame) -> pd.DataFrame:
    """Benchmark for every monitoring year with at least three time steps."""
    ts = sorted(series.t.unique())
    out = [performance_benchmark(series, int(t)).as_dict() for t in ts if t >= 2]
    return pd.DataFrame(out)
