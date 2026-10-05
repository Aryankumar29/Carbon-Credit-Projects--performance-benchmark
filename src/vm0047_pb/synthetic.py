"""Synthetic data generator for tests and examples. Contains no real plot data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_plots(n_project=30, n_control=300, seed=0):
    """Historic SI (t-8, t-4, t0) + projected x/y (metres) for project and control plots."""
    rng = np.random.default_rng(seed)

    def draw(n, prefix, spread):
        base = rng.normal(0.30, 0.05, n)
        trend = rng.normal(0.004, 0.002, n)
        df = pd.DataFrame(
            {
                "plot_id": [f"{prefix}{i:04d}" for i in range(n)],
                "SI_t-8": base - 8 * trend + rng.normal(0, 0.01, n),
                "SI_t-4": base - 4 * trend + rng.normal(0, 0.01, n),
                "SI_t0": base + rng.normal(0, 0.01, n),
                "x": rng.uniform(0, spread, n),
                "y": rng.uniform(0, spread, n),
            }
        )
        return df

    return draw(n_project, "P", 5_000), draw(n_control, "C", 120_000)


def make_series(matches, years=6, project_slope=0.05, control_slope=0.02, noise=0.01, seed=1):
    """Wide SI tables (index plot id, columns t=0..years-1) with a planted growth gap."""
    rng = np.random.default_rng(seed)
    t = np.arange(years)

    def build(ids, slope):
        base = rng.normal(0.3, 0.03, len(ids))[:, None]
        return pd.DataFrame(base + slope * t + rng.normal(0, noise, (len(ids), years)), index=ids, columns=t)

    return (
        build(matches.project_id.unique(), project_slope),
        build(matches.control_id.unique(), control_slope),
    )
