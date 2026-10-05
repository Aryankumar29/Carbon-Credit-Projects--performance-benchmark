"""Command line interface: `vm0047-pb match ...` and `vm0047-pb benchmark ...`."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmark import benchmark_over_time, to_long_series
from .matching import run_matching


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vm0047-pb", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="match control plots to project plots")
    m.add_argument("--project", required=True, help="CSV: plot_id, covariates, optional x,y (metres)")
    m.add_argument("--control", required=True)
    m.add_argument("--covariates", nargs="+", required=True)
    m.add_argument("-k", type=int, default=5)
    m.add_argument("--metric", choices=["mahalanobis", "euclidean"], default="mahalanobis")
    m.add_argument("--method", choices=["optimal", "greedy"], default="optimal")
    m.add_argument("--radius-km", type=float, default=100)
    m.add_argument("--max-radius-km", type=float, default=500)
    m.add_argument("--out", default="out")

    b = sub.add_parser("benchmark", help="compute the performance benchmark per monitoring year")
    b.add_argument("--project-si", required=True, help="CSV: plot_id + one column per year t=0,1,2...")
    b.add_argument("--control-si", required=True)
    b.add_argument("--matches", required=True, help="matches.csv written by `match`")
    b.add_argument("--out", default="out")

    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.cmd == "match":
        r = run_matching(
            pd.read_csv(a.project), pd.read_csv(a.control), a.covariates, k=a.k, metric=a.metric,
            method=a.method, radius_km=a.radius_km, max_radius_km=a.max_radius_km,
        )
        r.matches.to_csv(out / "matches.csv", index=False)
        r.sdm.to_csv(out / "sdm.csv")
        print(f"k={r.k} radius_km={r.radius_km} valid={r.valid}")
        print(r.sdm.to_string())
        return 0 if r.valid else 1

    def load(p):
        df = pd.read_csv(p, index_col="plot_id")
        df.columns = [int(c) for c in df.columns]
        return df

    series = to_long_series(load(a.project_si), load(a.control_si), pd.read_csv(a.matches))
    res = benchmark_over_time(series)
    res.to_csv(out / "benchmark.csv", index=False)
    print(res[["t", "n_project_plots", "delta_si_project", "delta_si_control", "z", "pb"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
