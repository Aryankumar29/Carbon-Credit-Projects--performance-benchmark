import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vm0047_pb import (a1_weights, benchmark_over_time, knn_match, performance_benchmark,
                       replace_invalid_controls, run_matching, to_long_series, wls_slope)
from vm0047_pb.synthetic import make_plots, make_series

COV = ["SI_t-8", "SI_t-4", "SI_t0"]


def test_a1_weights_sum_to_one_and_favor_close():
    w = a1_weights(np.array([[0.1, 0.5, 2.0]]))
    assert w.sum() == pytest.approx(1)
    assert w[0, 0] > w[0, 1] > w[0, 2]


def test_no_control_reuse_and_k_constant():
    proj, ctrl = make_plots()
    r = run_matching(proj, ctrl, COV, k=5, xy_cols=None)
    assert r.matches.control_id.is_unique
    assert (r.matches.groupby("project_id").size() == 5).all()
    assert r.matches.groupby("project_id").weight.sum().round(9).eq(1).all()


def test_optimal_not_worse_than_greedy():
    proj, ctrl = make_plots(seed=3)
    opt = run_matching(proj, ctrl, COV, k=3, xy_cols=None, method="optimal").matches.distance.sum()
    gre = run_matching(proj, ctrl, COV, k=3, xy_cols=None, method="greedy").matches.distance.sum()
    assert opt <= gre + 1e-9


def test_sdm_valid_on_similar_populations():
    proj, ctrl = make_plots(n_control=600)
    r = run_matching(proj, ctrl, COV, k=2, xy_cols=None)
    assert r.valid and (r.sdm <= 0.25).all()


def test_radius_filter_and_expansion():
    proj, ctrl = make_plots()
    ctrl = ctrl.copy()
    ctrl["x"] += 150_000  # all donors ~150 km away: 100 km radius infeasible, 200 km works
    r = run_matching(proj, ctrl, COV, k=2, radius_km=100, max_radius_km=300)
    assert r.radius_km == 200
    assert not r.attempts[0]["feasible"]


def test_wls_matches_scipy_linregress_when_unweighted():
    rng = np.random.default_rng(0)
    t = np.tile(np.arange(6), 20)
    y = 0.03 * t + rng.normal(0, 0.02, t.size)
    b, se, p = wls_slope(t, y, np.ones_like(y))
    ref = stats.linregress(t, y)
    assert (b, se, p) == pytest.approx((ref.slope, ref.stderr, ref.pvalue))


def _series(**kw):
    proj, ctrl = make_plots(n_control=600)
    m = run_matching(proj, ctrl, COV, k=3, xy_cols=None).matches
    ps, cs = make_series(m, **kw)
    return m, to_long_series(ps, cs, m)


def test_pb_recovers_planted_ratio():
    _, s = _series(project_slope=0.05, control_slope=0.02, noise=0.005)
    r = performance_benchmark(s, 5)
    assert r.significant and r.meets_min_plots
    assert r.pb == pytest.approx(0.4, abs=0.1)


def test_pb_is_one_when_no_difference():
    _, s = _series(project_slope=0.03, control_slope=0.03, noise=0.02)
    assert performance_benchmark(s, 5).pb == 1.0


def test_negative_control_slope_floors_to_zero():
    _, s = _series(project_slope=0.05, control_slope=-0.02, noise=0.005)
    assert performance_benchmark(s, 5).pb == 0.0


def test_incomplete_sets_dropped_and_min_plots_flagged():
    m, s = _series()
    drop = m.control_id.iloc[0]
    ps, cs = make_series(m)
    cs.loc[drop, 3] = np.nan
    s2 = to_long_series(ps, cs, m)
    assert s2[(s2.t == 3)].set_id.nunique() == s[(s.t == 3)].set_id.nunique() - 1
    assert benchmark_over_time(s).shape[0] == 4


def test_replace_invalid_controls_keeps_k_and_weights():
    proj, ctrl = make_plots(n_control=600)
    m = run_matching(proj, ctrl, COV, k=3, xy_cols=None).matches
    bad = list(m.control_id.iloc[:4])
    m2 = replace_invalid_controls(m, bad, proj, ctrl, COV)
    assert not set(bad) & set(m2.control_id)
    assert m2.control_id.is_unique
    assert (m2.groupby("project_id").size() == 3).all()
    assert m2.groupby("project_id").weight.sum().round(9).eq(1).all()
