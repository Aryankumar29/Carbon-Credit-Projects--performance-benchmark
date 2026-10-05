"""End-to-end demo on synthetic data: match controls, validate, compute the benchmark."""
from vm0047_pb import benchmark_over_time, run_matching, to_long_series
from vm0047_pb.synthetic import make_plots, make_series

COV = ["SI_t-8", "SI_t-4", "SI_t0"]

project, control = make_plots(n_project=30, n_control=600)
match = run_matching(project, control, COV, k=3, xy_cols=None)
print(f"valid={match.valid} k={match.k}\n{match.sdm.round(3).to_string()}\n")

# planted effect: project grows 0.05/yr, controls 0.02/yr -> expect PB ~ 0.4
project_si, control_si = make_series(match.matches, years=6, project_slope=0.05, control_slope=0.02)
series = to_long_series(project_si, control_si, match.matches)
print(benchmark_over_time(series)[["t", "delta_si_project", "delta_si_control", "z", "pb"]].round(3).to_string(index=False))
