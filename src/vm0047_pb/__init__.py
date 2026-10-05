"""VM0047 dynamic performance benchmark toolkit (unofficial; not endorsed by Verra)."""
from .benchmark import BenchmarkResult, benchmark_over_time, performance_benchmark, to_long_series, wls_slope
from .matching import MatchResult, a1_weights, knn_match, replace_invalid_controls, run_matching

__version__ = "0.1.0"
