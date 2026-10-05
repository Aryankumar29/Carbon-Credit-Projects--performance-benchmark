# Mapping to VM0047 (Appendix 1)

Reference: Verra VM0047 *Afforestation, Reforestation and Revegetation*, Appendix 1 (section 10).
Checked against v1.0 text; v1.1 (effective 14 May 2025) did not change the benchmark procedure as far as
public summaries show. Verify against the current Verra document before relying on results.

| Methodology step | Requirement | Code |
|---|---|---|
| Step 1 | Plots 0.09-10 ha, >=75% inside boundary | `plots.make_grid(size_m, min_inside=0.75)` |
| Step 1 | Representative sample, n >= 30 | `plots.sample_plots` |
| Step 2 / Table A1 | Donor pool: jurisdiction, ecoregion, policy environment, tenure, outside registered AFOLU projects, <= 100 km | `donor.delineate_donor_pool` |
| Step 2 | Donor pool units within +/-20% of mean project plot size | `donor.tile_donor_pool` |
| Table A2 | Covariates: SI at t=-10..-8, t=-8..-1, t=0 (>= 3 points) | columns passed to `run_matching(covariates=...)` |
| Step 2.2 | Multivariate distance (Mahalanobis / Euclidean) | `matching.mahalanobis_matrix`, `euclidean_matrix` |
| Step 2.3 | k-NN optimal matching without replacement, constant k | `matching.knn_match(method="optimal")` (Hungarian algorithm) |
| Eq. A1 | Weights `exp(-MD) / sum exp(-MD)` | `matching.a1_weights` |
| Eq. A2 / Step 3 | SDM <= 0.25 per covariate, else widen radius in 100 km steps and/or lower k | `matching.standardized_difference`, `run_matching` |
| Step 4 | Drop invalid controls, replace from donor pool, reweight to 1 | `matching.replace_invalid_controls` |
| Step 5 | Complete matched sets only; accumulated series t=0..t; >= 3 time steps; n >= 30 | `benchmark.to_long_series`, `performance_benchmark` |
| Eq. A3-A4 | Series weights | constant factor `1/sum(n_rs,t)` cancels in slope and SE; control weights = A1 weights, project weights = 1 |
| Step 5 / Eq. A5 | Weighted regression slopes, Z-test, \|Z\| >= 1.96 | `benchmark.wls_slope`, `performance_benchmark` |
| Step 6 / Eq. A6 | PB = 1 if not significant; else dSI_control / dSI_project, control slope floored at 0 if p > 0.05 or < 0 | `benchmark.performance_benchmark` |
| 10.1 | Update at each verification or every 5 years | `benchmark.benchmark_over_time` |

## Interpretation choices (flagged for your validator)

- **Matching order when SDM fails.** The text allows widening the radius "and/or" reducing k. `run_matching`
  reduces k from the requested value and, for each k, widens the radius first. Inspect `MatchResult.attempts`.
- **SDM denominators.** Project variance over project plots; control variance over distinct matched controls;
  control mean is the mean of per-project weighted control sums.
- **A1 weights.** The prose says "proportional to the inverse" of MD but Equation A1 is exponential; the equation is implemented.
- **Standard errors.** WLS standard errors treat observations as independent; the methodology does not specify
  an adjustment for clustering within match sets.

## Not covered

- VMD0054 leakage, crediting-baseline equations (Eq. 30) and field-based carbon accounting.
- Stocking index production. The methodology expects an SI from a Verra-vetted data service provider (or an
  equivalent that meets the published criteria). `si.py` offers NDVI as a transparent proxy for development
  and screening only.
- Producing the categorical GIS layers (ecoregion, tenure, policy). Supply official layers.
