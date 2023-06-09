"""
This script runs three versions of random search, plus ASTRO-DF, on 25
versions of the (s, S) inventory problem.
Produces plots appearing in the INFORMS Journal on Computing submission.
"""

import sys
import os.path as o
import os

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from wrapper_base import Experiment, plot_area_scatterplots, post_normalize, plot_progress_curves, \
    plot_solvability_cdfs, read_experiment_results, plot_solvability_profiles

# Default values of the (s, S) model:
# "demand_mean": 100.0
# "lead_mean": 6.0
# "backorder_cost": 4.0
# "holding_cost": 1.0
# "fixed_cost": 36.0
# "variable_cost": 2.0

# Create 25 problem instances by varying two factors, five levels.
#st_devs = [1]
#holding_costs = [0.1]
#inven_stop = [100]

st_devs = [1,15]
holding_costs = [0.1,100]
inven_stop = [100, 10000]

# Three versions of random search with varying sample sizes.
rs_sample_sizes = [10]


experiments = []

# Load ASTRO-DF results.
solver_rename = "ASTRODF"
experiments_same_solver = []
for sd in st_devs:
    for hc in holding_costs:
        for inst in inven_stop:
            problem_rename = f"IRONORECONT-1_sd={sd}_hc={hc}_is={inst}"
            #file_name = f"{solver_rename}_on_{problem_rename}"
            file_name = f"ASTRODF_on_IRONORECONT-1"
            # Load experiment.
            new_experiment = read_experiment_results(f"/experiments/outputs/{file_name}.pickle")
            # Rename problem and solver to produce nicer plot labels.
            #new_experiment.solver.name = "ASTRO-DF"
            #new_experiment.problem.name = fr"IRONORECONT-1 with $sd={round(sd)}$ and $hc={round(hc)}$ and $is={round(inst)}$"
            experiments_same_solver.append(new_experiment)
experiments.append(experiments_same_solver)

# PLOTTING

n_solvers = len(experiments)
n_problems = len(experiments[0])

# All progress curves for one experiment. Problem instance 0.
plot_progress_curves([experiments[solver_idx][0] for solver_idx in range(n_solvers)], plot_type="all", all_in_one=True)

# All progress curves for one experiment. Problem instance 22.
plot_progress_curves([experiments[solver_idx][22] for solver_idx in range(n_solvers)], plot_type="all", all_in_one=True)

# Mean progress curves from all solvers on one problem. Problem instance 0.
plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
                     plot_type="mean",
                     all_in_one=True,
                     plot_CIs=True,
                     print_max_hw=False
                     )

# Mean progress curves from all solvers on one problem. Problem instance 22.
plot_progress_curves(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
                     plot_type="mean",
                     all_in_one=True,
                     plot_CIs=True,
                     print_max_hw=False
                     )

# Plot 0.9-quantile progress curves from all solvers on one problem. Problem instance 0.
plot_progress_curves(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
                     plot_type="quantile",
                     beta=0.9,
                     all_in_one=True,
                     plot_CIs=True,
                     print_max_hw=False
                     )

# Plot 0.9-quantile progress curves from all solvers on one problem. Problem instance 22.
plot_progress_curves(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
                     plot_type="quantile",
                     beta=0.9,
                     all_in_one=True,
                     plot_CIs=True,
                     print_max_hw=False
                     )

# Plot cdf of 0.2-solve times for all solvers on one problem. Problem instance 0.
plot_solvability_cdfs(experiments=[experiments[solver_idx][0] for solver_idx in range(n_solvers)],
                      solve_tol=0.2,
                      all_in_one=True,
                      plot_CIs=True,
                      print_max_hw=False
                      )

# Plot cdf of 0.2-solve times for all solvers on one problem. Problem instance 22.
plot_solvability_cdfs(experiments=[experiments[solver_idx][22] for solver_idx in range(n_solvers)],
                      solve_tol=0.2,
                      all_in_one=True,
                      plot_CIs=True,
                      print_max_hw=False
                      )

# Plot area scatterplots of all solvers on all problems.
plot_area_scatterplots(experiments=experiments,
                       all_in_one=True,
                       plot_CIs=False,
                       print_max_hw=False
                       )

# Plot cdf 0.1-solvability profiles of all solvers on all problems.
plot_solvability_profiles(experiments=experiments,
                          plot_type="cdf_solvability",
                          all_in_one=True,
                          plot_CIs=True,
                          print_max_hw=False,
                          solve_tol=0.1
                          )

# Plot 0.5-quantile 0.1-solvability profiles of all solvers on all problems.
plot_solvability_profiles(experiments=experiments,
                          plot_type="quantile_solvability",
                          all_in_one=True,
                          plot_CIs=True,
                          print_max_hw=False,
                          solve_tol=0.1,
                          beta=0.5
                          )

# Plot difference of cdf 0.1-solvability profiles of all solvers on all problems.
# Reference solver = ASTRO-DF.
plot_solvability_profiles(experiments=experiments,
                          plot_type="diff_cdf_solvability",
                          all_in_one=True,
                          plot_CIs=True,
                          print_max_hw=False,
                          solve_tol=0.1,
                          ref_solver="ASTRO-DF"
                          )

# Plot difference of 0.5-quantile 0.1-solvability profiles of all solvers on all problems.
# Reference solver = ASTRO-DF.
plot_solvability_profiles(experiments=experiments,
                          plot_type="diff_quantile_solvability",
                          all_in_one=True,
                          plot_CIs=True,
                          print_max_hw=False,
                          solve_tol=0.1,
                          beta=0.5,
                          ref_solver="ASTRO-DF"
                          )
