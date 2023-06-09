"""
This script is intended to help with debugging problems and solvers.
It create a problem-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o
import os
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the Experiment class and other useful functions
from wrapper_base import read_experiment_results, plot_progress_curves

problems = ['SAN-1', 'RSBR-1']
for problem_name in problems:
    myexperiment = []
    n = 2

    file_name = f"{'ASTRODFRF_problem'}_on_{problem_name}"
    experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
    experiment.solver.name = "ASTRO-DF with Direct Search"
    myexperiment.append(experiment)

    file_name = f"{'ASTRODFDH_problem'}_on_{problem_name}"
    experiment = read_experiment_results(f"experiments/outputs/{file_name}.pickle")
    experiment.solver.name = "ASTRO-DF without Direct Search"
    myexperiment.append(experiment)

    print("Plotting results.")

    # Produce basic plots of the solver on the problem
    plot_progress_curves(experiments=[myexperiment[idx] for idx in range(n)], plot_type="mean", normalize=False)

    # Plots will be saved in the folder experiments/plots.
    print("Finished. " + problem_name + " Plots can be found in experiments/plots folder.")
