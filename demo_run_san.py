"""
It create a SAN-solver pairing (using the directory) and runs multiple
macroreplications of the solver on the problem.
"""

import sys
import os.path as o

sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import the Experiment class and other useful functions
from wrapper_base import Experiment, post_normalize

# !! When testing a new solver/problem, first go to directory.py.
# There you should add the import statement and an entry in the respective
# dictionary (or dictionaries).
# See directory.py for more details.

m = 20  # Macro-replication
L = 200 # Post-replication

# Specify the names of the solver and problem to test.
# These names are strings and should match those input to directory.py.
solvers = ["ASTRODFRF_problem", "ASTRODFDH_problem"]
problem_name = "SAN-1"

experiments_same_problem = []
for solver_name in solvers:
    # Temporarily store experiments on the same problem for post-normalization.
    print(f"Testing solver {solver_name} on problem {problem_name}.")

    # Specify file path name for storing experiment outputs in .pickle file.
    file_name_path = "experiments/outputs/" + solver_name + "_on_" + problem_name + ".pickle"
    print(f"Results will be stored as {file_name_path}.")

    # Initialize an instance of the experiment class.
    myexperiment = Experiment(solver_name=solver_name, problem_name=problem_name)

    # Run a fixed number of macroreplications of the solver on the problem.
    myexperiment.run(n_macroreps=m)

    print("Post-processing results.")

    # Run a fixed number of postreplications at all recommended solutions.
    myexperiment.post_replicate(n_postreps=L)
    experiments_same_problem.append(myexperiment)

# Find an optimal solution x* for normalization.
post_normalize(experiments=experiments_same_problem, n_postreps_init_opt=L)
