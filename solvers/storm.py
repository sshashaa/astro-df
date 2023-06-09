"""
Summary
-------
The ASTRO-DF solver progressively builds local models (quadratic with diagonal Hessian) using interpolation on a set of points on the coordinate bases of the best (incumbent) solution. Solving the local models within a trust region (closed ball around the incumbent solution) at each iteration suggests a candidate solution for the next iteration. If the candidate solution is worse than the best interpolation point, it is replaced with the latter (a.k.a. direct search). The solver then decides whether to accept the candidate solution and expand the trust-region or reject it and shrink the trust-region based on a success ratio test. The sample size at each visited point is determined adaptively and based on closeness to optimality.
A detailed description of the solver can be found `here <https://simopt.readthedocs.io/en/latest/astrodf.html>`_.
"""
import os
from numpy.linalg import pinv
from numpy.linalg import norm
import numpy as np
from math import log, ceil
import warnings
from scipy.optimize import NonlinearConstraint
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

from base import Solver


class STORM(Solver):
    """The ASTRO-DF solver.

    Attributes
    ----------
    name : string
        name of solver
    objective_type : string
        description of objective types:
            "single" or "multi"
    constraint_type : string
        description of constraints types:
            "unconstrained", "box", "deterministic", "stochastic"
    variable_type : string
        description of variable types:
            "discrete", "continuous", "mixed"
    gradient_needed : bool
        indicates if gradient of objective function is needed
    factors : dict
        changeable factors (i.e., parameters) of the solver
    specifications : dict
        details of each factor (for GUI, data validation, and defaults)
    rng_list : list of mrg32k3a.mrg32k3a.MRG32k3a objects
        list of RNGs used for the solver's internal purposes

    Arguments
    ---------
    name : str
        user-specified name for solver
    fixed_factors : dict
        fixed_factors of the solver
    See also
    --------
    base.Solver
    """
    def __init__(self, name="STORM", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "use CRN across solutions?",
                "datatype": bool,
                "default": True
            },
            "eta_1": {
                "description": "threshhold for a successful iteration",
                "datatype": float,
                "default": 0.1
            },
            "eta_2": {
                "description": "threshhold for a very successful iteration",
                "datatype": float,
                "default": 0.5
            },
            "gamma_1": {
                "description": "very successful step trust-region radius increase",
                "datatype": float,
                "default": 1.5
            },
            "gamma_2": {
                "description": "unsuccessful step trust-region radius decrease",
                "datatype": float,
                "default": 0.75
            },
            "w": {
                "description": "trust-region radius rate of shrinkage in contracation loop",
                "datatype": float,
                "default": 0.85
            },
            "mu": {
                "description": "trust-region radius ratio upper bound in contraction loop",
                "datatype": int,
                "default": 1000
            },
            "beta": {
                "description": "trust-region radius ratio lower bound in contraction loop",
                "datatype": int,
                "default": 10
            },
            "lambda_min": {
                "description": "minimum sample size value",
                "datatype": int,
                "default": 5
            },
            "simple_solve": {
                "description": "solve subproblem with Cauchy point (rough approximate)?",
                "datatype": bool,
                "default": True
            },
            "criticality_select": {
                "description": "skip contraction loop if not near critical region?",
                "datatype": bool,
                "default": True
            },
            "reuse_points": {
                "description": "reuse the previously visited points?",
                "datatype": bool,
                "default": False
            },
            "criticality_threshold": {
                "description": "threshold on gradient norm indicating near-critical region",
                "datatype": float,
                "default": 0.1
            }
        }
        self.check_factor_list = {
            "crn_across_solns": self.check_crn_across_solns,
            "eta_1": self.check_eta_1,
            "eta_2": self.check_eta_2,
            "gamma_1": self.check_gamma_1,
            "gamma_2": self.check_gamma_2,
            "w": self.check_w,
            "beta": self.check_beta,
            "mu": self.check_mu,
            "lambda_min": self.check_lambda_min,
            "criticality_threshold": self.check_criticality_threshold
        }
        super().__init__(fixed_factors)

    def check_eta_1(self):
        return self.factors["eta_1"] > 0

    def check_eta_2(self):
        return self.factors["eta_2"] > self.factors["eta_1"]

    def check_gamma_1(self):
        return self.factors["gamma_1"] > 1

    def check_gamma_2(self):
        return (self.factors["gamma_2"] < 1 and self.factors["gamma_2"] > 0)

    def check_w(self):
        return (self.factors["w"] < 1 and self.factors["w"] > 0)

    def check_beta(self):
        return (self.factors["beta"] < self.factors["mu"] and self.factors["beta"] > 0)

    def check_mu(self):
        return self.factors["mu"] > 0

    def check_lambda_min(self):
        return self.factors["lambda_min"] > 2

    def check_criticality_threshold(self):
        return self.factors["criticality_threshold"] > 0

    # generate the coordinate vector corresponding to the variable number v_no
    def get_coordinate_vector(self, size, v_no):
        arr = np.zeros(size)
        arr[v_no] = 1.0
        return arr

    # generate the basis (rotated coordinate) (the first vector comes from the visited design points (origin basis))
    def get_rotated_basis(self, first_basis, rotate_index):
        rotate_matrix = np.array(first_basis)
        rotation = np.matrix([[0, -1], [1, 0]])

        # rotate the coordinate basis based on the first basis vector (first_basis)
        # choose two dimensions which we use for the rotation (0,i)
        for i in range(1,len(rotate_index)):
            v1 = np.array([[first_basis[rotate_index[0]]],  [first_basis[rotate_index[i]]]])
            v2 = np.dot(rotation, v1)
            rotated_basis = np.copy(first_basis)
            rotated_basis[rotate_index[0]] = v2[0][0]
            rotated_basis[rotate_index[i]] = v2[1][0]
            # stack the rotated vector
            rotate_matrix = np.vstack((rotate_matrix,rotated_basis))

        return rotate_matrix

    # compute the local model value with a linear interpolation with a diagonal Hessian
    def evaluate_model(self, x_k, intercept, grad, Hessian):
        #X = [1]
        #X = np.append(X, np.array(x_k))
        #X = np.append(X, np.array(x_k) ** 2)
        M = 0
        M += intercept
        M += np.matmul(x_k,grad)
        M += np.dot(np.dot(x_k,Hessian),x_k)/2
        return M

    def generate_random_vector(self, dim, delta, seed):
        rng = np.random.default_rng(seed)
        coords = rng.uniform(-1, 1, size=dim)
        coords = coords/np.linalg.norm(coords)
        mean = 0.5*delta
        delta_ratio = rng.uniform(mean,delta)
        coords = coords*delta_ratio
        return coords

    # compute the sample size based on adaptive sampling stopping rule using the optimality gap
    def get_stopping_time(self, k, sig2, delta, kappa, dim):
        if kappa == 0:
            kappa = 1

        lambda_min = self.factors["lambda_min"]
        lambda_k = max(lambda_min, 2 * log(dim,10)) * max(log(k + 0.1, 10) ** (1.01), 1)

        # compute sample size
        N_k = ceil(max(lambda_k, lambda_k * sig2 / ((kappa ** 2) * max(delta ** 2, delta**4))))

        return N_k

    # construct the "qualified" local model for each iteration k with the center point x_k
    # reconstruct with new points in a shrunk trust-region if the model fails the criticality condition
    # the criticality condition keeps the model gradient norm and the trust-region size in lock-step
    def construct_model(self, x_k, delta, k, problem, expended_budget, kappa, new_solution, visited_pts_list):
        interpolation_solns = []
        w = self.factors["w"]
        mu = self.factors["mu"]
        beta = self.factors["beta"]
        lambda_min = self.factors["lambda_min"]
        criticality_select = self.factors["criticality_select"]
        criticality_threshold = self.factors["criticality_threshold"]
        reuse_points = self.factors["reuse_points"]

        num_design_set = int((problem.dim + 1)*(problem.dim + 2)/2)
        #num_design_set = 100

        j = 0
        budget = problem.factors["budget"]

        while True:
            fval = []
            j = j + 1
            delta_k = delta * w ** (j - 1)

            # Construct the interpolation set
            Y = self.get_regression_points(x_k, delta_k, problem)

            # Evaluate the function estimate for the interpolation points
            for i in range(num_design_set):
                # for new points, we need to run the simulation
                new_solution = self.create_new_solution(tuple(Y[i]), problem)
                visited_pts_list.append(new_solution)
                # pilot run # ??check if there is existing result
                pilot_run = int(max(lambda_min, .5 * problem.dim) - 1)
                problem.simulate(new_solution, pilot_run)
                expended_budget += pilot_run
                sample_size = pilot_run

                # adaptive sampling
                while True:
                    problem.simulate(new_solution, 1)
                    expended_budget += 1
                    sample_size += 1
                    sig2 = new_solution.objectives_var
                    if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= 0.1*budget:
                        break
                fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                interpolation_solns.append(new_solution)

            #print('fval', fval)
            # construct the model and obtain the model coefficients
            intercept, coef, grad, Hessian = self.get_model_coefficients(Y, fval, problem)
            #print('model info', intercept, coef, grad, Hessian)
            # CHECK
            #for i in range(num_design_set):
            #    t = self.evaluate_model(Y[i], intercept, grad, Hessian)
            #    print(t)
            #    print(intercept,'2')
            #print(intercept, coef, grad)
            #print(Hessian)

            if not criticality_select:
                # check the condition and break
                if norm(grad) > criticality_threshold:
                    break

            if delta_k <= mu * norm(grad):
                break

            # If a model gradient norm is zero, there is a possibility that the code stuck in this while loop
            if norm(grad) == 0:
                break

        delta_k = min(max(beta * norm(grad), delta_k), delta)

        return fval, Y, intercept, grad, Hessian, delta_k, expended_budget, interpolation_solns, visited_pts_list

    # compute the model coefficients using (2d+1) design points and their function estimates
    def get_model_coefficients(self, Y, fval, problem):
        # Transform the input data to include quadratic terms
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(Y)

        # Fit the model
        model = LinearRegression().fit(X_poly, fval)

        # Extract the coefficients
        intercept, coef = model.intercept_, model.coef_
        coef = coef[0]

        # Calculate grad
        grad = coef[1:problem.dim + 1]
        grad = np.reshape(grad, problem.dim)

        # Calculate the Hessian
        hessian = np.zeros((problem.dim, problem.dim))

        for i in range(problem.dim):
            for j in range(problem.dim):
                if i <= j:
                    hessian[i, j] = coef[problem.dim + 1 + i + j]
                    if i == j:
                        hessian[i, j] = 2*hessian[i, j]
                    if i != j:
                        hessian[j, i] = hessian[i, j]

        return intercept, coef, grad, hessian

    # pick the design set (d+1)(d+2)/2
    def get_regression_points(self, x_k, delta, problem):
        Y = []
        epsilon = 0.01
        num_design_set = (problem.dim + 1)*(problem.dim + 2)/2

        for i in range(int(num_design_set)):
            new_point = self.generate_random_vector(problem.dim, delta, i)
            new_point = x_k + new_point
            for j in range(problem.dim):
                if sum(x_k) != 0:
                    # block constraints
                    if new_point[j] <= problem.lower_bounds[j]:
                        new_point[j] = problem.lower_bounds[j] + epsilon
                    if new_point[j] >= problem.upper_bounds[j]:
                        new_point[j] = problem.upper_bounds[j] - epsilon

            Y.append(new_point)
        return Y

    # run one iteration of trust-region algorithm by bulding and solving a local model and updating the current incumbent and trust-region radius, and saving the data
    def iterate(self, k, delta_k, delta_max, problem, visited_pts_list, new_x, expended_budget, budget_limit, recommended_solns, intermediate_budgets, kappa, new_solution):
        # default values
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        simple_solve = self.factors["simple_solve"]
        lambda_min = self.factors["lambda_min"]


        if k == 1:
            new_solution = self.create_new_solution(tuple(new_x), problem)
            if len(visited_pts_list) == 0:
                visited_pts_list.append(new_solution)

            # calculate kappa
            # pilot run
            pilot_run = int(max(lambda_min, .5 * problem.dim) - 1)
            problem.simulate(new_solution, pilot_run)
            expended_budget += pilot_run
            sample_size = pilot_run
            while True:
                problem.simulate(new_solution, 1)
                expended_budget += 1
                sample_size += 1
                fn = new_solution.objectives_mean
                sig2 = new_solution.objectives_var
                # ...
                if sample_size >= self.get_stopping_time(k, sig2, delta_k, fn / (delta_k ** 2), problem.dim) or expended_budget >= budget_limit:
                    kappa = fn / (delta_k ** 2)
                    break

            recommended_solns.append(new_solution)
            intermediate_budgets.append(expended_budget)
        fval, Y, intercept, grad, Hessian, delta_k, expended_budget, interpolation_solns, visited_pts_list = self.construct_model(new_x, delta_k, k, problem, expended_budget, kappa, new_solution, visited_pts_list)

        if simple_solve:
            # Cauchy reduction
            if np.dot(np.dot(grad, Hessian), grad) <= 0:
                tau = 1
            else:
                tau = min(1, norm(grad) ** 3 / (delta_k * np.dot(np.dot(grad, Hessian), grad)))
            grad = np.reshape(grad, (1, problem.dim))[0]
            candidate_x = new_x - tau * delta_k * grad / norm(grad)
        else:
            # Search engine - solve subproblem
            def subproblem(s):
                return fval[0] + np.dot(s, grad) + np.dot(np.multiply(s, Hessian), s)

            con_f = lambda s: norm(s)
            nlc = NonlinearConstraint(con_f, 0, delta_k)
            solve_subproblem = minimize(subproblem, np.zeros(problem.dim), method='trust-constr', constraints=nlc)
            candidate_x = new_x + solve_subproblem.x

        # handle the box constraints
        for i in range(problem.dim):
            if candidate_x[i] <= problem.lower_bounds[i]:
                candidate_x[i] = problem.lower_bounds[i] + 0.01
            elif candidate_x[i] >= problem.upper_bounds[i]:
                candidate_x[i] = problem.upper_bounds[i] - 0.01

        candidate_solution = self.create_new_solution(tuple(candidate_x), problem)
        visited_pts_list.append(candidate_solution)

        # pilot run
        pilot_run = int(max(lambda_min, .5 * problem.dim) - 1)
        problem.simulate(candidate_solution, pilot_run)
        expended_budget += pilot_run
        sample_size = pilot_run

        # adaptive sampling
        while True:
            problem.simulate(candidate_solution, 1)
            expended_budget += 1
            sample_size += 1
            sig2 = candidate_solution.objectives_var
            if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget_limit:
                break

        # calculate success ratio
        fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

        # compute the success ratio
        if (self.evaluate_model(np.zeros(problem.dim), intercept, grad, Hessian) - self.evaluate_model(np.array(candidate_x) - np.array(new_x), intercept, grad, Hessian)) <= 0:
            rho = 0
        else:
            rho = (fval[0] - fval_tilde) / (self.evaluate_model(np.zeros(problem.dim), intercept, grad, Hessian) - self.evaluate_model(candidate_x - new_x, intercept, grad, Hessian))

        # very successful: expand and accept
        if rho >= eta_2:
            new_x = candidate_x
            new_solution = candidate_solution
            final_ob = candidate_solution.objectives_mean
            delta_k = min(gamma_1 * delta_k, delta_max)
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
        # successful: accept
        elif rho >= eta_1:
            new_x = candidate_x
            new_solution = candidate_solution
            final_ob = candidate_solution.objectives_mean
            delta_k = min(delta_k, delta_max)
            recommended_solns.append(candidate_solution)
            intermediate_budgets.append(expended_budget)
        # unsuccessful: shrink and reject
        else:
            delta_k = min(gamma_2 * delta_k, delta_max)
            final_ob = fval[0]

        norm_grad = norm(grad)
        #print(new_x)
        return final_ob, delta_k, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, new_solution, visited_pts_list, norm_grad


    # start the search and stop when the budget is exhausted
    def solve(self, problem):
        """
        Run a single macroreplication of a solver on a problem.
        Arguments
        ---------
        problem : Problem object
            simulation-optimization problem to solve
        crn_across_solns : bool
            indicates if CRN are used when simulating different solutions
        Returns
        -------
        recommended_solns : list of Solution objects
            list of solutions recommended throughout the budget
        intermediate_budgets : list of ints
            list of intermediate budgets when recommended solutions changes
        """

        budget = problem.factors["budget"]
        # Designate random number generator for random sampling
        find_next_soln_rng = self.rng_list[1]

        # Generate many dummy solutions without replication only to find a reasonable maximum radius
        dummy_solns = []
        for i in range(10000*problem.dim):
            dummy_solns += [problem.get_random_solution(find_next_soln_rng)]
        # Range for each dimension is calculated and compared with box constraints range if given
        # TODO: just use box constraints range if given
        # delta_max = min(self.factors["delta_max"], problem.upper_bounds[0] - problem.lower_bounds[0])
        delta_max_arr = []
        for i in range(problem.dim):
            delta_max_arr += [min(max([sol[i] for sol in dummy_solns])-min([sol[i] for sol in dummy_solns]),
                                  problem.upper_bounds[0] - problem.lower_bounds[0])]

        # TODO: update this so that it could be used for problems with decision variables at varying scales!
        delta_max = max(delta_max_arr)

        # Three values for the delta_0 obtained from a fraction on the delta_max
        delta_start = delta_max * 0.05
        delta_candidate = [0.1 * delta_start, delta_start, delta_start / 0.1]

        visited_pts_list = []
        k = 1

        # parameter tuning runs
        # run the first iteration with three choices of the initial trust region radius
        # return the one (of three) that more quickly progresses in search
        final_ob, delta_k, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, new_solution, visited_pts_list, norm_grad = self.iterate(k, \
        delta_candidate[0], delta_max, problem, visited_pts_list, problem.factors["initial_solution"], 0, budget * 0.01, recommended_solns =[], intermediate_budgets=[], kappa=1, new_solution=[])
        expended_budget_best = expended_budget

        for i in range(1, 3):
            final_ob_pt, delta_pt, recommended_solns_pt, intermediate_budgets_pt, expended_budget_pt, new_x_pt, kappa_pt, new_solution_pt, visited_pts_list, norm_grad_pt = self.iterate(k, \
                delta_candidate[i], delta_max, problem, visited_pts_list, problem.factors["initial_solution"], 0, budget * 0.01, recommended_solns=[], intermediate_budgets=[], kappa=1, new_solution=[])
            expended_budget += expended_budget_pt
            if -1 * problem.minmax[0] * final_ob_pt < -1 * problem.minmax[0] * final_ob:
                delta_k = delta_pt
                final_ob = final_ob_pt
                recommended_solns = recommended_solns_pt
                intermediate_budgets = intermediate_budgets_pt
                expended_budget_best = expended_budget_pt
                new_x = new_x_pt
                new_solution = new_solution_pt
                kappa = kappa_pt
                norm_grad = norm_grad_pt

        # continue the search from the best initial trust-region after parameter tuning
        intermediate_budgets = (intermediate_budgets + np.ones(len(intermediate_budgets))*(expended_budget - expended_budget_best)).tolist()
        intermediate_budgets[0] = 0

        # append delta history

        while (expended_budget < budget):
            k += 1
            final_ob, delta_k, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, new_solution, visited_pts_list, norm_grad = self.iterate(k,
                delta_k, delta_max, problem, visited_pts_list, new_x, expended_budget, budget, recommended_solns, intermediate_budgets, kappa, new_solution)

        return recommended_solns, intermediate_budgets