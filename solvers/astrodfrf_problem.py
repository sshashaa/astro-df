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
warnings.filterwarnings("ignore")

from base import Solver


class ASTRODFRF_problem(Solver):
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
    def __init__(self, name="ASTRODFRF_problem", fixed_factors=None):
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
    def evaluate_model(self, x_k, q):
        X = [1]
        X = np.append(X, np.array(x_k))
        X = np.append(X, np.array(x_k) ** 2)
        return np.matmul(X, q)

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
        j = 0
        budget = problem.factors["budget"]

        while True:
            fval = []
            j = j + 1
            delta_k = delta * w ** (j - 1)

            # Calculate the distance between the center point and other design points
            Dist = []
            for i in range(len(visited_pts_list)):
                Dist.append(norm(np.array(visited_pts_list[i].x) - np.array(x_k))-delta_k)
                # If the design point is outside the trust region, we will not reuse it (distance = -big M)
                if Dist[i] > 0:
                    Dist[i] = -delta_k*10000

            # Find the index of visited design points list for reusing points
            # The reused point will be the farthest point from the center point among the design points within the trust region
            f_index = Dist.index(max(Dist))

            # If it is the first iteration or there is no design point we can reuse within the trust region, use the coordinate basis
            if (k == 1) or (norm(np.array(x_k) - np.array(visited_pts_list[f_index].x))==0) or reuse_points == False:
                # Construct the interpolation set
                Y = self.get_coordinate_basis_interpolation_points(x_k, delta_k, problem)
                Z = self.get_coordinate_basis_interpolation_points(np.zeros(problem.dim), delta_k, problem)
            # Else if we will reuse one design point
            elif k > 1:
                first_basis = (np.array(visited_pts_list[f_index].x)-np.array(x_k)) / norm(np.array(visited_pts_list[f_index].x)-np.array(x_k))
                # if first_basis has some non-zero components, use rotated basis for those dimensions
                rotate_list = np.nonzero(first_basis)[0]
                rotate_matrix = self.get_rotated_basis(first_basis, rotate_list)

                # if first_basis has some zero components, use coordinate basis for those dimensions
                for i in range(problem.dim):
                    if first_basis[i] == 0:
                        rotate_matrix = np.vstack((rotate_matrix,self.get_coordinate_vector(problem.dim,i)))

                # construct the interpolation set
                Y = self.get_rotated_basis_interpolation_points(x_k, delta_k, problem, rotate_matrix, visited_pts_list[f_index].x)
                Z = self.get_rotated_basis_interpolation_points(np.zeros(problem.dim), delta_k, problem, rotate_matrix,
                                                         np.array(visited_pts_list[f_index].x) - np.array(x_k))

            # Evaluate the function estimate for the interpolation points
            for i in range(2 * problem.dim + 1):
                # for X_0, we don't need to simulate the new solution
                if (k == 1) and (i == 0):
                    fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                    interpolation_solns.append(new_solution)
                # if iteration k-1 is unsuccessful, we can reuse the replications for x_k
                elif (i == 0):
                    sample_size = new_solution.n_reps
                    sig2 = new_solution.objectives_var
                    # adaptive sampling
                    while True:
                        if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= 0.1*budget:
                            break
                        problem.simulate(new_solution, 1)
                        expended_budget += 1
                        sample_size += 1
                        sig2 = new_solution.objectives_var
                    fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                    interpolation_solns.append(new_solution)
                # else if we reuse one design point, reuse the replications
                elif (i == 1) and (norm(np.array(x_k) - np.array(visited_pts_list[f_index].x)) != 0) and reuse_points == True:
                    sample_size = visited_pts_list[f_index].n_reps
                    sig2 = visited_pts_list[f_index].objectives_var
                    # adaptive sampling
                    while True:
                        if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= 0.1*budget:
                            break
                        problem.simulate(visited_pts_list[f_index], 1)
                        expended_budget += 1
                        sample_size += 1
                        sig2 = visited_pts_list[f_index].objectives_var
                    fval.append(-1 * problem.minmax[0] * visited_pts_list[f_index].objectives_mean)
                    interpolation_solns.append(visited_pts_list[f_index])
                # for new points, we need to run the simulation
                else:
                    new_solution = self.create_new_solution(tuple(Y[i][0]), problem)
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

            # construct the model and obtain the model coefficients
            q, grad, Hessian = self.get_model_coefficients(Z, fval, problem)

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

        return fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns, visited_pts_list

    # compute the model coefficients using (2d+1) design points and their function estimates
    def get_model_coefficients(self, Y, fval, problem):
        M = []
        for i in range(0, 2 * problem.dim + 1):
            M.append(1)
            M[i] = np.append(M[i], np.array(Y[i]))
            M[i] = np.append(M[i], np.array(Y[i]) ** 2)

        q = np.matmul(pinv(M), fval)  # pinv returns the inverse of your matrix when it is available and the pseudo inverse when it isn't.
        grad = q[1:problem.dim + 1]
        grad = np.reshape(grad, problem.dim)
        Hessian = q[problem.dim + 1 : 2 * problem.dim + 1]
        Hessian = np.reshape(Hessian, problem.dim)
        return q, grad, Hessian

    # compute the interpolation points (2d+1) using the coordinate basis
    def get_coordinate_basis_interpolation_points(self, x_k, delta, problem):
        Y = [[x_k]]
        epsilon = 0.01
        for i in range(0, problem.dim):
            plus = Y[0] + delta * self.get_coordinate_vector(problem.dim, i)
            minus = Y[0] - delta * self.get_coordinate_vector(problem.dim, i)

            if sum(x_k) != 0:
                # block constraints
                if minus[0][i] <= problem.lower_bounds[i]:
                    minus[0][i] = problem.lower_bounds[i] + epsilon
                if plus[0][i] >= problem.upper_bounds[i]:
                    plus[0][i] = problem.upper_bounds[i] - epsilon

            Y.append(plus)
            Y.append(minus)
        return Y

    # compute the interpolation points (2d+1) using the rotated coordinate basis (reuse one design point)
    def get_rotated_basis_interpolation_points(self, x_k, delta, problem, rotate_matrix, reused_x):
        Y = [[x_k]]
        epsilon = 0.01
        for i in range(0, problem.dim):
            if i == 0:
                plus = [np.array(reused_x)]
            else:
                plus = Y[0] + delta * rotate_matrix[i]
            minus = Y[0] - delta * rotate_matrix[i]

            if sum(x_k) != 0:
                # block constraints
                for j in range(problem.dim):
                    if minus[0][j] <= problem.lower_bounds[j]:
                        minus[0][j] = problem.lower_bounds[j] + epsilon
                    elif minus[0][j] >= problem.upper_bounds[j]:
                        minus[0][j] = problem.upper_bounds[j] - epsilon
                    if plus[0][j] <= problem.lower_bounds[j]:
                        plus[0][j] = problem.lower_bounds[j] + epsilon
                    elif plus[0][j] >= problem.upper_bounds[j]:
                        plus[0][j] = problem.upper_bounds[j] - epsilon

            Y.append(plus)
            Y.append(minus)
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
        fval, Y, q, grad, Hessian, delta_k, expended_budget, interpolation_solns, visited_pts_list = self.construct_model(new_x, delta_k, k, problem, expended_budget, kappa, new_solution, visited_pts_list)


        if simple_solve:
            # Cauchy reduction
            if np.dot(np.multiply(grad, Hessian), grad) <= 0:
                tau = 1
            else:
                tau = min(1, norm(grad) ** 3 / (delta_k * np.dot(np.multiply(grad, Hessian), grad)))
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

        # replace the candidate x if the interpolation set has lower objective function value (pattern search)
        if (min(fval) < fval_tilde) and (fval[0] - min(fval) >= 0.01 * delta_k ** 2):
            minpos = fval.index(min(fval))
            fval_tilde = min(fval)
            candidate_x = Y[minpos][0]
            candidate_solution = interpolation_solns[minpos]

        #print(fval_tilde)
        # compute the success ratio
        if (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(np.array(candidate_x) - np.array(new_x), q)) <= 0:
            rho = 0
        else:
            rho = (fval[0] - fval_tilde) / (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(candidate_x - new_x, q))
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

        delta_list = []
        gradient_list = []
        function_list = []
        budget_list = []
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

        while k < 100:
            delta_list.append(delta_k)
            gradient_list.append(norm_grad)
            function_list.append(final_ob[0])
            budget_list.append(expended_budget)
            k += 1
            final_ob, delta_k, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, new_solution, visited_pts_list, norm_grad = self.iterate(k,
                delta_k, delta_max, problem, visited_pts_list, new_x, expended_budget, budget, recommended_solns, intermediate_budgets, kappa, new_solution)

        if problem.name == 'SAN-1':
            file_name_delta = 'delta_san_rf.npy'
            file_name_function = 'function_san_rf.npy'
            file_name_budget = 'budget_san_rf.npy'
        else:
            file_name_delta = 'delta_rsbr_rf.npy'
            file_name_function = 'function_rsbr_rf.npy'
            file_name_budget = 'budget_rsbr_rf.npy'

        # Save Delta history to plot it
        if os.path.exists(file_name_delta) == True:
            all_delta = np.load(file_name_delta)
            all_delta = np.vstack((all_delta, delta_list))
            np.save(file_name_delta,all_delta)

            all_delta2 = np.load(file_name_function)
            all_delta2 = np.vstack((all_delta2, function_list))
            np.save(file_name_function,all_delta2)

            all_delta3 = np.load(file_name_budget)
            all_delta3 = np.vstack((all_delta3, budget_list))
            np.save(file_name_budget,all_delta3)
        else:
            np.save(file_name_delta,delta_list)
            np.save(file_name_function,function_list)
            np.save(file_name_budget,budget_list)

        return recommended_solns, intermediate_budgets