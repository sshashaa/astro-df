"""
Summary
-------
ASTRODF

The solver progressively builds local models (linear) using interpolation on a set of points. The design set is designed by the algorithm AffPoints, which is used in the paper (Wild, S. M., Regis, R. G., and Shoemaker, C. A. (2008). ORBIT: Optimization by radial
basis function interpolation in trust-regions. SIAM Journal on Scientific Computing, 30(6):3197–3219)

"""
import math

from base import Solver
from numpy.linalg import pinv
from numpy.linalg import norm
import numpy as np
from math import log, ceil
import warnings

warnings.filterwarnings("ignore")


class ASTRODFORG(Solver):
    """
    Needed description
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
    rng_list : list of rng.MRG32k3a objects
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

    def __init__(self, name="ASTRODFORG", fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = {}
        self.name = name
        self.objective_type = "single"
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.gradient_needed = False
        self.specifications = {
            "crn_across_solns": {
                "description": "CRN across solutions?",
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
                "default": 15
            },
            "simple_solve": {
                "description": "subproblem solver with Cauchy point or the built-in solver? True - Cauchy point, False - built-in solver",
                "datatype": bool,
                "default": True
            },
            "criticality_select": {
                "description": "True - skip contraction loop if not near critical region, False - always run contraction loop",
                "datatype": bool,
                "default": True
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

    def CheckPoised_(self, X, Dist, theta1, maxdelta, xkin, radius, nmp, Intind, ModelIn):
        """Obtains additional affine indep points and generates geom-imp pts if nec

        Parameters
        ----------
        X      = [dbl] [nf-by-n] Matrix whose rows are points
        Dist   = [dbl] [nf-by-1] Vector of distances to X(xkin,:)
        theta1 = [dbl] Positive validity threshold
        maxdelta = [dbl] Maximum distance to look for points
        xkin   = [int] Index of current center
        radius = [dbl] Positive radius
        ModelIn = [int] [(n+1)-by-1] Indices in X of linearly independent points
        nmp     = [int] Number of Model points (# of nonzeros in ModelIn)
                            Note: nmp<n before we call this
        Intind  = [log] [1-by-nf] Logical indicating whether i is in AffIn


        Returns
        ----------
        ModelIn = [int] [(n+1)-by-1] Indices in X of linearly independent points
        nmp     = [int] Number of Model points (# of nonzeros in ModelIn)
                            Note: nmp<n before we call this
        Intind  = [log] [1-by-nf] Logical indicating whether i is in AffIn
        GPoints = [dbl] [n-by-n] Matrix of (n-nmp) points to be evaluated
        """

        ModelIn = np.array(ModelIn, dtype='int')

        nf, n = X.shape
        GPoints = np.zeros((n, n))

        if nmp != 0:
            R = (X[ModelIn[0:nmp], :] - np.repeat([X[xkin, :]], nmp, axis=0)) / radius  # The points we have so far
            Q, R = np.linalg.qr(R.T, mode='complete')  # Get QR of points so far
            R = np.atleast_2d(R)
            R = np.hstack((R, np.zeros((n, n - R.shape[1]))))
        else:
            Q = np.eye(n)
            R = np.zeros((n, n))

        for ind in np.arange(nf - 1, -1, -1):
            if (Intind[ind] == False) and (Dist[ind] <= maxdelta):
                D = (X[ind, :] - X[xkin, :]) / radius
                proj = np.linalg.norm(D.dot(Q[:, nmp:n]))  # [double] # Note that Q(:,nmp+1:n) is a matrix
                if (proj >= theta1):
                    nmp = nmp + 1
                    ModelIn[nmp - 1] = ind
                    Intind[ind] = True
                    if nmp == n:
                        return (ModelIn, nmp, Intind, GPoints)
                    R[:, nmp - 1] = (D.dot(Q)).T
                    for j in np.arange(n - 1, nmp - 1, -1):
                        G = self.planerot(R[[j - 1, j], nmp - 1])[0]
                        R[[j - 1, j], nmp - 1] = G.dot(R[[j - 1, j], nmp - 1])
                        Q[0:n, [j - 1, j]] = Q[0:n, [j - 1, j]].dot(G.T)

        # if you get out of this loop then nmp<n
        GPoints[0:n - nmp, :] = Q[:, nmp:n].T

        return (ModelIn, nmp, Intind, GPoints)

    def AffPoints2_(self, X, Dist, radius, theta1, xkin):
        """Obtains n linearly indep points of norm<=radius

        Parameters
        ----------
        X      = [dbl] [nf-by-n] Matrix whose rows are points
        Dist   = [dbl] [nf-by-1] Vector of distances to X(xkin,:)
        radius = [dbl] Positive radius
        theta1 = [dbl] Positive validity threshold
        xkin   = [int] Index of current center

        Returns
        -------
        AffIn  = [int] [(n+1)-by-1] Indices in X of linearly independent points
        valid  = [log] Logical set to 1 if valid within radius
        Modeld = [dbl] [1-by-n] Unit model-improving direction
        nmp    = [int] Number of aff indep points (# of nonzeros in AffIn)
        Intind = [log] [1-by-nf] Logical indicating whether i is in AffIn
        """
        nf, n = X.shape
        Modeld = np.zeros(n)  # Initialize for output
        AffIn = -1 * np.ones(n + 1)  # vector of integer indices (of size n+1 for output)
        Intind = np.zeros(nf, dtype=bool)  # vector of indicators for indices in AffIn
        nmp = 0  # number of initial l.i. points

        Q = np.eye(n)  # Get initial null space
        R = np.zeros((n, n))
        for ind in np.arange(nf - 1, -1, -1):
            if Dist[ind] <= radius:  # Only look at the nearby points
                D = (X[ind, :] - X[xkin, :]) / radius  # Current displacement
                proj = np.linalg.norm(D.dot(Q[:, nmp:n]))  # [double] # Note that Q(:,nmp+1:n) is a matrix
                if (proj >= theta1):  # add this index to AffIn
                    nmp = nmp + 1
                    AffIn[nmp - 1] = ind
                    Intind[ind] = True
                    if (nmp == n):
                        valid = True
                        return (AffIn, valid, Modeld, nmp, Intind)

                    # Update QR factorization:
                    R[:, nmp - 1] = D.dot(Q)  # add D
                    for k in np.arange(n - 1, nmp - 1, -1):
                        G, R[[k - 1, k], nmp - 1] = self.planerot(R[[k - 1, k], nmp - 1])
                        Q[:, [k - 1, k]] = Q[:, [k - 1, k]].dot(G.T)
        # if you get out of this loop then nmp<n
        Modeld = Q[:, nmp:n].T
        valid = False
        return (AffIn, valid, Modeld, nmp, Intind)


    def ORBIT_model(self, X, F, N, Dist, Intind, n, delta, xkin, nfmax, nfs, nf, Low, Upp):
        # Set trust-region RBF algorithm parameters and initialize output
        c1 = 1  # Factor for checking validity
        # c2 = maxdelta # Maximum distance for adding points
        c2 = delta  # Maximum distance for adding points
        theta1 = 0.001  # Pivot threshold for validity

        trnorm = 2  # Type f trust-region norm [0]

        # STEP 1: Find affinely independent points & check if fully linear
        AffIn, valid, Modeld, nmp, Intind[0:nfs + nf] = self.AffPoints2_(X[0:nfs + nf, :], Dist, c1 * delta, theta1,
                                                                    xkin)


        if (not valid):  # Model is not valid, check if poised
            AffIn, nmp, Intind[0:nfs + nf], GPoints = self.CheckPoised_(X[0:nfs + nf, :], Dist, theta1, c2, xkin,
                                                                   c1 * delta, nmp, Intind[0:nfs + nf], AffIn)
            if nmp < n:  # Need to include additional points to obtain a model
                T1 = np.zeros(n - nmp)
                T2 = np.zeros(n - nmp)
                for j in range(0, n - nmp):
                    GPoints[j, :] = GPoints[j, :] / np.linalg.norm(GPoints[j, :], ord=trnorm)  # Make unit length.
                    T1[j] = self.boxline_(GPoints[j, :], X[xkin, :], Low, Upp)
                    T2[j] = self.boxline_(- GPoints[j, :], X[xkin, :], Low, Upp)
                if min(np.max(np.vstack((T1, T2)), axis=0)) < theta1 * delta * c1:
                    for j in range(0, min(n, nfmax - nf)):
                        t1 = Low[j] - X[xkin, j]
                        t2 = Upp[j] - X[xkin, j]
                        if t2 > - t1:
                            t1 = min(t2, delta)
                        else:
                            t1 = max(t1, - delta)
                        nf = nf + 1
                        X[nfs + nf, :] = X[xkin, :]
                        X[nfs + nf, j] = max(Low[j], min(X[xkin, j] + t1,
                                                         Upp[j]))  # added min and max to make sure in bounds
                        Dist[nfs + nf, 1] = abs(t1)
                else:  # Safe to use our directions:
                    for j in range(0, min(n - nmp, nfmax - nf)):
                        if T1[j] >= theta1 * delta * c1:
                            X[nfs + nf, :] = self.boxproj_(X[xkin, :] + min(T1[j], delta) * GPoints[j, :], n, Low,
                                                      Upp)  # added projection to make sure in bounds
                        elif T2[j] >= theta1 * delta * c1:
                            X[nfs + nf, :] = self.boxproj_(X[xkin, :] - min(T2[j], delta) * GPoints[j, :], n, Low,
                                                      Upp)  # added projection to make sure in bounds
                        Dist[nfs + nf] = np.linalg.norm(X[nfs + nf, :] - X[xkin, :], ord=trnorm)
                        for i in range(0, n + 1):
                            if AffIn[i] == -1:
                                AffIn[i] = nfs + nf
                                break
                        nf = nf + 1
        nmp = nmp + 1

        AffIn[1:n + 1] = AffIn[0:n]
        AffIn[0] = xkin
        Intind[xkin] = True
        return X, AffIn, Dist, nfs, nf

    def planerot(self, x):
        if x[1] != 0:
            r = np.linalg.norm(x)
            G = np.vstack((x, np.array([-x[1], x[0]]))) / r
            x = np.array([r, 0])
        else:
            G = np.eye(2)
        return (G, x)

    def boxline_(self, D, X, L, U):
        """ This routine finds the smallest t>=0 for which X+t*D hits the box [L,U]
        Parameters
        ----------
        D      = [dbl] [n-by-1] Direction
        L      = [dbl] [n-by-1] Lower bounds
        X      = [dbl] [n-by-1] Current Point (assumed to live in [L,U])
        U      = [dbl] [n-by-1] Upper bounds

        Returns
        ----------
        t      = [dbl] Value of smallest t>=0 for which X+t*D hits a constraint. Set to 1 if t=1 does not hit constraint for t<1.
        """
        n = X.shape[0]
        t = 1
        for i in range(0, n):
            if D[i] > 0:
                t = max(t, (U[i] - X[i]) / D[i])
            else:
                if D[i] < 0:
                    t = max(t, (L[i] - X[i]) / D[i])
        return t

    def boxproj_(self, z, p, l, u):
        """ This subroutine projects the vector z onto the box [l,u]

        z,l,u are vectors of length p
        """

        z = z.flatten()
        for i in range(0, p):
            z[i] = min(max(l[i], z[i]), u[i])
        return z
    ########################### ABOVE ORBIT ############################################################################

    def get_standard_basis(self, size, index):
        arr = np.zeros(size)
        arr[index] = 1.0
        return arr

    def evaluate_model(self, x_k, q):
        X = [1]
        X = np.append(X, np.array(x_k))
        return np.matmul(X, q)

    def coefficient(self, Y, fval, dim):
        M = []
        for i in range(0, dim + 1):
            M.append(1)
            M[i] = np.append(M[i], np.array(Y[i]))
        q = np.matmul(np.linalg.pinv(M), fval)
        grad = q[1:dim + 1]
        grad = np.reshape(grad, dim)

        return q, grad

    def get_stopping_time(self, k, sig2, delta, kappa, dim):
        if kappa == 0:
            kappa = 1

        lambda_min = self.factors["lambda_min"]
        lambda_k = max(lambda_min, 2 * log(dim,10))
        # compute sample size
        N_k = ceil(max(lambda_k, lambda_k * sig2 / ((kappa ** 2) * delta ** 4)))
        return N_k

    def construct_model(self, x_k, delta, k, problem, expended_budget, kappa, new_solution, X, S, N, Dist, Intind, xkin, nfmax, nfs, nf, Low, Upp):
        interpolation_solns = []
        w = self.factors["w"]
        mu = self.factors["mu"]
        beta = self.factors["beta"]
        criticality_select = self.factors["criticality_select"]
        criticality_threshold = self.factors["criticality_threshold"]
        j = 0
        d = problem.dim
        budget = problem.factors["budget"]

        for i in range(nfs+nf):
            Dist[i] = np.linalg.norm(X[i,:] - X[xkin,:],ord=2)

       # while True:
        fval = []
        Y = []
        j = j + 1
        delta_k = delta * w ** (j - 1)

        # construct the interpolation set
        X, AffIn, Dist, nfs, nf = self.ORBIT_model(X, S, N, Dist, Intind, d, delta_k, xkin, nfmax, nfs, nf, Low, Upp)

        # Update the distance matrix
        for i in range(nfs + nf):
            Dist[i] = np.linalg.norm(X[i, :] - X[xkin, :], ord=2)
        for i in AffIn:
            i = int(i)
            # For X_0, we don't need to simulate the system
            if (k == 1) and (i == 0):
                fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                interpolation_solns.append(new_solution)

            # Otherwise, we need to simulate the system
            elif N[i] != 0:
                new_solution = S[i][0]
                # Adaptive sampling
                while True:
                    problem.simulate(new_solution, 1)
                    sig2 = new_solution.objectives_var
                    expended_budget += 1
                    N[i] += 1
                    if N[i] >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget:
                        break
                S[i] = [new_solution]
                fval.append(-1 * problem.minmax[0] * S[i][0].objectives_mean)
                interpolation_solns.append(S[i][0])
            else:
                new_solution = self.create_new_solution(tuple(X[i]), problem)
                # check if there is existing result
                problem.simulate(new_solution, 1)
                expended_budget += 1
                sample_size = 1

                # Adaptive sampling
                while True:
                    problem.simulate(new_solution, 1)
                    expended_budget += 1
                    sample_size += 1
                    sig2 = new_solution.objectives_var
                    if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa,
                                                             problem.dim) or expended_budget >= budget:
                        break
                S[i] = [new_solution]
                N[i] = sample_size
                fval.append(-1 * problem.minmax[0] * new_solution.objectives_mean)
                interpolation_solns.append(new_solution)
            Y.append(X[i] - X[xkin])
            #F_Y.append(new_solution.objectives_mean)

        # construct the model and obtain the model coefficients
        q, grad = self.coefficient(Y, fval, d)
        delta_k = min(max(beta * norm(grad), delta_k), delta)
        return fval, Y, q, grad, delta_k, expended_budget, interpolation_solns, X, S, Dist, Intind, xkin, nfmax, nfs, nf, AffIn

    def tune_parameters(self, delta, delta_max, X, S, N, Intind, Dist, problem, xkin, nfmax, nfs, nf, Low, Upp):  # use the delta_max determined in the solve(...) function
        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0
        k = 0  # iteration number

        # default values
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        simple_solve = self.factors["simple_solve"]
        lambda_min = self.factors["lambda_min"]

        budget = problem.factors["budget"]

        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(tuple(new_x), problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)
        delta_k = delta
        kappa = 1

        while expended_budget < budget * 0.01:
            # calculate kappa
            k += 1
            if k == 1:
                sample_size_lower_bound = ceil(max(lambda_min, .5 * problem.dim) - 1)
                problem.simulate(new_solution, sample_size_lower_bound)
                expended_budget += sample_size_lower_bound
                sample_size = sample_size_lower_bound
                while True:
                    problem.simulate(new_solution, 1)
                    expended_budget += 1
                    sample_size += 1
                    fn = new_solution.objectives_mean
                    sig2 = new_solution.objectives_var
                    if sample_size >= self.get_stopping_time(k, sig2, delta, fn / (delta ** 2),
                                                             problem.dim) or expended_budget >= budget * 0.01:
                        kappa = fn / (delta ** 2)
                        N[0] = sample_size
                        S[0] = [new_solution]
                        break

            fval, Y, q, grad, delta_k, expended_budget, interpolation_solns, X, S, Dist, Intind, xkin, nfmax, nfs, nf, AffIn = self.construct_model(new_x, delta_k, k, problem,
                                                                                                                                            expended_budget, kappa, new_solution, X, S, N, Dist, Intind,
                                                                                                                                            xkin, nfmax, nfs, nf, Low, Upp)


            # Cauchy reduction
            candidate_x = new_x - delta * grad / norm(grad)

            for i in range(problem.dim):
                if candidate_x[i] <= problem.lower_bounds[i]:
                    candidate_x[i] = problem.lower_bounds[i] + 0.01
                elif candidate_x[i] >= problem.upper_bounds[i]:
                    candidate_x[i] = problem.upper_bounds[i] - 0.01

            X[nfs + nf, :] = candidate_x
            candidate_xkin = nfs+nf
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            # pilot run
            problem.simulate(candidate_solution, 1)
            expended_budget += 1
            sample_size = 1
            # adaptive sampling
            while True:
                problem.simulate(candidate_solution, 1)
                expended_budget += 1
                sample_size += 1
                sig2 = candidate_solution.objectives_var
                if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget * 0.01:
                    S[nfs+nf] = candidate_solution
                    N[nfs+nf] = sample_size
                    nf += 1
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            if (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model
                (np.array(candidate_x) - np.array(new_x), q)) == 0:
                rho = 0
            else:
                rho = (fval[0] - fval_tilde) / \
                      (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(
                          candidate_x - new_x, q))

            if rho >= eta_2:  # very successful
                new_x = candidate_x
                xkin = candidate_xkin
                final_ob = candidate_solution.objectives_mean
                delta_k = min(gamma_1 * delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            elif rho >= eta_1:  # successful
                new_x = candidate_x
                xkin = candidate_xkin
                final_ob = candidate_solution.objectives_mean
                delta_k = min(delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            else:
                delta_k = min(gamma_2 * delta_k, delta_max)
                final_ob = fval[0]

        return final_ob, k, delta_k, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, X, S, Dist,Intind, xkin, nfmax, nfs, nf

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

        recommended_solns = []
        intermediate_budgets = []
        expended_budget = 0

        # default values
        eta_1 = self.factors["eta_1"]
        eta_2 = self.factors["eta_2"]
        gamma_1 = self.factors["gamma_1"]
        gamma_2 = self.factors["gamma_2"]
        simple_solve = self.factors["simple_solve"]


        k = 0  # iteration number

        # Start with the initial solution
        new_x = problem.factors["initial_solution"]
        new_solution = self.create_new_solution(tuple(new_x), problem)
        recommended_solns.append(new_solution)
        intermediate_budgets.append(expended_budget)

        # Data Structure - ORBIT
        xkin = 0
        nf = 0
        nfmax = 10000
        nfs = 1
        Low = problem.lower_bounds
        Upp = problem.upper_bounds

        X = np.array([new_x])
        S = np.array([new_solution])

        X = np.vstack((X, np.zeros((nfmax, problem.dim))))  # Stores the evaluation point locations
        S = np.vstack((S, np.zeros((nfmax, 1))))  # Stores the function mean values of evaluated points
        N = np.zeros(nfs + nfmax)
        Dist = np.ones((nfs + nfmax)) * 1000  # Stores displacement distances
        Intind = np.zeros(nfs + nfmax, dtype=bool)  # Stores indicators for model points


        # Parameter tuning run
        tp_final_ob_pt, k, delta, recommended_solns, intermediate_budgets, expended_budget, new_x, kappa, X, S, Dist, Intind, xkin, nfmax, nfs, nf = self.tune_parameters(
            delta_candidate[0], delta_max, X, S, N, Intind, Dist, problem, xkin, nfmax, nfs, nf, Low, Upp)

        delta_k = delta

        while expended_budget < budget:
            k += 1

            fval, Y, q, grad, delta_k, expended_budget, interpolation_solns, X, S, Dist, Intind, xkin, nfmax, nfs, nf, AffIn = self.construct_model(
                new_x, delta_k, k, problem, expended_budget, kappa, new_solution, X, S, N, Dist, Intind, xkin, nfmax, nfs, nf, Low, Upp)

            candidate_x = new_x - delta * grad / norm(grad)

            for i in range(problem.dim):
                if candidate_x[i] <= problem.lower_bounds[i]:
                    candidate_x[i] = problem.lower_bounds[i] + 0.01
                elif candidate_x[i] >= problem.upper_bounds[i]:
                    candidate_x[i] = problem.upper_bounds[i] - 0.01

            X[nfs + nf, :] = candidate_x
            candidate_xkin = nfs + nf
            candidate_solution = self.create_new_solution(tuple(candidate_x), problem)

            # pilot run
            problem.simulate(candidate_solution, 1)
            expended_budget += 1
            sample_size = 1

            # Adaptive sampling
            while True:
                problem.simulate(candidate_solution, 1)
                expended_budget += 1
                sample_size += 1
                sig2 = candidate_solution.objectives_var
                if sample_size >= self.get_stopping_time(k, sig2, delta_k, kappa, problem.dim) or expended_budget >= budget:
                    S[nfs+nf] = candidate_solution
                    N[nfs+nf] = sample_size
                    nf += 1
                    break

            # calculate success ratio
            fval_tilde = -1 * problem.minmax[0] * candidate_solution.objectives_mean

            if (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(np.array(candidate_x) - np.array(new_x), q)) == 0:
                rho = 0
            else:
                rho = (fval[0] - fval_tilde) / (self.evaluate_model(np.zeros(problem.dim), q) - self.evaluate_model(candidate_x - new_x, q))

            if rho >= eta_2:  # very successful
                new_x = candidate_x
                xkin = candidate_xkin
                delta_k = min(gamma_1 * delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            elif rho >= eta_1:  # successful
                new_x = candidate_x
                xkin = candidate_xkin
                delta_k = min(delta_k, delta_max)
                recommended_solns.append(candidate_solution)
                intermediate_budgets.append(expended_budget)
            else:
                delta_k = min(gamma_2 * delta_k, delta_max)

        return recommended_solns, intermediate_budgets

