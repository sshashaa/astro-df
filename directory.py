#!/usr/bin/env python
"""
Summary
-------
Provide dictionary directories listing solvers, problems, and models.

Listing
-------
solver_directory : dictionary
problem_directory : dictionary
model_directory : dictionary
"""
# import solvers
from solvers.randomsearch import RandomSearch
from solvers.neldmd import NelderMead
from solvers.astrodfdh import ASTRODFDH
from solvers.astrodfrf import ASTRODFRF
from solvers.astrodfdh_problem import ASTRODFDH_problem
from solvers.astrodfrf_problem import ASTRODFRF_problem
from solvers.astrodforg import ASTRODFORG
from solvers.storm import STORM
from solvers.adam import ADAM
from solvers.aloe import ALOE
from solvers.strong import STRONG

# import models and problems
from models.cntnv import CntNV, CntNVMaxProfit
from models.mm1queue import MM1Queue, MM1MinMeanSojournTime
from models.facilitysizing import FacilitySize, FacilitySizingTotalCost, FacilitySizingMaxService
from models.rmitd import RMITD, RMITDMaxRevenue
from models.sscont import SSCont, SSContMinCost
from models.san import SAN, SANLongestPath
from models.ironore import IronOreMaxRevCnt
from models.paramesti import ParameterEstimation, ParamEstiMinLogLik
from models.dynamnews import DynamNews, DynamNewsMaxProfit
from models.synthetic import SYNTHETIC, SYNTHETIC_MIN
from models.rosenbrock import RosenbrockModel, RosenbrockProblem

# directory dictionaries
solver_directory = {
    "ASTRODFRF": ASTRODFRF,
    "ASTRODFRF_problem": ASTRODFRF_problem,
    "ASTRODFDH": ASTRODFDH,
    "ASTRODFDH_problem": ASTRODFDH_problem,
    "ASTRODFORG": ASTRODFORG,
    "STORM": STORM,
    "STRONG": STRONG,
    "RNDSRCH": RandomSearch,
    "NELDMD": NelderMead,
    "ADAM": ADAM,
    "ALOE": ALOE

}
problem_directory = {
    "CNTNEWS-1": CntNVMaxProfit,
    "MM1-1": MM1MinMeanSojournTime,
    "FACSIZE-1": FacilitySizingTotalCost,
    "FACSIZE-2": FacilitySizingMaxService,
    "RMITD-1": RMITDMaxRevenue,
    "SSCONT-1": SSContMinCost,
    "SAN-1": SANLongestPath,
    "IRONORECONT-1": IronOreMaxRevCnt,
    "PARAMESTI-1": ParamEstiMinLogLik,
    "DYNAMNEWS-1": DynamNewsMaxProfit,
    "SYN-1": SYNTHETIC_MIN,
    "RSBR-1": RosenbrockProblem
}
model_directory = {
    "CNTNEWS": CntNV,
    "MM1": MM1Queue,
    "FACSIZE": FacilitySize,
    "RMITD": RMITD,
    "SSCONT": SSCont,
    "SAN": SAN,
    "DYNAMNEWS": DynamNews,
    "PARAMESTI": ParameterEstimation,
    "SYN-1": SYNTHETIC,
    "RSBR-1": RosenbrockModel
}
