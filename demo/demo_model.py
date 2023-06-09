"""
This script is intended to help with debugging an model.
It imports an model, initializes an model object with given factors,
sets up pseudorandom number generators, and runs one or more replications.
"""

import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

# Import random number generator.
from rng.mrg32k3a import MRG32k3a

# Import model.
# Replace <filename> with name of .py file containing model class.
# Replace <model_class_name> with name of model class.
# Ex: from models.mm1queue import MM1Queue
from models.<filename> import <model_class_name>

# Fix factors of model. Specify a dictionary of factors.
# Look at Model class definition to get names of factors.
# Ex: for the MM1Queue class,
#     fixed_factors = {"lambda": 3.0,
#                      "mu": 8.0}
fixed_factors = {}  # Resort to all default values.

# Initialize an instance of the specified model class.
# Replace <model_class_name> with name of model class.
# Ex: mymodel = MM1Queue(fixed_factors)
mymodel = <model_class_name>(fixed_factors)

# Working example for MM1 model. (Commented out)
# -----------------------------------------------
# from models.mm1queue import MM1Queue
# fixed_factors = {"lambda": 3.0, "mu": 8.0}
# mymodel = MM1Queue(fixed_factors)
# -----------------------------------------------

# The rest of this script requires no changes.

# Check that all factors describe a simulatable model.
# Check fixed factors individually.
for key, value in mymodel.factors.items():
    print(f"The factor {key} is set as {value}. Is this simulatable? {bool(mymodel.check_simulatable_factor(key))}.")
# Check all factors collectively.
print(f"Is the specified model simulatable? {bool(mymodel.check_simulatable_factors())}.")

# Create a list of RNG objects for the simulation model to use when
# running replications.
rng_list = [MRG32k3a(s_ss_sss_index=[0, ss, 0]) for ss in range(mymodel.n_rngs)]

# Run a single replication of the model.
responses, gradients = mymodel.replicate(rng_list)
print("\nFor a single replication:")
print("\nResponses:")
for key, value in responses.items():
    print(f"\t {key} is {value}.")
print("\n Gradients:")
for outerkey in gradients:
    print(f"\tFor the response {outerkey}:")
    for innerkey, value in gradients[outerkey].items():
        print(f"\t\tThe gradient w.r.t. {innerkey} is {value}.")
