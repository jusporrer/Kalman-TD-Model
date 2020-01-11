# Contributors : Juliana Sporrer (juliana.sporrer.18@ucl.ac.uk)
# Dual Masters in Brain and Mind Sciences (UCL, ENS, & Sorbonne Univeristy)
# -------------------------------------------------------------------------
# References:
# Gershman, S.J., 2015. A Unifying Probabilistic View of Associative Learning.
# PLoS Computational Biology 11, 1–20.
# -------------------------------------------------------------------------

import numpy as np
from matplotlib import *
from pylab import *
import sys


# -------------------------------------------------------------------------
# PARAMETERS + INITIALIZATION
# -------------------------------------------------------------------------

prior_covariance = 1; #c
noise_variance = 1; #s
diffusion_variance = 0.01; #q
discount_factor = 0.98; #g
trace_decay = 0.985; #decay
learning_rate = 0.3; #lr
standard_TD = 0; #TD







# -----------------------------------
# Plot
# -----------------------------------
dashes = ['--', '-.', '-']
colors = ['black', 'grey']
