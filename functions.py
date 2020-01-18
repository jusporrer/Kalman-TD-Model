# Contributors : Juliana Sporrer (juliana.sporrer.18@ucl.ac.uk)
# Dual Masters in Brain and Mind Sciences (UCL, ENS, & Sorbonne Univeristy)
# -------------------------------------------------------------------------
# References:
# Gershman, S.J., 2015. A Unifying Probabilistic View of Associative Learning.
# PLoS Computational Biology 11, 1–20.
# -------------------------------------------------------------------------

import sys
import os
import numpy as np

def create_stimulus(onset, duration, trial_length):

    x = np.zeros((trial_length), dtype = np.int)

    if onset >= 0:
        x[onset:(onset + duration)] = 1
        return x

def construct_matrix(x, trial_length, nb_stimuli):
    
    matrix = np.zeros((trial_length,nb_stimuli*10), dtype = np.int)

    start=0
    finish=1
    for i in range(nb_stimuli):

        matrix[::,start*trial_length:finish*trial_length] = np.diag(x[:,i])

        start += 1
        finish += 1

    return matrix
