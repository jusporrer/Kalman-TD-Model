# Contributors : Juliana Sporrer (juliana.sporrer.18@ucl.ac.uk)
# Dual Masters in Brain and Mind Sciences (UCL, ENS, & Sorbonne Univeristy)
# -------------------------------------------------------------------------
# References:
# Gershman, S.J., 2015. A Unifying Probabilistic View of Associative Learning.
# PLoS Computational Biology 11, 1–20.
# -------------------------------------------------------------------------

import numpy as np

def create_stimuli(onset, duration, trial_length):

    stimuli = np.zeros((trial_length), dtype = np.int)

    if onset >= 0:
        stimuli[onset:(onset + duration)] = 1
        return stimuli

def construct_covariance_matrix(stimuli, trial_length, nb_stimuli):

    covariance_matrix = np.zeros((trial_length,nb_stimuli*10), dtype = np.int)

    start=0
    finish=1
    for i in range(nb_stimuli):

        covariance_matrix[::,start*trial_length:finish*trial_length] = np.diag(stimuli[:,i])

        start += 1
        finish += 1

    return covariance_matrix
