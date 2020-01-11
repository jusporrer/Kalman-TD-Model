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

# -----------------------------------
# FONCTIONS
# -----------------------------------

def create_stimulus(onset, duration, trial_length):

    x = np.zeros((trial_length), dtype = np.int)

    if onset >= 0:
        x[onset:(onset + duration)] = 1
        return x

#x2 = np.zeros((10,4)) #nb of lines then nb of coloms
#x2[:,3] = create_stimulus(4,5,10)
#print(x2)


def construct_matrix(x):
    x_nb_columns = np.size(x,1)
    m = []

    for i in range(0, x_nb_columns):
        m = np.append(m,(np.diag(x[:,i])))

    matrix = np.asmatrix(m)
    matrix = np.reshape(m,(10,40),order='F')

    return matrix

#M = construct_matrix(x2)
#print(M)
#print(np.size(M,0))
#print(np.size(M,1))

#(covariance_matrix,reward)
#Kalman_TD_stimulation_serial_overshadowing()
