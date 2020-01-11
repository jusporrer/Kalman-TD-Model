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
import numpy.matlib
from functions import *
import run

nb_stimuli = 4
nb_trials = 10
trial_length = 10
duration = 3
onset2 = 2
onset1 = 0
onset = 0

def serial_overshadowing_stimulation():

    # A -> X -> +
    x1 = np.zeros((trial_length, nb_stimuli),dtype = np.int)
    x1[:,0] = create_stimulus(onset1, duration, trial_length)
    x1[:,1] = create_stimulus(onset2, duration, trial_length)
    matrix_x1 = construct_matrix(x1)
    reward_x1 = create_stimulus((onset2 + duration - 1), 1, trial_length)

    # B -> Y -> +
    x2 = np.zeros((trial_length, nb_stimuli), dtype = np.int)
    x2[:,2] = create_stimulus(onset1, duration, trial_length)
    x2[:,3] = create_stimulus(onset2, duration, trial_length)
    matrix_x2 = construct_matrix(x2)
    reward_x2 = create_stimulus((onset2 + duration - 1), 1, trial_length)

    # A -> -
    x3 = np.zeros((trial_length, nb_stimuli),dtype = np.int)
    x3[:,0] = create_stimulus(onset1, duration, trial_length)
    matrix_x3 = construct_matrix(x3)
    reward_x3 = np.zeros((trial_length, 1), dtype = np.int)

    covariance_matrix = np.append( np.matlib.repmat( (matrix_x1), nb_trials, 1), np.matlib.repmat(matrix_x2, nb_trials,1))
    covariance_matrix = np.append(covariance_matrix, np.matlib.repmat(matrix_x2, nb_trials,1))
    covariance_matrix = np.asmatrix(np.append(covariance_matrix, np.zeros((1,(nb_stimuli*trial_length)), dtype = np.int))) #add buffer
    covariance_matrix = np.reshape(covariance_matrix,(301,(nb_stimuli*trial_length)),order='F')

    #print(covariance_matrix)
    #print(np.size(covariance_matrix,0))
    #print(np.size(covariance_matrix,1))

    reward = np.asmatrix(np.append( np.matlib.repmat( (reward_x1, reward_x2), nb_trials, 1), np.matlib.repmat(reward_x3, nb_trials,1)))

    return (covariance_matrix, reward)



def simultaneous_overshadowing_stimulation ():
    print(1)


def Kalman_TD(covariance_matrix,reward):
    matrix_nb_columns = np.size(covariance_matrix,1)  #(buffer)
    matrix_nb_rows = np.size(covariance_matrix,0) - 1

    weights = np.zeros((matrix_nb_columns,1), dtype = np.int)

    prior_covariance = run.prior_covariance*np.eye(matrix_nb_columns)

    for i in range (matrix_nb_rows+1):
        noise_variance = np.zeros((i,1), dtype = np.int)+run.noise_variance
        diffusion_variance = np.zeros((i,1), dtype = np.int)+run.diffusion_variance
        learning_rate = np.zeros((i,1), dtype = np.int)+run.learning_rate

    for i in range(matrix_nb_rows):
        values = covariance_matrix[i,:]*weights
        print(i)
        print(weights)
        print(values)


        Diffusion_variance = diffusion_variance[i]*np.eye(matrix_nb_columns)
        temporal_difference = covariance_matrix[i,:] - run.discount_factor*covariance_matrix[i+1,:]
        rhat = temporal_difference*weights
        prediction_error = reward[0,i] - rhat
        prior_covariance = prior_covariance + Diffusion_variance
        residual_covariance = temporal_difference*prior_covariance*temporal_difference.transpose() + noise_variance[i]
        kalman_gain = prior_covariance*temporal_difference.transpose() / residual_covariance

        if run.standard_TD == 1:
            weights = weights + learning_rate[i]*temporal_difference.transpose()*prediction_error
        else:
            weights = weights + kalman_gain*prediction_error

        prior_covariance = prior_covariance - kalman_gain*temporal_difference*prior_covariance #posterior covariance influencing  prior covariance

    return
(covariance_matrix, reward) = serial_overshadowing_stimulation()
Kalman_TD(covariance_matrix,reward)
