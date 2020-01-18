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
import param


nb_trials = param.nb_trials
trial_length = param.trial_length
duration = param.duration
onset2 = param.onset2
onset1 = param.onset1
onset = param.onset

def serial_overshadowing_stimulation():
    nb_stimuli = 4

    # A -> X -> +
    x1 = np.zeros((trial_length, nb_stimuli),dtype = np.int)
    x1[:,0] = create_stimuli(onset1, duration, trial_length)
    x1[:,1] = create_stimuli(onset2, duration, trial_length)
    matrix_x1 = construct_covariance_matrix(x1, trial_length, nb_stimuli)
    reward_x1 = create_stimuli((onset2 + duration - 1), 1, trial_length)

    # B -> Y -> +
    x2 = np.zeros((trial_length, nb_stimuli), dtype = np.int)
    x2[:,2] = create_stimuli(onset1, duration, trial_length)
    x2[:,3] = create_stimuli(onset2, duration, trial_length)
    matrix_x2 = construct_covariance_matrix(x2, trial_length, nb_stimuli)
    reward_x2 = create_stimuli((onset2 + duration - 1), 1, trial_length)

    # A -> -
    x3 = np.zeros((trial_length, nb_stimuli),dtype = np.int)
    x3[:,0] = create_stimuli(onset1, duration, trial_length)
    matrix_x3 = construct_covariance_matrix(x3, trial_length, nb_stimuli)
    reward_x3 = np.zeros((trial_length), dtype = np.int)

    covariance_matrix = np.concatenate((np.matlib.repmat( (matrix_x1), nb_trials, 1), np.matlib.repmat( (matrix_x2), nb_trials, 1)), axis=0)
    covariance_matrix = np.concatenate((covariance_matrix, np.matlib.repmat( (matrix_x3), nb_trials, 1)), axis=0)

    reward = np.asmatrix( np.append( np.matlib.repmat( (reward_x1, reward_x2), nb_trials, 1), np.matlib.repmat(reward_x3, nb_trials, 1)))

    return covariance_matrix, reward


def Kalman_TD(covariance_matrix, reward, nb_stimuli, TD):
    matrix_nb_columns = np.size(covariance_matrix,1)
    matrix_nb_rows = np.size(covariance_matrix,0)

    covariance_matrix = np.asmatrix(np.concatenate((covariance_matrix, np.zeros( (1,(nb_stimuli * trial_length)))), axis=0))

    weights = np.asmatrix(np.zeros((matrix_nb_columns,1)))
    weights_matrix = np.asmatrix(np.zeros((matrix_nb_rows,matrix_nb_columns)))

    covariance = param.covariance*np.eye(matrix_nb_columns)
    diffusion_variance = param.diffusion_variance*np.eye(matrix_nb_columns)

    for i in range(matrix_nb_rows):

        values = covariance_matrix[i,:]*weights

        weights_matrix[i,::] = weights.flatten()

        temporal_difference = covariance_matrix[i,:] - param.discount_factor*covariance_matrix[i+1,:]

        rhat = temporal_difference*weights

        prediction_error = reward[0,i] - rhat

        covariance = covariance + diffusion_variance

        residual_covariance = temporal_difference*covariance*temporal_difference.transpose() + param.noise_variance

        kalman_gain = covariance*temporal_difference.transpose() / residual_covariance

        if TD == 1:
            weights = weights + param.learning_rate*temporal_difference.transpose()*prediction_error
        else:
            weights = weights + kalman_gain*prediction_error

        covariance = covariance - kalman_gain*temporal_difference*covariance

    weights = weights_matrix[param.onset2::param.trial_length, (param.trial_length + param.onset2)]
    weights = np.concatenate((weights, weights_matrix[param.onset2::param.trial_length, (3*param.trial_length + param.onset2)]), axis=1)

    #return weights, values, kalman_gain, covariance, temporal_difference, rhat
    return weights


[covariance_matrix, reward] = serial_overshadowing_stimulation()
weights_matrix = Kalman_TD(covariance_matrix,reward,4, 0)

print(weights_matrix)
