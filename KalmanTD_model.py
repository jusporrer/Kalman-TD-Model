# Contributors : Juliana Sporrer (juliana.sporrer.18@ucl.ac.uk)
# Dual Masters in Brain and Mind Sciences (UCL, ENS, & Sorbonne Univeristy)
# -------------------------------------------------------------------------
# References:
# Gershman, S.J., 2015. A Unifying Probabilistic View of Associative Learning.
# PLoS Computational Biology 11, 1–20.
# -------------------------------------------------------------------------

import numpy as np
import numpy.matlib
from functions import *

# -----------------------------------
# Kalman TD & Standard TD Model
# -----------------------------------

def Kalman_TD(covariance_matrix, reward_matrix, nb_stimuli, trial_length, TD, covariance, diffusion_variance, discount_factor, noise_variance, learning_rate):
    matrix_nb_columns = np.size(covariance_matrix,1)
    matrix_nb_rows = np.size(covariance_matrix,0)

    covariance_matrix = np.asmatrix(np.concatenate((covariance_matrix, np.zeros( (1,(nb_stimuli * trial_length)))), axis=0)) #adds buffer

    weights = np.asmatrix(np.zeros((matrix_nb_columns,1)))
    weights_matrix = np.asmatrix(np.zeros((matrix_nb_rows,matrix_nb_columns)))

    covariance = covariance*np.eye(matrix_nb_columns)
    diffusion_variance = diffusion_variance*np.eye(matrix_nb_columns)

    for i in range(matrix_nb_rows):

        values = covariance_matrix[i,:]*weights

        weights_matrix[i,::] = weights.flatten()

        temporal_difference = covariance_matrix[i,:] - discount_factor*covariance_matrix[i+1,:]

        rhat = temporal_difference*weights

        prediction_error = reward_matrix[0,i] - rhat

        covariance = covariance + diffusion_variance

        residual_covariance = temporal_difference*covariance*temporal_difference.transpose() + noise_variance

        kalman_gain = covariance*temporal_difference.transpose() / residual_covariance

        if TD == 1:
            weights = weights + learning_rate*temporal_difference.transpose()*prediction_error
        else:
            weights = weights + kalman_gain*prediction_error

        covariance = covariance - kalman_gain*temporal_difference*covariance

    return weights_matrix

# -----------------------------------
# Serial overshadowing Simulation
# -----------------------------------

class serial_overshadowing_simulation():
    """ Simulates the recovery from serial compount overshadowing (with second-order extinction)
            First, creates stimuli.
            Second, creates covariance and reward matrices.
            Third, applies Kalman-TD and standard-TD models.
    """

    def __init__(self, nb_stimuli, nb_trials, trial_length, onset1, onset2, duration, covariance, noise_variance, diffusion_variance, discount_factor, trace_decay, learning_rate):
        self.nb_stimuli = nb_stimuli
        self.nb_trials = nb_trials
        self.trial_length = trial_length
        self.onset1 = onset1
        self.onset2 = onset2
        self.duration = duration
        self.covariance = covariance
        self.noise_variance = noise_variance
        self.diffusion_variance = diffusion_variance
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        self.learning_rate = learning_rate

    def initialise_stimuli(self):
        """
        Creates the stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        stimuli_1 = np.zeros((self.trial_length, self.nb_stimuli),dtype = np.int)
        stimuli_1[:,0] = create_stimuli(self.onset1, self.duration, self.trial_length)
        stimuli_1[:,1] = create_stimuli(self.onset2, self.duration, self.trial_length)

        # B -> Y -> +
        stimuli_2 = np.zeros((self.trial_length, self.nb_stimuli), dtype = np.int)
        stimuli_2[:,2] = create_stimuli(self.onset1, self.duration, self.trial_length)
        stimuli_2[:,3] = create_stimuli(self.onset2, self.duration, self.trial_length)

        # A -> -
        stimuli_3 = np.zeros((self.trial_length, self.nb_stimuli),dtype = np.int)
        stimuli_3[:,0] = create_stimuli(self.onset1, self.duration, self.trial_length)

        return stimuli_1, stimuli_2, stimuli_3

    def initialise_covariance_matrix(self, stimuli_1, stimuli_2, stimuli_3):
        """
        Creates the covariance matrix for each stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        matrix_stimuli_1 = construct_covariance_matrix(stimuli_1, self.trial_length, self.nb_stimuli)

        # B -> Y -> +
        matrix_stimuli_2 = construct_covariance_matrix(stimuli_2, self.trial_length, self.nb_stimuli)

        # A -> -
        matrix_stimuli_3 = construct_covariance_matrix(stimuli_3, self.trial_length, self.nb_stimuli)

        # All the stimuli
        covariance_matrix = np.concatenate((np.matlib.repmat( (matrix_stimuli_1), self.nb_trials, 1), np.matlib.repmat( (matrix_stimuli_2), self.nb_trials, 1)))
        covariance_matrix = np.concatenate((covariance_matrix, np.matlib.repmat( (matrix_stimuli_3), self.nb_trials, 1)))

        return covariance_matrix

    def initialise_reward_matrix(self, stimuli_1, stimuli_2, stimuli_3):
        """
        Creates the reward matrix for each stimulus presented together (or alone) with second-order stimulus extinction.
        """
        # A -> X -> +
        reward_stimuli_1 = create_stimuli((self.onset2 + self.duration - 1), 1, self.trial_length)

        # B -> Y -> +
        reward_stimuli_2 = create_stimuli((self.onset2 + self.duration - 1), 1, self.trial_length)

        # A -> -
        reward_stimuli_3 = np.zeros((self.trial_length), dtype = np.int)

        # All the stimuli
        reward_matrix = np.asmatrix( np.append( np.matlib.repmat( (reward_stimuli_1, reward_stimuli_2), self.nb_trials, 1), np.matlib.repmat(reward_stimuli_3, self.nb_trials, 1)))

        return reward_matrix

    def Kalman_TD(self, covariance_matrix, reward_matrix):
        """
        Creates the Kalman-TD weight matrix for second-order stimulus extinction simulation.
        """
        weights_matrix_kalmanTD = Kalman_TD(covariance_matrix, reward_matrix, self.nb_stimuli, self.trial_length, 0, self.covariance, self.diffusion_variance, self.discount_factor, self.noise_variance, self.learning_rate)

        weights_kalmanTD = weights_matrix_kalmanTD[self.onset2::self.trial_length, (self.trial_length + self.onset2)]
        weights_kalmanTD = np.concatenate((weights_kalmanTD, weights_matrix_kalmanTD[self.onset2::self.trial_length, ((self.nb_stimuli-1)*self.trial_length + self.onset2)]), axis=1)

        return weights_kalmanTD

    def standard_TD(self, covariance_matrix, reward_matrix):
        """
        Creates the standard-TD weight matrix for second-order stimulus extinction simulation.
        """

        weights_matrix_TD = Kalman_TD(covariance_matrix, reward_matrix, self.nb_stimuli, self.trial_length, 1, self.covariance, self.diffusion_variance, self.discount_factor, self.noise_variance, self.learning_rate)

        weights_TD = weights_matrix_TD[self.onset2::self.trial_length, (self.trial_length + self.onset2)]
        weights_TD = np.concatenate((weights_TD, weights_matrix_TD[self.onset2::self.trial_length, ((self.nb_stimuli-1)*self.trial_length + self.onset2)]), axis=1)

        return weights_TD
