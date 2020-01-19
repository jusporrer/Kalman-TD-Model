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
from KalmanTD_model import *

# -----------------------------------
# Gobal Simulation Parameters
# -----------------------------------

covariance = 1;
noise_variance = 1;
diffusion_variance = 0.01;
discount_factor = 0.98;
trace_decay = 0.985;
learning_rate = 0.3;

# --------------------------------------------------------------------------------
# Plot - Recovery from serial overshadowing after second order stimulus extinction
# --------------------------------------------------------------------------------

#local parameters
nb_stimuli = 4
nb_trials = 10
trial_length = 10
duration = 3
onset2 = 2
onset1 = 0

#launching simulation
second_order_extinction = serial_overshadowing_simulation(nb_stimuli, nb_trials, trial_length, onset1, onset2, duration, covariance, noise_variance, diffusion_variance, discount_factor, trace_decay, learning_rate)
[stimuli_1, stimuli_2, stimuli_3] = second_order_extinction.initialise_stimuli()
covariance_matrix = second_order_extinction.initialise_covariance_matrix(stimuli_1, stimuli_2, stimuli_3)
reward_matrix = second_order_extinction.initialise_reward_matrix(stimuli_1, stimuli_2, stimuli_3)
weights_kalmanTD = second_order_extinction.Kalman_TD(covariance_matrix, reward_matrix)
weights_TD = second_order_extinction.standard_TD(covariance_matrix, reward_matrix)
weights = [[weights_kalmanTD[-1,0],weights_kalmanTD[-1,1]], [weights_TD[-1,0], weights_TD[-1,1]]]

labels = ["Stimulus X", "Stimulus Y"]
title = ['Kalman TD Model', 'Standard TD Model']
nb_plot = [121, 122]
color = [(0.5, 0.7, 0.8), (0.8, 0.6, 0.7)]

x = np.arange(len(labels))
fig1 = figure(figsize = (15,9))
fig1.suptitle('Simulation: Recovery from Serial Compound Overshadowing after Second-Order Stimulus Extinction', fontsize=20)

for i in range(2):
    plt.subplot(nb_plot[i])
    plt.grid(linestyle=':', linewidth='0.9')
    plt.bar(x,weights[i], align='center', alpha=0.8, color=color[i], edgecolor=(0.5,0.5,0.5))
    plt.xticks(x, labels, fontsize=14)
    plt.ylabel('Value of (weight)', fontsize=14)
    plt.title(title[i], fontsize=16)
    plt.ylim(0,0.5)

plt.show()
fig1.savefig('fig_simulation_KalmanTD.pdf', bbox_inches='tight')
