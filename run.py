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
from KalmanTD_model import *


[covariance_matrix, reward] = serial_overshadowing_stimulation()


# -----------------------------------
# Plot
# -----------------------------------
dashes = ['--', '-.', '-']
colors = ['black', 'grey']
labels = ["Stimulus X", "Stimulus Y"]
TD = [0, 1]
nb_stimuli = 4
title = ['Kalman TD Model', 'Standard TD Model']
ylabel = ['Value of (weight)', 'Value']
nb_plot = [221, 222]
color = [(0.2, 0.4, 0.6), (0.5, 0.5, 0.6)]

x = np.arange(len(labels))
fig1 = figure(figsize = (15,9))
fig1.suptitle('Serial recovery from overshadowing stimulation', fontsize=18)

for i in range(2):
    weights = Kalman_TD(covariance_matrix,reward, nb_stimuli, TD[i])
    print(weights)

    plt.subplot(nb_plot[i])
    plt.grid(linestyle=':', linewidth='0.9')
    plt.bar(x,[weights[-1,0],weights[-1, 1]], align='center', alpha=0.8, color=color[i])
    plt.xticks(x, labels, fontsize=11)
    plt.ylabel(ylabel[i], fontsize=11)
    plt.title(title[i], fontsize=15)
    plt.ylim(0,0.5)

plt.show()
