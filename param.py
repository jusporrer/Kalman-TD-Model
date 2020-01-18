# Contributors : Juliana Sporrer (juliana.sporrer.18@ucl.ac.uk)
# Dual Masters in Brain and Mind Sciences (UCL, ENS, & Sorbonne Univeristy)
# -------------------------------------------------------------------------
# References:
# Gershman, S.J., 2015. A Unifying Probabilistic View of Associative Learning.
# PLoS Computational Biology 11, 1–20.
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# PARAMETERS + INITIALIZATION
# -------------------------------------------------------------------------

nb_trials = 10
trial_length = 10
duration = 3
onset2 = 2
onset1 = 0
onset = 0

covariance = 1;
noise_variance = 1;
diffusion_variance = 0.01;
discount_factor = 0.98;
trace_decay = 0.985;
learning_rate = 0.3;
