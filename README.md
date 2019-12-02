[Kalman-Filter](https://jusporrer.github.io/PCBS-Kalman-Filter/)
================

This program in Python allows to fit the **Kalman Filter Rescorla-Wagner Model** to behavioral data.

The initial code in Matlab is based on the previous work of **Samuel Gershman**, at the Department of Psychology and Center for Brain Science, Harvard University [(See his GitHub here)](https://github.com/sjgershm/KF-learning). This model is introduced in his paper [*A Unifying Probabilistic View of Associative Learning*](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1004567&type=printable) which describes a framework encompassing Bayesian and reinforcement learning theories of associative learning. 

For more information please contact me at <juliana.sporrer.18@ucl.ac.uk>.


### Table of Contents

1. [Kalman-Filter Rescorla-Wagner Model](#Kalman-Filter)
    1. [What is the Kalman Filter?](#What-is-the-Kalman-Filter-?)
    1. [Conclusion](#Conclusion)
    1. [PBCS Feedback](#Programming-for-Brain-and-Cognitive-Sciences)



## What is the Kalman Filter ? 

### Associative Learning 

The ability to learn is essential to the survival of animals. Two recent concepts have allowed us to have a better understanding of how this learning is occurring. 
- An agent estimates the strength of associations and tracks its uncertainty using **Bayesian principles** and embodied by the **Kalman Filter**. 
- An agent learns about long-term cumulative future reward using **Reinforcement Learning (RL) principles** and represented by **Temporal Difference, TD**. 

Bayesian and RL theories are derived from different, but not exclusive, assumptions about the target and uncertainty representation of the learning task. The Kalman-filter learns the posterior distribution of expected immediate reward in contrast to the TD that learns a single value of expected future reward.   

However, these two theoretical models can be brought together in the form of the **Kalman-TD Model**. 


<img src="https://github.com/jusporrer/PCBS-Kalman-Filter/blob/master/FIG1.JPG" alt="alt text" width="186.8" height="123.2">
 



## Conclusion 

## Programming for Brain and Cognitive Sciences

This code was part of the course PBCS from the Cogmaster at the ENS (Paris, France). 






