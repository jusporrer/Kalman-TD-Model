[Kalman-TD Model](https://jusporrer.github.io/PCBS-Kalman-Filter/)
================

This program in Python allows to fit the **Kalman-TD Model** to behavioural data.

The initial code in Matlab is based on the previous work of **Samuel Gershman**, at the Department of Psychology and Center for Brain Science, Harvard University [(See his GitHub here)](https://github.com/sjgershm/KF-learning). This model is introduced in his paper [*A Unifying Probabilistic View of Associative Learning*](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1004567&type=printable) which describes a framework encompassing Bayesian and Reinforcement Learning theories of associative learning.

For more information please contact me at <juliana.sporrer.18@ucl.ac.uk>.


## Table of Contents

1. [What is the Kalman TD Model?](#What-is-the-Kalman-TD-Model-?)
      1. [Associative Learning Theories](#Associative-Learning-Theories)
      1. [Unifying Model](#Unifying-Model)  
1. [How is the Kalman TD Model Build?](#How-is-the-Kalman-TD-Model-Build?)
1. [How to Test the Superiority of the Kalman TD Model?](#How-to-Test-the-Superiority-of-the-Kalman-TD-Model-?)
      1. [Recovery from Serial Compound Overshadowing](#Recovery-from-Serial-Compound-Overshadowing)
1. [How to Simulate the Kalman TD Model?](#How-to-Simulate-the-Kalman-TD-Model-?)
      1. [Create the Stimuli](#Create-the-Stimuli)
      1. [Create the Matrices](#Create-the-Matrices)
      1. [Build the Models](#Build-the-Models)
      1. [Plot the Results](#Plot-the-Results)
1. [Conclusion](#Conclusion)
1. [PBCS Feedback](#Programming-for-Brain-and-Cognitive-Sciences)


# What is the Kalman-TD Model?

## Associative Learning Theories

The ability to learn is essential to the survival of animals. Two recent concepts have allowed us to have a better understanding of how this learning is occurring.
Both theories can be seen as a generalisation of the seminal Rescorla-Wagner but are derived from different assumptions about the target and uncertainty representation of the learning task.
- An agent estimates the strength of associations and tracks its uncertainty using **Bayesian principles** and embodied by the **Kalman Filter (KF)**.
      - KF learns the *posterior distribution* of expected *immediate* reward
- An agent learns about long-term cumulative future reward using **Reinforcement Learning (RL) principles** and represented by **Temporal Difference (TD)**.
      - TD learns a *single value* of expected *future* cumulative reward

<img src="https://github.com/jusporrer/PCBS-Kalman-Filter/blob/master/FIG1.JPG" alt="alt text" width="186.8" height="123.2">

## Unifying Model

These two theoretical models can be brought together in the form of the **Kalman-TD Model**. This real-time model represent a distribution over weights rather than a point estimate.

This is consistent with a more normative approach as we can believe that an ideal agent should  be able to track its uncertainty using Bayesian Principles and learn about long-term (rather than just immediate) reward.


# How is the Kalman TD Model Build?

# How to Test the Superiority of the Kalman TD Model?
Even though the KF and TD can each explain well some violations of associative learning that the seminal Rescorla-Wagner model cannot account for (i.e. latent inhibition), some instances require the combination of both models to fit the behavioural data.  

## Recovery from Serial Compound Overshadowing
One of such instances include the recovery from serial compound overshadowing.

Overshadowing results from reinforcing the compound (i.e. AB -> +) which leads to weaker responding to the individual stimulus (i.e. A) compared to a condition in which each element is reinforced separately (i.e. A -> +). This effect can be counteracted by extinguishing one of the stimulus elements after the test.

The KF can account for this phenomenon only when the compound is presented simultaneously but not serially as it does not allow within-trial representation.  

## Behavioural Example (Shevill & Hall, 2004)
In particular, Shevill & Hall (2004) look at the effect of extinguishing the second-order stimulus Z on the responding to the first order stimulus.

| Phase 1        | Phase 2       | Test   |
| :-------------:|:-------------:| :-----:|
| A -> X -> +    | A -> -        | X -> ? |
| B -> Y -> +    |               | Y -> ? |

In a serial overshadowing procedure, the second-order stimulus (i.e. A) overshadows the first-order stimulus (i.e. X). Extinguishing the second-order stimulus (i.e. by not reinforcing it)  thus causes recovery from overshadowing. The behavioural results show higher value of the stimulus X relative to the stimulus Y whose associated second-order stimulus (i.e. B) had not be extinguished.

We expect that the Kalman TD model predicts well these results compared to the standard TD. 
Only the Kalman TD model learns that cue weights must sum to 1 (i.e. value of the reward) and is encoded as negative covariance between weights. This implies that post-training inflation or deflation of one stimulus will cause changes in beliefs about the other stimulus, which is not possible in TD as it does not measure covariance. 









# How to Simulate the Kalman TD Model?

## Create the Stimuli

## Create the Matrices

## Build the Models

## Plot the Results


# Conclusion

## Programming for Brain and Cognitive Sciences

This code was part of the course PBCS from the Cogmaster at the ENS (Paris, France).

## Differences between Python and Matlab

Sam Gersham made available his codes written in Matlab which allowed me to have a better understanding of the mathematics behind the model. Even if Matlab and Python are similar, the transition from one to the other is made difficult by a few major differences. Amongst those, zero-based indexing is easily ajusted for but the lack of true forms of matrices in Python and the data ordering of arrays which is row-major compared to column-major in Matlab requires more adjustements.

### Previous Experience in Coding

I had previous experience in coding using JavaScript, HTML and CSS. I also used MATLAB to analyse my data. However, these skills were self-thought, so I never learned how to optimise and write a clean code.

I think that through this course I gained proficiency using Python, which is a language I never programmed in before. This is especially useful as nowadays, more and more Cognitive Neuroscience lab use Python to code their experiments and analyse their data.

It also encouraged me to use more GitHub to organise my code (especially through literate programming of which I am now a fan), and to allow more reproducible science. It also opened my eyes on the benefits of open source, and I am considering switching to Linux.

### Feedback

Nonetheless, I still wish that we would have been able to do more language (I know it is difficult to implement MATLAB, but the reality is still that most labs are using it). I also wish that we would have been able to learn more optimisation of our codes rather than the coding of experiment itself. Even though it is great to have a wide variety of levels, I think that it can results in frustration in both those with lower levels and those with strong coding abilities. Two classes with different levels would benefit most of the students.
