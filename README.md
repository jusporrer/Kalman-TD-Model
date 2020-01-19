[Kalman-TD Model](https://jusporrer.github.io/PCBS-Kalman-Filter/)
================

This program in Python allows the **Kalman-TD Model** to fit simulated behavioural data.

The initial code in Matlab is based on the previous work of **Samuel Gershman**, at the Department of Psychology and Center for Brain Science, Harvard University [(See his GitHub here)](https://github.com/sjgershm/KF-learning). This model is introduced in his paper [*A Unifying Probabilistic View of Associative Learning*](https://journals.plos.org/ploscompbiol/article/file?id=10.1371/journal.pcbi.1004567&type=printable) which describes a framework encompassing Bayesian and Reinforcement Learning theories of associative learning.

For more information please contact me at <juliana.sporrer.18@ucl.ac.uk>.


## Table of Contents

1. [What is the Kalman TD Model?](#What-is-the-Kalman-TD-Model-?)
      1. [Associative Learning Theories](#Associative-Learning-Theories)
      1. [Unifying Model](#Unifying-Model)  
1. [How is the Kalman TD Model Build?](#How-is-the-Kalman-TD-Model-Build-?)
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
> KF learns the *posterior distribution* of expected *immediate* reward
- An agent learns about long-term cumulative future reward using **Reinforcement Learning (RL) principles** and represented by **Temporal Difference (TD)**.
> TD learns a *single value* of expected *future* cumulative reward


## Unifying Model

These two theoretical models can be brought together in the form of the **Kalman-TD Model**. This real-time model represent a distribution over weights rather than a point estimate.

This is consistent with a more normative approach as we can believe that an ideal agent should be able to track its uncertainty using Bayesian Principles and learn about long-term (rather than just immediate) reward.


# How is the Kalman TD Model Build ?
The Kalman-TD allows the agent to track the entire distribution over discounted future returns by combining the KF and the TD. More precisely, the learner represents uncertainty in the form of a posterior distribution over hypotheses given data according to the Bayes’ rule. This posterior Gaussian distribution has a mean Wn (weight at time n) and covariance matrix ∑n.
-	The weight is updated through the Kalman Gain (equivalent of learning rate) and the traditional prediction error delta.
> `Wn+1 = Wn + Kn*δn`

-	The covariance matrix is updated through two factors: increasing uncertainty over time due to gaussian random diffusion (represented by tau square) and decreasing uncertainty when observing data (expressed by the term with the Kalman gain).
> `∑n+1 = ∑n + τ²I - Kn*Xn.transpose*(∑n + τ²I)`

-	The Kalman Gain is stimulus-specific, dynamic and grows with the uncertainty encoded in the diagonals of the posterior covariance matrix.
> `Kn = ((∑n + τ²I)*Xn)/(Xn.transpose*(∑n + τ²I)*Xn + σ²r)`

In sum, the Kalman-TD mainly relies on the KF with the stimulus features being replaced by their discounted time derivate:
> `Ht = γ * Xt+1 - Xt`

# How to Test the Superiority of the Kalman TD Model ?
Even though the KF and TD can each explain well some violations of associative learning that the seminal Rescorla-Wagner model cannot account for (i.e. latent inhibition), some instances require the combination of both models to fit the behavioural data.  

## Recovery from Serial Compound Overshadowing
One of such instances include the recovery from serial compound overshadowing.

Overshadowing results from reinforcing the compound (i.e. AB -> +) which leads to weaker responding to the individual stimulus (i.e. A) compared to a condition in which each element is reinforced separately (i.e. A -> +). This effect can be counteracted by extinguishing one of the stimulus elements after the test.

The KF can account for this phenomenon only when the compound is presented simultaneously but not serially as it does not allow within-trial representation.  

## Behavioural Example (Shevill & Hall, 2004)
In particular, Shevill & Hall (2004) looked at the effect of extinguishing the second-order stimulus A on the responding to the first order stimulus.

| Phase 1                         | Phase 2                        | Test                            |
| :------------------------------:|:------------------------------:| :------------------------------:|
| A -> X -> +                     | A -> -                         | X -> ?                          |
| B -> Y -> +                     |                                | Y -> ?                          |

In a **serial overshadowing** procedure, the second-order stimulus (i.e. A) overshadows the first-order stimulus (i.e. X). Extinguishing the second-order stimulus (i.e. by not reinforcing it)  thus causes recovery from overshadowing. The behavioural results showed that the **stimulus X has a higher value than the stimulus Y** whose associated second-order stimulus (i.e. B) has not be extinguished.

We expect that the **Kalman TD model predicts well these results** compared to the standard TD.
Only the Kalman TD model learns that cue weights must sum to 1 (i.e. value of the reward) and is encoded as negative covariance between weights. This implies that post-training inflation or deflation of one stimulus will cause changes in beliefs about the other stimulus, which is not possible in TD as it does not measure covariance.

# How to Simulate the Kalman TD Model ?

## Create the Stimuli
First, we need to create the stimuli (i.e. conditioned stimuli) and their associated reward (i.e. unconditioned stimulus). For this we create a small function in the *functions.py* file.
The *onset* determines when the reward starts and *duration* for how long.

```
import numpy as np

def create_stimuli(onset, duration, trial_length):

    stimuli = np.zeros((trial_length), dtype = np.int)

    if onset >= 0:
        stimuli[onset:(onset + duration)] = 1
        return stimuli
```

The function *create_stimuli()* is called from our main class called *serial_overshadowing_simulation()* in the *KalmanTD_model.py* file. The class is first initialised with all the parameters needed for the creation of the stimuli and later on for the model.
Then, the method *initialise_stimuli()* creates the three component of our simulation. Stimuli_1 being the serial compound (A -> X -> +), Stimuli_2 being the serial compound (B -> Y -> +) and finally Stimuli_3 being the extinguished second-order stimuli A.


```

class serial_overshadowing_simulation():
    """ Simulates the recovery from serial compound overshadowing (with second-order extinction)
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

```

## Create the Matrices
With these associations of stimuli with their reward, we need to create their respective covariance matrix. They are then added together to create a general covariance of the entire experiment (through the method *initialise_covariance_matrix()* of the same class than before).

```
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

```

The same is effectuated for the reward matrices (through the method *initialise_reward_matrix()* of the same class than before).

```
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
```

You may have noticed that the covariance matrices were created using the function *construct_covariance_matrix()* from the file *functions.py*. I decided to pre-allocate the matrix before the loop iteration to improve processing speed as growing an array by assignment can be costly.

```
def construct_covariance_matrix(stimuli, trial_length, nb_stimuli):

    covariance_matrix = np.zeros((trial_length,nb_stimuli*10), dtype = np.int)

    start=0
    finish=1
    for i in range(nb_stimuli):

        covariance_matrix[::,start*trial_length:finish*trial_length] = np.diag(stimuli[:,i])

        start += 1
        finish += 1

    return covariance_matrix

```

## Build the Models
The Kalman TD and the standard TD models are built in the same method called *Kalman_TD* which is in the same file as our main class (i.e. *KalmanTD_model.py*).


```

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
```

This function is then called by the class *serial_overshadowing_simulation()* under the methods *Kalman_TD()* and *standard_TD()*. The weight matrix is sliced in such a way to only obtain the values of the stimuli X and Y.  

```
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

```

## Plot the Results
As mentioned earlier, we expect that as compared to the standard TD, the Kalman TD model predicts well the behavioural results in which the value of the stimulus X is higher than the stimuli Y as its respective second-order stimuli (i.e. A) has been extinguished.

Therefore, after launching our simulation in the main file *run.py*, we plot the results using simple bars which is saved as a pdf file.

```
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

```

The *run.py* file is also the one in which all the parameters are set. Thus, it is easily possible to generate new simulations with different parameters.

# Conclusion

I am very happy about this project as it allowed me to understand better seminal models used very frequently in Computational Neuroscience.
Even though I had access to the original codes, sole replication using MATLAB was already challenging due to a few bugs. However, they brought me a better understanding of how to build a model and the mathematics behind it. Utilising the object-oriented features of Python, I changed the structure and optimised it. Finally, I tried to write it in such a way to be able to include other stimulations utilising different stimuli associations.   


## Programming for Brain and Cognitive Sciences

This code was part of the course PBCS from the Cogmaster at the ENS (Paris, France).

### Previous Experience in Coding

I had previous experience in coding using JavaScript, HTML and CSS. I also used MATLAB to analyse my data. However, these skills were self-thought, so I never learned how to optimise and write a clean code.

Through this course, I gained proficiency using Python, which is a language I never programmed in before. This is especially useful as nowadays, more and more Cognitive Neuroscience labs use Python to code their experiments and analyse their data.

It also encouraged me to use more GitHub to organise my code (especially through literate programming), and to allow more reproducible science. It also opened my eyes to the benefits of open source.

### Feedback

Nonetheless, I still wish that we would have been able to do more languages (I know it is difficult to implement Matlab, but the reality is still that most labs are using it). I also wish that we would have been able to learn more optimisation of our codes rather than coding the experiment itself.
