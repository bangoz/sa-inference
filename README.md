# A Statistical Online Inference Approach in Averaged Stochastic Approximation
This repository contains the code for the paper "A Statistical Online Inference Approach in Averaged Stochastic Approximation" in Python. Two stochastic approximation algorithms (i.e., SGD and Q-learning) are implemented along with three inferential methods for model parameters: random scaling ("rs"), spectral variance ("sv"), and batch means ("bm").

## Requirement
* Python 3: numpy, matplotlib, seaborn (for plotting), ray (for acceleration)

## File Overview
* sgd.py: Experiments for SGD on linear/logistic regression tasks
* qlearning.py: Experiments for Q-learning on Grid World
* plot_qlearning.py: Plotting coverage rates and CI lengths in Q-learning experiments
* sgd_data: Containing SGD experiment results
* qlearning_data: Containing Q-learning experiment results
* qlearning_fig: Containing figures generated according to Q-learning experiment results

## How to Run
xxx
