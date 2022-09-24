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
* SGD: 
  - Run `python sgd.py --n_dim 5 --lr 0.01 --n_rep 1000 --n_iter 40000 --method rs --task lin` in terminal.
  - Run `python sgd.py -h` for descriptions of each optional parameter.
* Q-learning: 
  - Run `python qlearning.py --lr 0.1 --std 2.0 --nrep 200 --niter 5000 --method all` in terminal.
  - Run `python qlearning.py -h` for descriptions of each optional parameter.
  - Run `python plot_qlearning.py --lrs 0.1 0.2 0.3 --stds 2.0 4.0 --nrep 200 --niter 5000 --method all` for plotting the experiment results. In this example, you should ensure that experiments with `lr=0.1/0.2/0.3` and `std=2.0/4.0` are all conducted in the first step.
