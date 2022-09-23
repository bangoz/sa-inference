### Plotting the Q-learning experiment results

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='white', palette='Set2')

# Arguments
parser = argparse.ArgumentParser(description='Plotting Q-learning experiment results.')
parser.add_argument('--lrs', type=float, default=0.1, nargs='+' , help='learning rates, e.g. 0.1/0.2/0.3')
parser.add_argument('--stds', type=float, default=2.0, nargs='+', help='standard deviations of random rewards, e.g. 2.0/4.0')
parser.add_argument('--nrep', type=int, default=200, help='number of repeats, default=200')
parser.add_argument('--niter', type=int, default=5000, help='number of iterations per repeat, default=5000')
parser.add_argument('--method', type=str, default='all', help='method of inference, e.g. "rs"/"sv"/"bm"/"all", default="all"')

args = parser.parse_args()
lrs = args.lrs
stds = args.stds
nrep = args.nrep
niter = args.niter
methods = ['rs', 'sv', 'bm'] if args.method == 'all' else [args.method]


def save_fig(lr, std):
    path = os.getcwd() + '/qlearning_data/lr{}-niter{}-nrep{}-std{}'.format(lr, niter, nrep, std)
    data = []
    for m in methods:
        data.append(np.load(path+'/{}.npy'.format(m)))

    for i in range(3):
        for j in range(len(methods)):
            tmp = data[j][:, i, :].mean(axis=0)
            if tmp.shape[0] == niter:
                plt.plot(range(niter), tmp)
            else:
                plt.plot(range(5, niter, 5), tmp)
        if i != 2: # not CI length
            plt.plot(0.95*np.ones(niter))
            plt.xlabel('number of iterations')
            plt.ylabel('coverage rate')
            plt.legend(['random scaling', 'spectral variance', 'batch means', 'nominal'], loc='lower right')
        else:
            plt.xlabel('number of iterations')
            plt.ylabel('CI length')
            plt.legend(['random scaling', 'spectral variance', 'batch means', 'nominal'], loc='upper right')
        plt.title('lr={}, noise std={}'.format(lr, std))
        # plt.show()

        fig_path = os.getcwd() + '/qlearning_fig'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(fig_path + '/lr{}-std{}-{}.png'.format(lr, std, i))
        plt.clf()


for lr in lrs:
    for std in stds:
        save_fig(lr, std)
