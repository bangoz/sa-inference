### Implementation of SGD on linear/logistic regression tasks

import argparse
import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import ray

ray.init(ignore_reinit_error=True, include_dashboard=False)


# Arguments
parser = argparse.ArgumentParser(description='SGD on linear/logistic regression tasks.')
parser.add_argument('--n_sample', type=int, default=0, help='number of samples, n_sample=0 if samples are not pre-generated, default=0')
parser.add_argument('--n_dim', type=int, default=5, help='dimension of parameters, e.g. default=5')
parser.add_argument('--infer_dim', type=int, default=0, help='which dimension to perform inference')
parser.add_argument('--lr', type=float, default=0.01 , help='learning rate, e.g. 0.01/0.05/0.1')
parser.add_argument('--n_rep', type=int, default=1000, help='number of repeats, default=1000')
parser.add_argument('--n_iter', type=int, default=40000, help='number of iterations per repeat, default=40000')
parser.add_argument('--method', type=str, default='rs', help='method of inference, e.g. "rs"/"sv"/"bm"')
parser.add_argument('--task', type=str, default='lin', help='linear ("lin") or logistic ("log")')

args = parser.parse_args()
n_sample = args.n_sample
n_dim = args.n_dim
infer_dim = args.infer_dim
lr = args.lr
n_rep = args.n_rep
n_iter = args.n_iter
method = args.method
task = args.task


# data generation
beta = np.linspace(0, 1, n_dim + 1)[1:] # equispaced on [0,1]
if n_sample > 0:
    x = np.random.randn(n_sample, n_dim)
    if task == 'lin':
        eps = np.random.randn(n_sample)
        y = x @ beta + eps
    else:
        eps = np.random.logistic(size=n_sample)
        y = (x @ beta - eps >= 0) + 0
else:
    x, y = None, None


@ray.remote
def sgd_rs(x, y, n_sample, n_dim, lr, n_iter, beta_star, beta_expect, task, infer_dim=0):
    """
    beta_star: optimal parameter
    beta_expect: expectation of the stationary distribution
    """
    
    # online implementation of random scaling
    beta = np.zeros(n_dim)
    beta_bar = beta.copy()
    A_mat = np.outer(beta, beta)
    b_mat = beta.copy()
    is_cov_list_beta_star, is_cov_list_beta_expect, ci_length_list = [], [], [] # cov rate for beta_star/beta_expect; CI length

    tik = time.time()
    for i in range(n_iter):
        if n_sample > 0:
            idx = np.random.choice(n_sample)
            if task == 'lin':
                beta -= lr * (x[idx] @ beta - y[idx]) * x[idx]
            else:
                beta += lr * (y[idx] - 1 / (1 + np.exp(-x[idx] @ beta))) * x[idx]
        else: # n_sample = 0
            if task == 'lin':
                x = np.random.randn(n_dim)
                eps = np.random.randn(1)
                y = x @ beta_star + eps
                beta -= lr * (x @ beta - y) * x
            else:
                x = np.random.randn(n_dim)
                eps = np.random.logistic(size=1)
                y = (x @ beta_star - eps >= 0) + 0
                beta += lr * (y - 1 / (1 + np.exp(-x @ beta))) * x

        # construct V_n (online fashion)
        beta_bar = beta_bar * i / (i + 1) + beta / (i + 1)
        A_mat = A_mat + (i + 1)**2 * np.outer(beta_bar, beta_bar)
        b_mat = b_mat + (i + 1)**2 * beta_bar
        V_mat = (A_mat - np.outer(beta_bar, b_mat) - np.outer(b_mat, beta_bar) + ((i + 1) * (i + 2) * (2 * i + 3) / 6) * np.outer(beta_bar, beta_bar)) / (i + 1)**2

        # construct 95% CI q_{0.025}=6.753
        is_cov_beta_star = np.abs(beta_star[infer_dim] - beta_bar[infer_dim]) < 6.753 * np.sqrt(V_mat[infer_dim, infer_dim] / (i + 1))
        is_cov_beta_expect = np.abs(beta_expect[infer_dim] - beta_bar[infer_dim]) < 6.753 * np.sqrt(V_mat[infer_dim, infer_dim] / (i + 1))
        ci_length = 2 * 6.753 * np.sqrt(V_mat[infer_dim, infer_dim] / (i + 1))

        is_cov_list_beta_star.append(is_cov_beta_star)
        is_cov_list_beta_expect.append(is_cov_beta_expect)
        ci_length_list.append(ci_length)
    
    tok = time.time()
    t = tok - tik

    return is_cov_list_beta_star, is_cov_list_beta_expect, ci_length_list, t * np.ones(n_iter)


@ray.remote
def sgd_sv(x, y, n_sample, n_dim, lr, n_iter, beta_star, beta_expect, task, estimate_times, infer_dim=0):
    b_func = lambda n: int(np.sqrt(n**1.5))
    w_func = lambda n, k: (abs(k) < b_func(n)) * (0.5 + 0.5 * np.cos(np.pi * abs(k) / b_func(n)))

    beta = np.zeros(n_dim)
    beta_list = []
    beta_bar = beta.copy()
    is_cov_list_beta_star, is_cov_list_beta_expect, ci_length_list = [], [], [] # cov rate for beta_star/beta_expect; CI length

    tik = time.time()
    for i in range(n_iter):
        if n_sample > 0:
            idx = np.random.choice(n_sample)
            if task == 'lin':
                beta -= lr * (x[idx] @ beta - y[idx]) * x[idx]
            else:
                beta += lr * (y[idx] - 1 / (1 + np.exp(-x[idx] @ beta))) * x[idx]
        else: # n_sample = 0
            if task == 'lin':
                x = np.random.randn(n_dim)
                eps = np.random.randn(1)
                y = x @ beta_star + eps
                beta -= lr * (x @ beta - y) * x
            else:
                x = np.random.randn(n_dim)
                eps = np.random.logistic(size=1)
                y = (x @ beta_star - eps >= 0) + 0
                beta += lr * (y - 1 / (1 + np.exp(-x @ beta))) * x

        beta_list.append(beta.copy())
        beta_bar = beta_bar * i / (i + 1) + beta / (i + 1)

        if i + 1 in estimate_times:
            tmp = np.array(beta_list)[:, infer_dim]
            gammas = np.array([(tmp[:i+1-s] - beta_bar[infer_dim]) @ (tmp[s:] - beta_bar[infer_dim]) for s in range(0, b_func(i+1))]) / (i+1)
            sigma_hat = 2 * np.array([w_func(i+1, s) for s in range(0, b_func(i+1))]) @ gammas - gammas[0]

            # construct 95% CI z_{0.025}=1.96
            is_cov_beta_star = np.abs(beta_star[infer_dim] - beta_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            is_cov_beta_expect = np.abs(beta_expect[infer_dim] - beta_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            ci_length = 2 * 1.96 * np.sqrt(sigma_hat / (i + 1))

            is_cov_list_beta_star.append(is_cov_beta_star)
            is_cov_list_beta_expect.append(is_cov_beta_expect)
            ci_length_list.append(ci_length)
    
    tok = time.time()
    t = tok - tik
    
    return is_cov_list_beta_star, is_cov_list_beta_expect, ci_length_list, t * np.ones_like(ci_length_list)


@ray.remote
def sgd_bm(x, y, n_sample, n_dim, lr, n_iter, beta_star, beta_expect, task, estimate_times, infer_dim=0):
    an = lambda n: int(np.sqrt(n))
    bn = lambda n: int(np.sqrt(n))

    beta = np.zeros(n_dim)
    beta_list = []
    beta_bar = beta.copy()
    is_cov_list_beta_star, is_cov_list_beta_expect, ci_length_list = [], [], [] # cov rate for beta_star/beta_expect; CI length

    tik = time.time()
    for i in range(n_iter):
        if n_sample > 0:
            idx = np.random.choice(n_sample)
            if task == 'lin':
                beta -= lr * (x[idx] @ beta - y[idx]) * x[idx]
            else:
                beta += lr * (y[idx] - 1 / (1 + np.exp(-x[idx] @ beta))) * x[idx]
        else: # n_sample = 0
            if task == 'lin':
                x = np.random.randn(n_dim)
                eps = np.random.randn(1)
                y = x @ beta_star + eps
                beta -= lr * (x @ beta - y) * x
            else:
                x = np.random.randn(n_dim)
                eps = np.random.logistic(size=1)
                y = (x @ beta_star - eps >= 0) + 0
                beta += lr * (y - 1 / (1 + np.exp(-x @ beta))) * x

        beta_list.append(beta.copy())
        beta_bar = beta_bar * i / (i + 1) + beta / (i + 1)

        if i + 1 in estimate_times:
            beta_ary = np.array(beta_list)[:an(i+1) * bn(i+1), infer_dim]
            beta_mean = beta_ary.reshape(an(i+1), bn(i+1)).mean(axis=1)
            overall_mean = beta_mean.mean()
            sigma_hat = bn(i+1) / (an(i+1) - 1) * (beta_mean - overall_mean) @ (beta_mean - overall_mean)


            # construct 95% CI z_{0.025}=1.96
            is_cov_beta_star = np.abs(beta_star[infer_dim] - beta_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            is_cov_beta_expect = np.abs(beta_expect[infer_dim] - beta_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            ci_length = 2 * 1.96 * np.sqrt(sigma_hat / (i + 1))

            is_cov_list_beta_star.append(is_cov_beta_star)
            is_cov_list_beta_expect.append(is_cov_beta_expect)
            ci_length_list.append(ci_length)
    
    tok = time.time()
    t = tok - tik
    
    return is_cov_list_beta_star, is_cov_list_beta_expect, ci_length_list, t * np.ones_like(ci_length_list)


def cal_lin_expectation(x, y, n_sample, n_dim, lr, n_iter, beta_star):
    """
    calculate expectation of stationary distribution for linear regression
    """
    beta = np.zeros(n_dim)
    beta_list = []

    for i in range(n_iter):
        if n_sample > 0:
            idx = np.random.choice(n_sample)
            beta -= lr * (x[idx] @ beta - y[idx]) * x[idx]
        else:
            x = np.random.randn(n_dim)
            eps = np.random.randn(1)
            y = x @ beta_star + eps
            beta -= lr * (x @ beta - y) * x

        beta_list.append(beta.copy())
    
    return np.mean(beta_list[n_iter//2:], axis=0)


def cal_log_expectation(x, y, n_sample, n_dim, lr, n_iter, beta_star):
    """
    calculate expectation of stationary distribution for logistic regression
    """
    beta = np.zeros(n_dim)
    beta_list = []

    for i in range(n_iter):
        if n_sample > 0:
            idx = np.random.choice(n_sample)
            beta += lr * (y[idx] - 1 / (1 + np.exp(-x[idx] @ beta))) * x[idx]
        else:
            x = np.random.randn(n_dim)
            eps = np.random.logistic(size=1)
            y = (x @ beta_star - eps >= 0) + 0
            beta += lr * (y - 1 / (1 + np.exp(-x @ beta))) * x

        beta_list.append(beta.copy())
    
    return np.mean(beta_list[n_iter//2:], axis=0)



### main ###

# calculate beta_expect
print('Calculating expected value of beta ...')
beta_expect = cal_log_expectation(x, y, n_sample, n_dim, lr, n_iter=1000000, beta_star=beta) if task == 'log' else cal_lin_expectation(x, y, n_sample, n_dim, lr, n_iter=1000000, beta_star=beta)
print(beta_expect)

# checkpoints to perform inference when using 'sv' and 'bm'
if n_dim == 5 and n_iter == 40000:
    estimate_times = [5000, 10000, 20000, 40000]
elif n_dim == 20 and n_iter == 100000:
    estimate_times = [20000, 50000, 80000, 100000]
else:
    estimate_times = list((n_iter * np.arange(1, 5)) // 4)


# large scale experiment
future = []
for i in range(n_rep):
    if method == 'rs':
        future.append(sgd_rs.remote(x, y, n_sample, n_dim, lr, n_iter, beta_star=beta, beta_expect=beta_expect, task=task, infer_dim=0))
    elif method == 'sv':
        future.append(sgd_sv.remote(x, y, n_sample, n_dim, lr, n_iter, beta_star=beta, beta_expect=beta_expect, task=task, estimate_times=estimate_times, infer_dim=0))
    elif method == 'bm':
        future.append(sgd_bm.remote(x, y, n_sample, n_dim, lr, n_iter, beta_star=beta, beta_expect=beta_expect, task=task, estimate_times=estimate_times, infer_dim=0))
    else:
        raise NotImplementedError

    if (i + 1) % 100 == 0 or i + 1 == n_rep:
        print(i + 1, 'repeats are done.')

future = np.array(ray.get(future))
print('Getting data from remote done.')

cov_rates_beta_star_mean, cov_rates_beta_star_std = future[:, 0, :].mean(axis=0), future[:, 0, :].std(axis=0)
cov_rates_beta_expect_mean, cov_rates_beta_expect_std = future[:, 1, :].mean(axis=0), future[:, 1, :].std(axis=0)
ci_len_mean, ci_len_std = future[:, 2, :].mean(axis=0), future[:, 2, :].std(axis=0)
t_mean, t_std = future[:, 3, :].mean(axis=0), future[:, 3, :].std(axis=0)


# save data
path = os.getcwd() + '/sgd_data'
if not os.path.exists(path):
    os.makedirs(path)
new_path = path + '/ndim{}-lr{}-niter{}-nrep{}-{}-{}'.format(n_dim, lr, n_iter, n_rep, task, method)
if not os.path.exists(new_path):
    os.makedirs(new_path)

np.save(new_path + '/cov_rates_beta_star_mean.npy', cov_rates_beta_star_mean)
np.save(new_path + '/cov_rates_beta_star_std.npy', cov_rates_beta_star_std)
np.save(new_path + '/cov_rates_beta_expect_mean.npy', cov_rates_beta_expect_mean)
np.save(new_path + '/cov_rates_beta_expect_std.npy', cov_rates_beta_expect_std)
np.save(new_path + '/ci_len_mean.npy', ci_len_mean)
np.save(new_path + '/ci_len_std.npy', ci_len_std)
np.save(new_path + '/t_mean.npy', t_mean)
np.save(new_path + '/t_std.npy', t_std)


if method == 'rs':
    for i in estimate_times:
        print('{}-th iteration:'.format(i))
        print('beta_star_mean={}, beta_star_std={}, beta_expect_mean={}, beta_expect_std={}, ci_len_mean={}, ci_len_std={}, t_mean={}'\
            .format(cov_rates_beta_star_mean[i-1], 
            np.sqrt(cov_rates_beta_star_mean[i-1] * (1 - cov_rates_beta_star_mean[i-1]) / n_rep), 
            cov_rates_beta_expect_mean[i-1], 
            np.sqrt(cov_rates_beta_expect_mean[i-1] * (1 - cov_rates_beta_expect_mean[i-1]) / n_rep), 
            ci_len_mean[i-1], ci_len_std[i-1], t_mean[i-1]))
else:
    for i in range(4):
        print('{}-th iteration:'.format(i))
        print('beta_star_mean={}, beta_star_std={}, beta_expect_mean={}, beta_expect_std={}, ci_len_mean={}, ci_len_std={}, t_mean={}'\
            .format(cov_rates_beta_star_mean[i], 
            np.sqrt(cov_rates_beta_star_mean[i] * (1 - cov_rates_beta_star_mean[i]) / n_rep), 
            cov_rates_beta_expect_mean[i], 
            np.sqrt(cov_rates_beta_expect_mean[i] * (1 - cov_rates_beta_expect_mean[i]) / n_rep), 
            ci_len_mean[i], ci_len_std[i], t_mean[i]))

ray.shutdown()


# plt.plot(np.arange(n_iter), cov_rates_beta_star_mean)
# plt.plot(np.arange(n_iter), cov_rates_beta_expect_mean)
# plt.plot(np.arange(n_iter), np.ones(n_iter) * 0.95)
# plt.legend(['beta_star', 'beta_expect'])
# plt.show()

# plt.plot(np.arange(n_iter), ci_len_mean)
# plt.show()
