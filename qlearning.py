### Implementation of synchronous Q-learning on Grid World

import argparse
import os
import numpy as np

# Grid World parameters
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)

# Arguments
parser = argparse.ArgumentParser(description='Q-learning experiment on Grid World.')
parser.add_argument('--lr', type=float, default=0.1 , help='learning rate, e.g. 0.1/0.2/0.3')
parser.add_argument('--std', type=float, default=2.0, help='standard deviation of random rewards, e.g. 2.0/4.0')
parser.add_argument('--nrep', type=int, default=200, help='number of repeats, default=200')
parser.add_argument('--niter', type=int, default=5000, help='number of iterations per repeat, default=5000')
parser.add_argument('--method', type=str, default='all', help='method of inference, e.g. "rs"/"sv"/"bm"/"all", default="all"')

args = parser.parse_args()
lr = args.lr
reward_std = args.std
nrep = args.nrep
niter = args.niter
methods = ['rs', 'sv', 'bm'] if args.method == 'all' else [args.method]


class Agent:

    def __init__(self, lr=0.1, is_dtm_reward=True, reward_std=0.2, inference=True, method=None):
        self.actions = ['up', 'down', 'left', 'right']
        self.decay_gamma = 0.9
        self.lr = lr # learning rate
        self.is_dtm_reward = is_dtm_reward # whether to use deterministic rewards
        self.reward_std = reward_std # standard deviation of rewards (if random rewards are used)
        self.inference = inference # whether to perform an inference step
        self.method = method # method of inference, ['rs', 'sv', 'bm', 'obm']

        # initialize Q values
        self.Q_values = np.zeros((BOARD_ROWS, BOARD_COLS, 4))
        
        # parameters used for inference
        self.Q_star = None
        self.Q_expect = None

        self.Q_flat = self.Q_values.flatten()
        self.Q_list = []
        self.Q_bar = self.Q_flat.copy()
        self.A_mat = np.outer(self.Q_flat, self.Q_flat)
        self.b_mat = self.Q_flat.copy()
        self.is_cov_list_q_star, self.is_cov_list_q_expect, self.ci_length_list = [], [], []

    def next_max_Q_value(self, state0, state1, action):
        if action == 'up':
            nxtState = (state0 - 1, state1)
        elif action == 'down':
            nxtState = (state0 + 1, state1)
        elif action == 'left':
            nxtState = (state0, state1 - 1)
        else:
            nxtState = (state0, state1 + 1)

        # if next state is illegal
        if nxtState[0] < 0 or nxtState[0] > 2 or nxtState[1] < 0 or nxtState[1] > 3 or nxtState == (1, 1):
            nxtState = (state0, state1)
        
        return self.Q_values[nxtState[0], nxtState[1], :].max()

        
    def play(self, niter=50):
        new_Q_values = self.Q_values.copy()
        for i in range(niter):
            reward = (1-self.is_dtm_reward) * self.reward_std * np.random.randn(BOARD_ROWS, BOARD_COLS, 4) - np.ones((BOARD_ROWS, BOARD_COLS, 4))
            reward[0, 3, :] = 10 # WIN_STATE
            reward[1, 3, :] = -10 # LOSE_STATE
            next_max_Q_values = np.array([self.next_max_Q_value(i, j, k) for i in range(BOARD_ROWS) for j in range(BOARD_COLS) for k in ['up', 'down', 'left', 'right']]).reshape(BOARD_ROWS, BOARD_COLS, 4)
            new_Q_values = (1 - self.lr) * self.Q_values + self.lr * (reward + self.decay_gamma * next_max_Q_values)
            new_Q_values[0, 3, :] = 10
            new_Q_values[1, 3, :] = -10
            new_Q_values[1, 1, :] = 0

            self.Q_values = new_Q_values.copy()
            self.Q_list.append(self.Q_values.flatten())

            # inference step
            if self.inference:
                if self.method == 'rs':
                    # random sampling
                    rs_result = self.infer_rs(i)
                    self.is_cov_list_q_star.append(rs_result[0])
                    self.is_cov_list_q_expect.append(rs_result[1])
                    self.ci_length_list.append(rs_result[2])
                elif self.method == 'sv':
                    # spectral variance
                    checkpoints = range(0, niter, 5)
                    sv_result = self.infer_sv(i, checkpoints)
                    if i + 1 in checkpoints:
                        self.is_cov_list_q_star.append(sv_result[0])
                        self.is_cov_list_q_expect.append(sv_result[1])
                        self.ci_length_list.append(sv_result[2])
                elif self.method == 'bm':
                    # batch means
                    checkpoints = range(0, niter, 5)
                    bm_result = self.infer_bm(i, checkpoints)
                    if i + 1 in checkpoints:
                        self.is_cov_list_q_star.append(bm_result[0])
                        self.is_cov_list_q_expect.append(bm_result[1])
                        self.ci_length_list.append(bm_result[2])
                elif self.method == 'obm':
                    # overlapping batch means
                    checkpoints = range(0, niter, 5)
                    obm_result = self.infer_obm(i, checkpoints)
                    if i + 1 in checkpoints:
                        self.is_cov_list_q_star.append(obm_result[0])
                        self.is_cov_list_q_expect.append(obm_result[1])
                        self.ci_length_list.append(obm_result[2])
                else:
                    raise NotImplementedError

            elif not self.is_dtm_reward: # used to compute Q_expect
                if i >= 2500: # warm up
                    ii = i - 2500
                    self.Q_flat = self.Q_values.flatten()
                    self.Q_bar = self.Q_bar * ii / (ii + 1) + self.Q_flat / (ii + 1)

    
    def infer_rs(self, i, infer_dim=0): # random scaling
        # construct V_n
        self.Q_flat = self.Q_values.flatten()
        self.Q_bar = self.Q_bar * i / (i + 1) + self.Q_flat / (i + 1)
        self.A_mat = self.A_mat + (i + 1)**2 * np.outer(self.Q_bar, self.Q_bar)
        self.b_mat = self.b_mat + (i + 1)**2 * self.Q_bar
        self.V_mat = (self.A_mat - np.outer(self.Q_bar, self.b_mat) - np.outer(self.b_mat, self.Q_bar) + ((i + 1) * (i + 2) * (2 * i + 3) / 6) * np.outer(self.Q_bar, self.Q_bar)) / (i + 1)**2

        is_cov_q_star = np.abs(self.Q_star[infer_dim] - self.Q_bar[infer_dim]) < 6.747 * np.sqrt(self.V_mat[infer_dim, infer_dim] / (i + 1))
        is_cov_q_expect = np.abs(self.Q_expect[infer_dim] - self.Q_bar[infer_dim]) < 6.747 * np.sqrt(self.V_mat[infer_dim, infer_dim] / (i + 1))
        ci_length = 2 * 6.747 * np.sqrt(self.V_mat[infer_dim, infer_dim] / (i + 1))

        return is_cov_q_star, is_cov_q_expect, ci_length
    

    def infer_sv(self, i, checkpoints, infer_dim=0): # spectral variance
        b_func = lambda n: int(np.sqrt(n**1.25))
        w_func = lambda n, k: (abs(k) < b_func(n)) * (0.5 + 0.5 * np.cos(np.pi * abs(k) / b_func(n)))

        self.Q_flat = self.Q_values.flatten()
        self.Q_bar = self.Q_bar * i / (i + 1) + self.Q_flat / (i + 1)
        
        if i + 1 in checkpoints:
            tmp = np.array(self.Q_list)[:, infer_dim] # length i+1
            gammas = np.array([(tmp[:i+1-s] - self.Q_bar[infer_dim]) @ (tmp[s:] - self.Q_bar[infer_dim]) for s in range(0, b_func(i+1))]) / (i+1)
            sigma_hat = 2 * np.array([w_func(i+1, s) for s in range(0, b_func(i+1))]) @ gammas - gammas[0]

            # construct 95% CI z_{0.025}=1.96
            is_cov_q_star = np.abs(self.Q_star[infer_dim] - self.Q_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            is_cov_q_expect = np.abs(self.Q_expect[infer_dim] - self.Q_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            ci_length = 2 * 1.96 * np.sqrt(sigma_hat / (i + 1))

            return is_cov_q_star, is_cov_q_expect, ci_length
        else:
            return None
    

    def infer_bm(self, i, checkpoints, infer_dim=0): # batch means
        an = lambda n: int(np.sqrt(n))
        bn = lambda n: int(np.sqrt(n))

        self.Q_flat = self.Q_values.flatten()
        self.Q_bar = self.Q_bar * i / (i + 1) + self.Q_flat / (i + 1)
        
        if i + 1 in checkpoints:
            beta_ary = np.array(self.Q_list)[:an(i+1) * bn(i+1), infer_dim]
            beta_mean = beta_ary.reshape(an(i+1), bn(i+1)).mean(axis=1)
            overall_mean = beta_mean.mean()
            sigma_hat = bn(i+1) / (an(i+1) - 1) * (beta_mean - overall_mean) @ (beta_mean - overall_mean)

            # construct 95% CI z_{0.025}=1.96
            is_cov_q_star = np.abs(self.Q_star[infer_dim] - self.Q_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            is_cov_q_expect = np.abs(self.Q_expect[infer_dim] - self.Q_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            ci_length = 2 * 1.96 * np.sqrt(sigma_hat / (i + 1))

            return is_cov_q_star, is_cov_q_expect, ci_length
        else:
            return None

    
    def infer_obm(self, i, checkpoints, infer_dim=0): # overlapping batch means
        bn = lambda n: int(n**(3/4))

        self.Q_flat = self.Q_values.flatten()
        self.Q_bar = self.Q_bar * i / (i + 1) + self.Q_flat / (i + 1)
        
        if i + 1 in checkpoints:
            overall_mean = np.array(self.Q_list)[:, infer_dim].mean()
            sigma_hat = (i+1)*bn(i+1)/(i+1-bn(i+1))/(i+2-bn(i+1)) * np.sum([(np.array(self.Q_list)[j:j+bn(i+1), infer_dim].mean() - overall_mean)**2 for j in range(i+1-bn(i+1))])

            # construct 95% CI z_{0.025}=1.96
            is_cov_q_star = np.abs(self.Q_star[infer_dim] - self.Q_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            is_cov_q_expect = np.abs(self.Q_expect[infer_dim] - self.Q_bar[infer_dim]) < 1.96 * np.sqrt(sigma_hat / (i + 1))
            ci_length = 2 * 1.96 * np.sqrt(sigma_hat / (i + 1))

            return is_cov_q_star, is_cov_q_expect, ci_length
        else:
            return None



### computing Q_star ###
ag = Agent(lr=lr, reward_std=reward_std, is_dtm_reward=True, inference=False)
ag.play(5000)
print('optimal Q-values ...')
print(ag.Q_values)
Q_star = ag.Q_values.flatten()

### computing Q_expect ###
ag = Agent(lr=lr, reward_std=reward_std, is_dtm_reward=False, inference=False)
ag.play(500000)
print('expected Q-values ...')
print(ag.Q_bar)
Q_expect = ag.Q_bar


### repeated experiment ###
path = os.getcwd() + '/qlearning_data'
if not os.path.exists(path):
    os.makedirs(path)
new_path = path + '/lr{}-niter{}-nrep{}-std{}'.format(lr, niter, nrep, reward_std)
if not os.path.exists(new_path):
    os.makedirs(new_path)

for method in methods:
    future = []
    for i in range(nrep):
        print('{}-th repeat begins, method={}'.format(i, method))
        ag = Agent(lr=lr, is_dtm_reward=False, reward_std=reward_std, inference=True, method=method)
        ag.Q_star = Q_star
        ag.Q_expect = Q_expect
        ag.play(niter)
        future.append([ag.is_cov_list_q_star, ag.is_cov_list_q_expect, ag.ci_length_list])

    future = np.array(future)
    np.save(new_path + '/{}.npy'.format(method), future)
