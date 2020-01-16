import numpy as np
np.set_printoptions(suppress=True)
from scipy.stats import multivariate_normal, poisson
import matplotlib.pyplot as plt
import pyhsmm.basic.distributions as distributions
"""
Generate sequence of arrays of mice observations
"""


reward_prob = 0.5

def generate_n_obs(n):

    obs = np.zeros((n, 5))
    for i in range(n):
        obs[i, 0] = np.random.normal((1 if np.random.rand() < 0.5 else -1) * 35, 2) # contrast is on one of the two sides, visual angle of 35 degree
        reward = np.random.rand() > reward_prob # right answer or not, 50%?
        obs[i, 1] = np.random.normal(reward * 0.3, 0.01)  # implement reward decrement?
        obs[i, 2] = np.random.normal((not reward) * 1, 0.1) # punishment time
        obs[i, 3] = np.random.normal(np.random.uniform(-1, 1), 0.1) # phase of contrasts, completely random (2 pi is the real interval)
        obs[i, 4] = np.random.normal(1, 0.1) # strength of contrast

    return obs

def quiescence_phase():
    left = np.random.normal(0, 0.1) # is contrast on left
    right = np.random.normal(0, 0.1) # is contrast on right, make noise multiplicative?
    water = np.random.normal(0, 0.01) # reward
    phase = np.random.normal(0, 0.1) # phase of stimulus
    one_side = np.random.normal(1, 0.1) # random stimulus which is always on
    tone = np.random.normal(0, 0.1) # whether or not a tone was played

    return left, right, water, phase, one_side, tone

def response_phase(j, side, phase):
    left = np.random.normal(side, 0.1)
    right = np.random.normal((not side), 0.1)
    water = np.random.normal(0, 0.01)
    phase = np.random.normal(phase, 0.5)
    one_side = np.random.normal(1, 0.1)
    tone = np.random.normal(j < 15, 0.1) # play stim_on tone in the first 5 time steps

    return left, right, water, phase, one_side, tone

def reward_phase(j, reward):
    left = np.random.normal(0, 0.1)
    right = np.random.normal(0, 0.1)
    water = np.random.normal(reward, 0.01)
    phase = np.random.normal(0, 0.1)
    one_side = np.random.normal(1, 0.1)
    tone = np.random.normal(- (j < 45 and not reward), 0.1) # play punishment tone if no reward

    return left, right, water, phase, one_side, tone



trial_length = 50 # 10 quiescence, 5 stim_on, 25 response, 10 reward or not


def generate_n_trials(n):

    obs = np.zeros((n * trial_length, 6))

    for i in range(n):
        reward = np.random.rand() > reward_prob
        side = np.random.rand() > 0.5
        phase = 0 # np.random.uniform(-1, 1)

        for j in range(trial_length):
            if j < 10:
                obs[i * trial_length + j] = quiescence_phase()
            elif j < 40:
                obs[i * trial_length + j] = response_phase(j, side, phase)
            else:
                obs[i * trial_length + j] = reward_phase(j, reward)

    return obs

class HSMM(object):

    def __init__(self, obs_dim):

        self.state = 0
        self.state_list = []
        self.log_like = 0
        self.obs = np.zeros((0, obs_dim))


    def init_real_model(self):
        self.dur_params = [10., 5., 5., 25., 25., 10., 5.]
        self.dur_distns = [lambda param=param: poisson.rvs(param) for param in self.dur_params]

        self.trans_distns = [np.array([0., 0.5, 0.5, 0., 0., 0., 0.]),
                             np.array([0., 0., 0., 1., 0., 0., 0.]),
                             np.array([0., 0., 0., 0., 1., 0., 0.]),
                             np.array([0., 0., 0., 0., 0., 0.5, 0.5]),
                             np.array([0., 0., 0., 0., 0., 0.5, 0.5]),
                             np.array([1., 0., 0., 0., 0., 0., 0.]),
                             np.array([1., 0., 0., 0., 0., 0., 0.])]

        self.obs_means = [np.array([0., 0., 0., 0., 1., 0.]),
                          np.array([1., 0., 0., 0., 1., 1.]),
                          np.array([0., 1., 0., 0., 1., 1.]),
                          np.array([1., 0., 0., 0., 1., 0.]),
                          np.array([0., 1., 0., 0., 1., 0.]),
                          np.array([0., 0., 1., 0., 1., 0.]),
                          np.array([0., 0., 0., 0., 1., -1.])]

        self.obs_covs = [np.diag([0.05, 0.05, 0.01, 0.05, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.05, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.05, 0.05, 0.05])]

    def init_test_model(self):
        self.dur_params = [20., 20.]
        self.dur_distns = [lambda param=param: poisson.rvs(param) for param in self.dur_params]

        self.trans_distns = [np.array([0., 1.]),
                             np.array([1., 0.])]

        self.obs_means = [np.array([0.55 * 0.05]),
                          np.array([-0.55 * 0.05])]

        self.obs_covs = [np.diag([np.sqrt(0.05)]),
                         np.diag([np.sqrt(0.05)])]


    def init_noise_test(self, means=0.5, delta_t=0.1):
        self.dur_distns = []
        self.dur_distns.append(distributions.NegativeBinomial(1 / delta_t, 0.5))
        self.dur_distns.append(distributions.NegativeBinomial(1 / delta_t, 0.5))
        self.dur_distns.append(distributions.NegativeBinomial(1.4 / delta_t, 0.42))

        # left and right action state

        self.trans_distns = [np.array([0., 0., 1.]),
                             np.array([0., 0., 1.]),
                             np.array([0.5, 0.5, 0.])]

        self.obs_means = [np.array([means * delta_t]),
                          np.array([-means * delta_t]),
                          np.array([0.])]

        self.obs_covs = [np.diag([delta_t]),
                         np.diag([delta_t]),
                         np.diag([0.005])]


    def init_multi_test(self):
        self.dur_distns = []
        self.dur_distns.append(distributions.NegativeBinomial(10, 0.5))
        self.dur_distns.append(distributions.NegativeBinomial(10, 0.5))
        self.dur_distns.append(distributions.NegativeBinomial(2, 0.2))
        self.dur_distns.append(distributions.NegativeBinomial(2, 0.2))
        self.dur_distns.append(distributions.NegativeBinomial(14, 0.42))

        means = 3 #0.5
        delta_t = 0.1

        # left and right action state

        self.trans_distns = [np.array([0., 0., 1., 0., 0.]),
                             np.array([0., 0., 0., 1., 0.]),
                             np.array([0., 0., 0., 0., 1.]),
                             np.array([0., 0., 0., 0., 1.]),
                             np.array([0.5, 0.5, 0., 0., 0.])]

        self.obs_means = [np.array([means * delta_t, 0.]),
                          np.array([-means * delta_t, 0.]),
                          np.array([means * delta_t, 1.]),
                          np.array([-means * delta_t, -1.]),
                          np.array([0., 0.])]

        self.obs_covs = [np.diag([delta_t, 0.005]),
                         np.diag([delta_t, 0.005]),
                         np.diag([delta_t, 0.005]),
                         np.diag([delta_t, 0.005]),
                         np.diag([0.005, 0.005])]

    # generating doesnt work twice so far
    def generate_n_obs(self, n):

        generated = 0
        while generated < n:
            dur = self.dur_distns[self.state].rvs()
            self.log_like += self.dur_distns[self.state].log_likelihood(dur)
            dur += 1
            generated += dur
            self.state_list += dur * [self.state]

            temp = multivariate_normal.rvs(self.obs_means[self.state], self.obs_covs[self.state], dur)
            self.log_like += np.sum(np.log(multivariate_normal.pdf(temp, self.obs_means[self.state], self.obs_covs[self.state])))

            # does it make sense to potentially have state not at all? How does transition matrix deal with that
            if self.obs.shape[1] == 1:
                self.obs = np.concatenate((self.obs, temp[:,None]))
            elif dur > 1:
                self.obs = np.concatenate((self.obs, temp))


            self.state, old_state = np.random.choice(len(self.dur_distns), p=self.trans_distns[self.state]), self.state
            self.log_like += np.log(self.trans_distns[old_state][self.state])

        return self.obs, self.log_like, self.state_list

class new_HSMM(object):

    def __init__(self):
        self.dur_params = [(18, 0.2), (2, 0.2), (2, 0.2), (30, 0.4), (30, 0.4), (18, 0.2), (2, 0.2)]
        self.dur_distns = [distributions.NegativeBinomial(*param) for param in self.dur_params]
        self.min_dur = 5

        self.trans_distns = [np.array([0., 0.5, 0.5, 0., 0., 0., 0.]),
                             np.array([0., 0., 0., 1., 0., 0., 0.]),
                             np.array([0., 0., 0., 0., 1., 0., 0.]),
                             np.array([0., 0., 0., 0., 0., 0.5, 0.5]),
                             np.array([0., 0., 0., 0., 0., 0.5, 0.5]),
                             np.array([1., 0., 0., 0., 0., 0., 0.]),
                             np.array([1., 0., 0., 0., 0., 0., 0.])]

        self.obs_means = [np.array([0., 0., 0., 0., 1., 0.]),
                          np.array([1., 0., 0., 0., 1., 1.]),
                          np.array([0., 1., 0., 0., 1., 1.]),
                          np.array([1., 0., 0., 0., 1., 0.]),
                          np.array([0., 1., 0., 0., 1., 0.]),
                          np.array([0., 0., 1., 0., 1., 0.]),
                          np.array([0., 0., 0., 0., 1., -1.])]

        self.obs_covs = [np.diag([0.05, 0.05, 0.01, 0.05, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.5, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.05, 0.05, 0.05]),
                         np.diag([0.05, 0.05, 0.01, 0.05, 0.05, 0.05])]

        self.state = 0
        self.state_list = []
        self.log_like = 0
        self.obs = np.zeros((0, 6))

    # generating doenst work twice so far
    def generate_n_obs(self, n):

        generated = 0
        while generated < n:
            dur = self.dur_distns[self.state].rvs() + self.min_dur + 1
            self.log_like += self.dur_distns[self.state].log_likelihood(dur - self.min_dur -1) # meh
            generated += dur
            self.state_list += dur * [self.state]

            temp = multivariate_normal.rvs(self.obs_means[self.state], self.obs_covs[self.state], dur)
            self.log_like += np.sum(np.log(multivariate_normal.pdf(temp, self.obs_means[self.state], self.obs_covs[self.state])))

            # does it make sense to potentially have state not at all? How does transition matrix deal with that
            if dur > 1:
                self.obs = np.concatenate((self.obs, temp))
            if dur == 1:
                self.obs = np.concatenate((self.obs, temp[None]))

            self.state, old_state = np.random.choice(len(self.dur_params), p=self.trans_distns[self.state]), self.state
            self.log_like += np.log(self.trans_distns[old_state][self.state])

        return self.obs, self.log_like, self.state_list


    def generate_next_state(self):

        dur = self.dur_distns[self.state].rvs() + self.min_dur + 1
        self.log_like += self.dur_distns[self.state].log_likelihood(dur - self.min_dur -1) # meh
        self.state_list += dur * [self.state]

        temp = multivariate_normal.rvs(self.obs_means[self.state], self.obs_covs[self.state], dur)
        self.log_like += np.sum(np.log(multivariate_normal.pdf(temp, self.obs_means[self.state], self.obs_covs[self.state])))

        # does it make sense to potentially have state not at all? How does transition matrix deal with that
        if dur > 1:
            self.obs = np.concatenate((self.obs, temp))
        if dur == 1:
            self.obs = np.concatenate((self.obs, temp[None]))

        self.state, old_state = np.random.choice(len(self.dur_params), p=self.trans_distns[self.state]), self.state
        self.log_like += np.log(self.trans_distns[old_state][self.state])

        return self.obs[-dur:], self.log_like, dur * [old_state]

    def generate_next_state_action_dependent(self, action):

        if self.obs.shape[0] != 0:
            if self.state != 3 and self.state != 4:
                self.state = np.random.choice(len(self.dur_params), p=self.trans_distns[self.state])
            else:
                if (self.state == 3 and action[0] == 1) or (self.state == 4 and action[0] == 2):
                    self.state = 5
                else:
                    self.state = 6

        dur = self.dur_distns[self.state].rvs() + self.min_dur + 1
        self.state_list += dur * [self.state]

        temp = multivariate_normal.rvs(self.obs_means[self.state], self.obs_covs[self.state], dur)

        # does it make sense to potentially have state not at all? How does transition matrix deal with that
        if dur > 1:
            self.obs = np.concatenate((self.obs, temp))
        if dur == 1:
            self.obs = np.concatenate((self.obs, temp[None]))

        return self.obs[-dur:], self.log_like, dur * [self.state]

a = new_HSMM()
