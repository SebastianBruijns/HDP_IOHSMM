import numpy as np
from generate_obs import HSMM
import matplotlib.pyplot as plt
import pyhsmm
import pyhsmm.basic.distributions as distributions
from pybasicbayes.util.text import progprint_xrange
import copy
import warnings

seeds = [12, 13, 16, 10, 17, 6, 1, 14, 3, 4]
seeds = [17]#, 16] 17

np.random.seed(4)
hsmm = HSMM(1)
hsmm.init_noise_test(means=1, delta_t=0.1)
n_obs = 500
data, ll, states = hsmm.generate_n_obs(n_obs)
print(ll)
print(data.shape)
states = np.array(states)
#print(np.mean(data[states == 0]))
#print(np.mean(data[states == 1]))
#print(np.mean(data[states == 2]))

Nmax = 4
obs_dim = data.shape[1]


n = 200
args_aggregate = np.zeros(3)
passes = 0
likes = np.zeros((len(seeds), n))
models = []
for i, seed in enumerate(seeds):
    np.random.seed(seed)
    print(i+1)
    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.eye(obs_dim)*0.1,
                    'kappa_0':0.3,
                    'nu_0':obs_dim+5}
    dur_hypparams = dict(
            #r_support=np.arange(20)+1,r_probs=np.ones(20)/20.,
            r_support=np.arange(2, 28, 2), r_probs=np.ones(13)/13.,
            alpha_0=2,
            beta_0=2,
            )

    obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    dur_distns = [distributions.NegativeBinomialIntegerR2Duration(**dur_hypparams) for state in range(Nmax)]

    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
            alpha_a_0=.05, alpha_b_0=1./20,
            gamma_a_0=1.,gamma_b_0=1./4,
            init_state_concentration=6., # pretty inconsequential
            obs_distns=obs_distns,
            dur_distns=dur_distns)

    posteriormodel.add_data(data,trunc=60)

    with warnings.catch_warnings(): # ignore the scipy warning
        warnings.simplefilter("ignore")
        for j in range(n):
            # problem in /home/sebastian/anaconda3/envs/hsmmEnv/lib/python2.7/site-packages/scipy/stats/_distn_infrastructure.py line 896
            # I think it's ok, code seems to deal with infinities
            posteriormodel.resample_model()

            likes[i, j] = posteriormodel.log_likelihood()

            if j >= 120:
                #indices = posteriormodel.state_usages.argsort()[-3:][::-1]

                #for k, ind in enumerate([4, 16, 24]):
                for k, ind in enumerate([0, 9, 10]):
                    args_aggregate[k] += obs_distns[ind].mu / 80


    print(posteriormodel.log_likelihood())
    models.append(copy.deepcopy(posteriormodel))
    print(posteriormodel.log_likelihood())
    state_counts = np.bincount(posteriormodel.stateseqs[0])
    print(state_counts)
    for state in posteriormodel.used_states:
        print(state)
        print(posteriormodel.obs_distns[state].mu)
    print()

print(args_aggregate)
print(np.argmin(np.abs(likes[:, -1] - ll)))
print(passes / len(seeds))
posteriormodel = models[np.argmin(np.abs(likes[:, -1] - ll))] # maybe dont take best likelihood but the one that is closest to real one

#print(posteriormodel.num_parameters) # I changed the working of num_parameters

def reduce(x):
    nums = np.unique(x)
    transf = dict(zip([0, 1, 2, 3], [2, 0, 1, 3]))
    for i, n in enumerate(np.sort(nums)):
        x[x == n] = transf[i]
    return x

fs = 22
plt.figure(figsize=(16, 9))
plt.plot(states[:400], 'o', markersize=8, color='red', label='Correct')
plt.plot(states[:400], 'o', color='white', markersize=6.5)
plt.plot(reduce(posteriormodel.stateseqs[0])[:400], 'o', markersize=5., color='green', label='Estimated')
plt.yticks([0, 1, 2], ['Contrast left', 'Contrast right', 'ITI'], fontsize= fs)
plt.xticks(fontsize= fs)
plt.legend(frameon=0, fontsize= fs)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel("Time in 0.1 ms", fontsize= fs)
plt.ylabel('States', fontsize= fs)

plt.tight_layout()
plt.savefig('./mean=1.png')
plt.show()


plt.plot(likes.T)
plt.gca().axhline(ll, color='black')
plt.show()

quit()
print(posteriormodel.state_usages)
print(list(posteriormodel.used_states))
print(np.bincount(posteriormodel.stateseqs[0]))

for state in posteriormodel.used_states:
    print(state)
    print(posteriormodel.obs_distns[state].mu)
    print(posteriormodel.dur_distns[state])
    print(np.bincount(posteriormodel.stateseqs[0]))
    #print(posteriormodel.trans_distn.trans_matrix[state])
    #print(posteriormodel.obs_distns[state].sigma)

print(posteriormodel.log_likelihood())

plt.plot(posteriormodel.stateseqs[0][:100], 'k.')
plt.plot(states[:100], 'r.')
plt.show()
