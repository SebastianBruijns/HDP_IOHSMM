import numpy as np

x = None

transitions = x
emissions = x
durations = x
duration_dists = x
T = x
n_states = x
observations

def B(i, t):
    if t == T: return 1
    temp = 0
    for j in range(n_states):
        temp += B_star(j, t) * transitions[j:i]
    return temp

def B_star(i, t):
    temp = 0
    for d in range(1, T - t + 1):
        temp += B(i, t + d) * duration[i](d) * np.prod(emissions[i](observations[t+1:d+t+1]))
    temp +=
