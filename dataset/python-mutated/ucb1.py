from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
NUM_TRIALS = 100000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:

    def __init__(self, p):
        if False:
            print('Hello World!')
        self.p = p
        self.p_estimate = 0.0
        self.N = 0.0

    def pull(self):
        if False:
            i = 10
            return i + 15
        return np.random.random() < self.p

    def update(self, x):
        if False:
            i = 10
            return i + 15
        self.N += 1.0
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

def ucb(mean, n, nj):
    if False:
        while True:
            i = 10
    return mean + np.sqrt(2 * np.log(n) / nj)

def run_experiment():
    if False:
        while True:
            i = 10
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.empty(NUM_TRIALS)
    total_plays = 0
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
    for i in range(NUM_TRIALS):
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        rewards[i] = x
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.xscale('log')
    plt.show()
    plt.plot(cumulative_average)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()
    for b in bandits:
        print(b.p_estimate)
    print('total reward earned:', rewards.sum())
    print('overall win rate:', rewards.sum() / NUM_TRIALS)
    print('num times selected each bandit:', [b.N for b in bandits])
    return cumulative_average
if __name__ == '__main__':
    run_experiment()