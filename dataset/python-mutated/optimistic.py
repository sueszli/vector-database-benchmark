from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:

    def __init__(self, p):
        if False:
            return 10
        self.p = p
        self.p_estimate = 5.0
        self.N = 1.0

    def pull(self):
        if False:
            while True:
                i = 10
        return np.random.random() < self.p

    def update(self, x):
        if False:
            return 10
        self.N += 1.0
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

def experiment():
    if False:
        i = 10
        return i + 15
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = np.argmax([b.p_estimate for b in bandits])
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
    for b in bandits:
        print('mean estimate:', b.p_estimate)
    print('total reward earned:', rewards.sum())
    print('overall win rate:', rewards.sum() / NUM_TRIALS)
    print('num times selected each bandit:', [b.N for b in bandits])
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.ylim([0, 1])
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()
if __name__ == '__main__':
    experiment()