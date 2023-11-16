from __future__ import print_function, division
from builtins import range
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:

    def __init__(self, p):
        if False:
            return 10
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0

    def pull(self):
        if False:
            while True:
                i = 10
        return np.random.random() < self.p

    def sample(self):
        if False:
            i = 10
            return i + 15
        return np.random.beta(self.a, self.b)

    def update(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.a += x
        self.b += 1 - x
        self.N += 1

def plot(bandits, trial):
    if False:
        while True:
            i = 10
    x = np.linspace(0, 1, 200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label=f'real p: {b.p:.4f}, win rate = {b.a - 1}/{b.N}')
    plt.title(f'Bandit distributions after {trial} trials')
    plt.legend()
    plt.show()

def experiment():
    if False:
        print('Hello World!')
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.zeros(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])
        if i in sample_points:
            plot(bandits, i)
        x = bandits[j].pull()
        rewards[i] = x
        bandits[j].update(x)
    print('total reward earned:', rewards.sum())
    print('overall win rate:', rewards.sum() / NUM_TRIALS)
    print('num times selected each bandit:', [b.N for b in bandits])
if __name__ == '__main__':
    experiment()