from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(1)
NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]

class Bandit:

    def __init__(self, true_mean):
        if False:
            print('Hello World!')
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0

    def pull(self):
        if False:
            print('Hello World!')
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        if False:
            while True:
                i = 10
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        if False:
            while True:
                i = 10
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1

def plot(bandits, trial):
    if False:
        return 10
    x = np.linspace(-3, 6, 200)
    for b in bandits:
        y = norm.pdf(x, b.m, np.sqrt(1.0 / b.lambda_))
        plt.plot(x, y, label=f'real mean: {b.true_mean:.4f}, num plays: {b.N}')
    plt.title(f'Bandit distributions after {trial} trials')
    plt.legend()
    plt.show()

def run_experiment():
    if False:
        return 10
    bandits = [Bandit(m) for m in BANDIT_MEANS]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]
    rewards = np.empty(NUM_TRIALS)
    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])
        if i in sample_points:
            plot(bandits, i)
        x = bandits[j].pull()
        bandits[j].update(x)
        rewards[i] = x
    cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)
    plt.plot(cumulative_average)
    for m in BANDIT_MEANS:
        plt.plot(np.ones(NUM_TRIALS) * m)
    plt.show()
    return cumulative_average
if __name__ == '__main__':
    run_experiment()