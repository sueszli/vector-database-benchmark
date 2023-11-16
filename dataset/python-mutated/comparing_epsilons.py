from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt

class BanditArm:

    def __init__(self, m):
        if False:
            while True:
                i = 10
        self.m = m
        self.m_estimate = 0
        self.N = 0

    def pull(self):
        if False:
            print('Hello World!')
        return np.random.randn() + self.m

    def update(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.N += 1
        self.m_estimate = (1 - 1.0 / self.N) * self.m_estimate + 1.0 / self.N * x

def run_experiment(m1, m2, m3, eps, N):
    if False:
        print('Hello World!')
    bandits = [BanditArm(m1), BanditArm(m2), BanditArm(m3)]
    means = np.array([m1, m2, m3])
    true_best = np.argmax(means)
    count_suboptimal = 0
    data = np.empty(N)
    for i in range(N):
        p = np.random.random()
        if p < eps:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.m_estimate for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        if j != true_best:
            count_suboptimal += 1
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()
    for b in bandits:
        print(b.m_estimate)
    print('percent suboptimal for epsilon = %s:' % eps, float(count_suboptimal) / N)
    return cumulative_average
if __name__ == '__main__':
    (m1, m2, m3) = (1.5, 2.5, 3.5)
    c_1 = run_experiment(m1, m2, m3, 0.1, 100000)
    c_05 = run_experiment(m1, m2, m3, 0.05, 100000)
    c_01 = run_experiment(m1, m2, m3, 0.01, 100000)
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_05, label='eps = 0.05')
    plt.plot(c_01, label='eps = 0.01')
    plt.legend()
    plt.show()