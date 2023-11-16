"""
================
The Bayes update
================

This animation displays the posterior estimate updates as it is refitted when
new data arrives.
The vertical line represents the theoretical value to which the plotted
distribution should converge.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def beta_pdf(x, a, b):
    if False:
        return 10
    return x ** (a - 1) * (1 - x) ** (b - 1) * math.gamma(a + b) / (math.gamma(a) * math.gamma(b))

class UpdateDist:

    def __init__(self, ax, prob=0.5):
        if False:
            i = 10
            return i + 15
        self.success = 0
        self.prob = prob
        (self.line,) = ax.plot([], [], 'k-')
        self.x = np.linspace(0, 1, 200)
        self.ax = ax
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 10)
        self.ax.grid(True)
        self.ax.axvline(prob, linestyle='--', color='black')

    def start(self):
        if False:
            print('Hello World!')
        return (self.line,)

    def __call__(self, i):
        if False:
            return 10
        if i == 0:
            self.success = 0
            self.line.set_data([], [])
            return (self.line,)
        if np.random.rand() < self.prob:
            self.success += 1
        y = beta_pdf(self.x, self.success + 1, i - self.success + 1)
        self.line.set_data(self.x, y)
        return (self.line,)
np.random.seed(19680801)
(fig, ax) = plt.subplots()
ud = UpdateDist(ax, prob=0.7)
anim = FuncAnimation(fig, ud, init_func=ud.start, frames=100, interval=100, blit=True)
plt.show()