"""
=========================
Two subplots using pyplot
=========================

Create a figure with two subplots using `.pyplot.subplot`.
"""
import matplotlib.pyplot as plt
import numpy as np

def f(t):
    if False:
        print('Hello World!')
    return np.exp(-t) * np.cos(2 * np.pi * t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), color='tab:blue', marker='o')
plt.plot(t2, f(t2), color='black')
plt.subplot(212)
plt.plot(t2, np.cos(2 * np.pi * t2), color='tab:orange', linestyle='--')
plt.show()