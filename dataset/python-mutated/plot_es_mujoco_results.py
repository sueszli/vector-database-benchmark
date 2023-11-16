import matplotlib.pyplot as plt
import numpy as np

def smooth(x):
    if False:
        i = 10
        return i + 15
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:i + 1].sum()) / (i - start + 1)
    return y
j = np.load('es_mujoco_results.npz')
returns = j['train']
plt.plot(returns)
plt.plot(smooth(np.array(returns)))
plt.title('Train returns')
plt.show()