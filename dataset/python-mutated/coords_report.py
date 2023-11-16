"""
=============
Coords Report
=============

Override the default reporting of coords as the mouse moves over the axes
in an interactive backend.
"""
import matplotlib.pyplot as plt
import numpy as np

def millions(x):
    if False:
        print('Hello World!')
    return '$%1.1fM' % (x * 1e-06)
np.random.seed(19680801)
x = np.random.rand(20)
y = 10000000.0 * np.random.rand(20)
(fig, ax) = plt.subplots()
ax.fmt_ydata = millions
plt.plot(x, y, 'o')
plt.show()