"""
=====
Decay
=====

This example showcases:

- using a generator to drive an animation,
- changing axes limits during an animation.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def data_gen():
    if False:
        print('Hello World!')
    for cnt in itertools.count():
        t = cnt / 10
        yield (t, np.sin(2 * np.pi * t) * np.exp(-t / 10.0))

def init():
    if False:
        print('Hello World!')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 1)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return (line,)
(fig, ax) = plt.subplots()
(line,) = ax.plot([], [], lw=2)
ax.grid()
(xdata, ydata) = ([], [])

def run(data):
    if False:
        i = 10
        return i + 15
    (t, y) = data
    xdata.append(t)
    ydata.append(y)
    (xmin, xmax) = ax.get_xlim()
    if t >= xmax:
        ax.set_xlim(xmin, 2 * xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)
    return (line,)
ani = animation.FuncAnimation(fig, run, data_gen, interval=100, init_func=init, save_count=100)
plt.show()