"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.

Tutorial reference:
http://matplotlib.org/examples/animation/simple_anim.html

More animation example code:
http://matplotlib.org/examples/animation/
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
(fig, ax) = plt.subplots()
x = np.arange(0, 2 * np.pi, 0.01)
(line,) = ax.plot(x, np.sin(x))

def animate(i):
    if False:
        while True:
            i = 10
    line.set_ydata(np.sin(x + i / 10.0))
    return (line,)

def init():
    if False:
        return 10
    line.set_ydata(np.sin(x))
    return (line,)
ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init, interval=20, blit=False)
plt.show()