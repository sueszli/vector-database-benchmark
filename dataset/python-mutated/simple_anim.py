"""
==================
Animated line plot
==================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
(fig, ax) = plt.subplots()
x = np.arange(0, 2 * np.pi, 0.01)
(line,) = ax.plot(x, np.sin(x))

def animate(i):
    if False:
        return 10
    line.set_ydata(np.sin(x + i / 50))
    return (line,)
ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=50)
plt.show()