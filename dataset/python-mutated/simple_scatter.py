"""
=============================
Animated scatter saved as GIF
=============================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
(fig, ax) = plt.subplots()
ax.set_xlim([0, 10])
scat = ax.scatter(1, 0)
x = np.linspace(0, 10)

def animate(i):
    if False:
        i = 10
        return i + 15
    scat.set_offsets((x[i], 0))
    return (scat,)
ani = animation.FuncAnimation(fig, animate, repeat=True, frames=len(x) - 1, interval=50)
plt.show()