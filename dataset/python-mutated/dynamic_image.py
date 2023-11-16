"""
=================================================
Animated image using a precomputed list of images
=================================================

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
(fig, ax) = plt.subplots()

def f(x, y):
    if False:
        i = 10
        return i + 15
    return np.sin(x) + np.cos(y)
x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
ims = []
for i in range(60):
    x += np.pi / 15
    y += np.pi / 30
    im = ax.imshow(f(x, y), animated=True)
    if i == 0:
        ax.imshow(f(x, y))
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
plt.show()