"""
======================================================
Controlling view limits using margins and sticky_edges
======================================================

The first figure in this example shows how to zoom in and out of a
plot using `~.Axes.margins` instead of `~.Axes.set_xlim` and
`~.Axes.set_ylim`. The second figure demonstrates the concept of
edge "stickiness" introduced by certain methods and artists and how
to effectively work around that.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def f(t):
    if False:
        while True:
            i = 10
    return np.exp(-t) * np.cos(2 * np.pi * t)
t1 = np.arange(0.0, 3.0, 0.01)
ax1 = plt.subplot(212)
ax1.margins(0.05)
ax1.plot(t1, f(t1))
ax2 = plt.subplot(221)
ax2.margins(2, 2)
ax2.plot(t1, f(t1))
ax2.set_title('Zoomed out')
ax3 = plt.subplot(222)
ax3.margins(x=0, y=-0.25)
ax3.plot(t1, f(t1))
ax3.set_title('Zoomed in')
plt.show()
(y, x) = np.mgrid[:5, 1:6]
poly_coords = [(0.25, 2.75), (3.25, 2.75), (2.25, 0.75), (0.25, 0.75)]
(fig, (ax1, ax2)) = plt.subplots(ncols=2)
ax2.use_sticky_edges = False
for (ax, status) in zip((ax1, ax2), ('Is', 'Is Not')):
    cells = ax.pcolor(x, y, x + y, cmap='inferno', shading='auto')
    ax.add_patch(Polygon(poly_coords, color='forestgreen', alpha=0.5))
    ax.margins(x=0.1, y=0.05)
    ax.set_aspect('equal')
    ax.set_title(f'{status} Sticky')
plt.show()