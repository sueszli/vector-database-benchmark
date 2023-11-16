"""
==================
Animated histogram
==================

Use histogram's `.BarContainer` to draw a bunch of rectangles for an animated
histogram.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
np.random.seed(19680801)
HIST_BINS = np.linspace(-4, 4, 100)
data = np.random.randn(1000)
(n, _) = np.histogram(data, HIST_BINS)

def prepare_animation(bar_container):
    if False:
        i = 10
        return i + 15

    def animate(frame_number):
        if False:
            i = 10
            return i + 15
        data = np.random.randn(1000)
        (n, _) = np.histogram(data, HIST_BINS)
        for (count, rect) in zip(n, bar_container.patches):
            rect.set_height(count)
        return bar_container.patches
    return animate
(fig, ax) = plt.subplots()
(_, _, bar_container) = ax.hist(data, HIST_BINS, lw=1, ec='yellow', fc='green', alpha=0.5)
ax.set_ylim(top=55)
ani = animation.FuncAnimation(fig, prepare_animation(bar_container), 50, repeat=False, blit=True)
plt.show()