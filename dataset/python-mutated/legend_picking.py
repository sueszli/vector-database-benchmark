"""
==============
Legend picking
==============

Enable picking on the legend to toggle the original line on and off

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
import matplotlib.pyplot as plt
import numpy as np
t = np.linspace(0, 1)
y1 = 2 * np.sin(2 * np.pi * t)
y2 = 4 * np.sin(2 * np.pi * 2 * t)
(fig, ax) = plt.subplots()
ax.set_title('Click on legend line to toggle line on/off')
(line1,) = ax.plot(t, y1, lw=2, label='1 Hz')
(line2,) = ax.plot(t, y2, lw=2, label='2 Hz')
leg = ax.legend(fancybox=True, shadow=True)
lines = [line1, line2]
map_legend_to_ax = {}
pickradius = 5
for (legend_line, ax_line) in zip(leg.get_lines(), lines):
    legend_line.set_picker(pickradius)
    map_legend_to_ax[legend_line] = ax_line

def on_pick(event):
    if False:
        print('Hello World!')
    legend_line = event.artist
    if legend_line not in map_legend_to_ax:
        return
    ax_line = map_legend_to_ax[legend_line]
    visible = not ax_line.get_visible()
    ax_line.set_visible(visible)
    legend_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()
fig.canvas.mpl_connect('pick_event', on_pick)
leg.set_draggable(True)
plt.show()