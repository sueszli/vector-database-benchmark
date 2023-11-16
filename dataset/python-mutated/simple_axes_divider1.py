"""
=====================
Simple Axes Divider 1
=====================

See also :ref:`axes_grid`.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

def label_axes(ax, text):
    if False:
        print('Hello World!')
    'Place a label at the center of an Axes, and remove the axis ticks.'
    ax.text(0.5, 0.5, text, transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
fig = plt.figure(figsize=(6, 6))
fig.suptitle('Fixed axes sizes, fixed paddings')
horiz = [Size.Fixed(1.0), Size.Fixed(0.5), Size.Fixed(1.5), Size.Fixed(0.5)]
vert = [Size.Fixed(1.5), Size.Fixed(0.5), Size.Fixed(1.0)]
rect = (0.1, 0.1, 0.8, 0.8)
div = Divider(fig, rect, horiz, vert, aspect=False)
ax1 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0))
label_axes(ax1, 'nx=0, ny=0')
ax2 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=2))
label_axes(ax2, 'nx=0, ny=2')
ax3 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=2))
label_axes(ax3, 'nx=2, ny=2')
ax4 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, nx1=4, ny=0))
label_axes(ax4, 'nx=2, nx1=4, ny=0')
fig = plt.figure(figsize=(6, 6))
fig.suptitle('Scalable axes sizes, fixed paddings')
horiz = [Size.Scaled(1.5), Size.Fixed(0.5), Size.Scaled(1.0), Size.Scaled(0.5)]
vert = [Size.Scaled(1.0), Size.Fixed(0.5), Size.Scaled(1.5)]
rect = (0.1, 0.1, 0.8, 0.8)
div = Divider(fig, rect, horiz, vert, aspect=False)
ax1 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=0))
label_axes(ax1, 'nx=0, ny=0')
ax2 = fig.add_axes(rect, axes_locator=div.new_locator(nx=0, ny=2))
label_axes(ax2, 'nx=0, ny=2')
ax3 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, ny=2))
label_axes(ax3, 'nx=2, ny=2')
ax4 = fig.add_axes(rect, axes_locator=div.new_locator(nx=2, nx1=4, ny=0))
label_axes(ax4, 'nx=2, nx1=4, ny=0')
plt.show()