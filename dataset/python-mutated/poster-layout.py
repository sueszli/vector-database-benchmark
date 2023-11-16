import numpy as np
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.text import TextPath
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.font_manager import FontProperties

def box(ax, index, x, y, width, height):
    if False:
        for i in range(10):
            print('nop')
    rectangle = mpatches.Rectangle((x, y), width, height, zorder=10, facecolor='none', edgecolor='black')
    ax.add_artist(rectangle)
    ax.text(x + (width - 1) / 2, y + (height - 1) / 2, '%d' % index, weight='bold', zorder=50, size='32', va='center', ha='center', color='k', alpha=0.25, family='GlassJaw BB')
fig = plt.figure(figsize=(6.5, 9.3))
ax = fig.add_axes([0, 0, 1, 1], aspect=1, frameon=False, xlim=(0, 65), ylim=(0, 93), xticks=[], yticks=[])
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
boxes = [(1, 1, 15, 21), (17, 1, 15, 10), (17, 12, 15, 10), (33, 1, 15, 15), (49, 1, 15, 15), (1, 23, 31, 8), (1, 32, 8, 14), (10, 32, 22, 14), (1, 47, 15, 13), (17, 47, 15, 13), (33, 45, 15, 15), (33, 32, 15, 12), (49, 32, 15, 18), (49, 51, 15, 9), (1, 61, 15, 15), (17, 61, 36, 15), (54, 61, 10, 15), (49, 77, 15, 15)]
for (index, (x, y, width, height)) in enumerate(boxes):
    box(ax, index, x, y, width, height)
ax.text(33, 17.25, 'Matplotlib can also be used to layout\na poster where each box can be filled\nwith different subplot.\n\nTo compute the boxes, just draw them\non a piece of graph of paper and use\nthe measure to get the bounds.\n\nThe cartoon fonts used in this example\nare "Lint McCree" and "GlassJaw"\n', size=6, color='k', family='Lint McCree Intl BB')
textpath = TextPath((8, 80), 'MATPLOTLIB', size=8, prop=FontProperties(family='GlassJaw BB'))
patch = mpatches.PathPatch(textpath, facecolor='none', edgecolor='none', zorder=5, joinstyle='round')
patch.set_path_effects([path_effects.Stroke(linewidth=8, foreground='black'), path_effects.Stroke(linewidth=6, foreground='yellow'), path_effects.Stroke(linewidth=3, foreground='black')])
transform = ax.transData + mpl.transforms.Affine2D().rotate_deg(2.5)
patch.set_transform(transform)
ax.add_artist(patch)
Z = np.linspace(1, 0, 100).reshape(100, 1)
im = ax.imshow(Z, cmap='autumn', extent=[1, 100, 79, 87], zorder=15)
im.set_transform(transform)
im.set_clip_path(patch._path, patch.get_transform())
ax.text(47, 79, 'Scientific visualization made simple & beautiful', color='black', zorder=20, family='Lint McCree Intl BB', weight='bold', size='xx-small', ha='right', va='baseline')
ax.text(8, 87, '  BSD  Licensed  ', color='white', zorder=30, rotation=-1.5, family='Lint McCree Intl BB', ha='center', va='center', size=7, bbox=dict(boxstyle='roundtooth', fc='k', ec='w', lw=1, pad=0.75))
I = imageio.imread('../data/John-Hunter-comic.png')
ax.imshow(I, extent=[49, 49 + 15, 77, 77 + 15], zorder=0, interpolation='bicubic')
ax.text(49.7, 77.5, 'John D. Hunter III', color='black', zorder=30, family='Lint McCree Intl BB', weight='bold', ha='left', va='bottom', size=5, bbox=dict(boxstyle='square', fc='w', ec='k', lw=1, pad=0.5))
plt.show()