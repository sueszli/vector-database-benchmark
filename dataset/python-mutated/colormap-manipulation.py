"""
.. redirect-from:: /tutorials/colors/colormap-manipulation

.. _colormap-manipulation:

******************
Creating Colormaps
******************

Matplotlib has a number of built-in colormaps accessible via
`.matplotlib.colormaps`.  There are also external libraries like
palettable_ that have many extra colormaps.

.. _palettable: https://jiffyclub.github.io/palettable/

However, we may also want to create or manipulate our own colormaps.
This can be done using the class `.ListedColormap` or
`.LinearSegmentedColormap`.
Both colormap classes map values between 0 and 1 to colors. There are however
differences, as explained below.

Before manually creating or manipulating colormaps, let us first see how we
can obtain colormaps and their colors from existing colormap classes.

Getting colormaps and accessing their values
============================================

First, getting a named colormap, most of which are listed in
:ref:`colormaps`, may be done using `.matplotlib.colormaps`,
which returns a colormap object.  The length of the list of colors used
internally to define the colormap can be adjusted via `.Colormap.resampled`.
Below we use a modest value of 8 so there are not a lot of values to look at.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
viridis = mpl.colormaps['viridis'].resampled(8)
print(viridis(0.56))
print('viridis.colors', viridis.colors)
print('viridis(range(8))', viridis(range(8)))
print('viridis(np.linspace(0, 1, 8))', viridis(np.linspace(0, 1, 8)))
print('viridis(np.linspace(0, 1, 12))', viridis(np.linspace(0, 1, 12)))
copper = mpl.colormaps['copper'].resampled(8)
print('copper(range(8))', copper(range(8)))
print('copper(np.linspace(0, 1, 8))', copper(np.linspace(0, 1, 8)))

def plot_examples(colormaps):
    if False:
        while True:
            i = 10
    '\n    Helper function to plot data with associated colormap.\n    '
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    (fig, axs) = plt.subplots(1, n, figsize=(n * 2 + 2, 3), layout='constrained', squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()
cmap = ListedColormap(['darkorange', 'gold', 'lawngreen', 'lightseagreen'])
plot_examples([cmap])
viridis = mpl.colormaps['viridis'].resampled(256)
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)
plot_examples([viridis, newcmp])
viridis_big = mpl.colormaps['viridis']
newcmp = ListedColormap(viridis_big(np.linspace(0.25, 0.75, 128)))
plot_examples([viridis, newcmp])
top = mpl.colormaps['Oranges_r'].resampled(128)
bottom = mpl.colormaps['Blues'].resampled(128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')
plot_examples([viridis, newcmp])
N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(90 / 256, 1, N)
vals[:, 1] = np.linspace(40 / 256, 1, N)
vals[:, 2] = np.linspace(40 / 256, 1, N)
newcmp = ListedColormap(vals)
plot_examples([viridis, newcmp])
cdict = {'red': [[0.0, 0.0, 0.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]], 'green': [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 1.0, 1.0], [1.0, 1.0, 1.0]], 'blue': [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]]}

def plot_linearmap(cdict):
    if False:
        return 10
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    (fig, ax) = plt.subplots(figsize=(4, 3), layout='constrained')
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256) / 256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    plt.show()
plot_linearmap(cdict)
cdict['red'] = [[0.0, 0.0, 0.3], [0.5, 1.0, 0.9], [1.0, 1.0, 1.0]]
plot_linearmap(cdict)
colors = ['darkorange', 'gold', 'lawngreen', 'lightseagreen']
cmap1 = LinearSegmentedColormap.from_list('mycmap', colors)
nodes = [0.0, 0.4, 0.8, 1.0]
cmap2 = LinearSegmentedColormap.from_list('mycmap', list(zip(nodes, colors)))
plot_examples([cmap1, cmap2])
colors = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
my_cmap = ListedColormap(colors, name='my_cmap')
my_cmap_r = my_cmap.reversed()
plot_examples([my_cmap, my_cmap_r])
mpl.colormaps.register(cmap=my_cmap)
mpl.colormaps.register(cmap=my_cmap_r)
data = [[1, 2, 3, 4, 5]]
(fig, (ax1, ax2)) = plt.subplots(nrows=2)
ax1.imshow(data, cmap='my_cmap')
ax2.imshow(data, cmap='my_cmap_r')
plt.show()