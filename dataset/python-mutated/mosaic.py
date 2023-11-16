"""
.. redirect-from:: /tutorials/provisional/mosaic
.. redirect-from:: /gallery/subplots_axes_and_figures/mosaic

.. _mosaic:

========================================================
Complex and semantic figure composition (subplot_mosaic)
========================================================

Laying out Axes in a Figure in a non-uniform grid can be both tedious
and verbose.  For dense, even grids we have `.Figure.subplots` but for
more complex layouts, such as Axes that span multiple columns / rows
of the layout or leave some areas of the Figure blank, you can use
`.gridspec.GridSpec` (see :ref:`arranging_axes`) or
manually place your axes.  `.Figure.subplot_mosaic` aims to provide an
interface to visually lay out your axes (as either ASCII art or nested
lists) to streamline this process.

This interface naturally supports naming your axes.
`.Figure.subplot_mosaic` returns a dictionary keyed on the
labels used to lay out the Figure.  By returning data structures with
names, it is easier to write plotting code that is independent of the
Figure layout.


This is inspired by a `proposed MEP
<https://github.com/matplotlib/matplotlib/pull/4384>`__ and the
`patchwork <https://github.com/thomasp85/patchwork>`__ library for R.
While we do not implement the operator overloading style, we do
provide a Pythonic API for specifying (nested) Axes layouts.

"""
import matplotlib.pyplot as plt
import numpy as np

def identify_axes(ax_dict, fontsize=48):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper to identify the Axes in the examples below.\n\n    Draws the label in a large font in the center of the Axes.\n\n    Parameters\n    ----------\n    ax_dict : dict[str, Axes]\n        Mapping between the title / label and the Axes.\n    fontsize : int, optional\n        How big the label should be.\n    '
    kw = dict(ha='center', va='center', fontsize=fontsize, color='darkgrey')
    for (k, ax) in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)
np.random.seed(19680801)
hist_data = np.random.randn(1500)
fig = plt.figure(layout='constrained')
ax_array = fig.subplots(2, 2, squeeze=False)
ax_array[0, 0].bar(['a', 'b', 'c'], [5, 7, 9])
ax_array[0, 1].plot([1, 2, 3])
ax_array[1, 0].hist(hist_data, bins='auto')
ax_array[1, 1].imshow([[1, 2], [2, 1]])
identify_axes({(j, k): a for (j, r) in enumerate(ax_array) for (k, a) in enumerate(r)})
fig = plt.figure(layout='constrained')
ax_dict = fig.subplot_mosaic([['bar', 'plot'], ['hist', 'image']])
ax_dict['bar'].bar(['a', 'b', 'c'], [5, 7, 9])
ax_dict['plot'].plot([1, 2, 3])
ax_dict['hist'].hist(hist_data)
ax_dict['image'].imshow([[1, 2], [2, 1]])
identify_axes(ax_dict)
print(ax_dict)
mosaic = '\n    AB\n    CD\n    '
fig = plt.figure(layout='constrained')
ax_dict = fig.subplot_mosaic(mosaic)
identify_axes(ax_dict)
mosaic = 'AB;CD'
fig = plt.figure(layout='constrained')
ax_dict = fig.subplot_mosaic(mosaic)
identify_axes(ax_dict)
axd = plt.figure(layout='constrained').subplot_mosaic('\n    ABD\n    CCD\n    ')
identify_axes(axd)
axd = plt.figure(layout='constrained').subplot_mosaic('\n    A.C\n    BBB\n    .D.\n    ')
identify_axes(axd)
axd = plt.figure(layout='constrained').subplot_mosaic('\n    aX\n    Xb\n    ', empty_sentinel='X')
identify_axes(axd)
axd = plt.figure(layout='constrained').subplot_mosaic('αб\n       ℝ☢')
identify_axes(axd)
axd = plt.figure(layout='constrained').subplot_mosaic('\n    .a.\n    bAc\n    .d.\n    ', height_ratios=[1, 3.5, 1], width_ratios=[1, 3.5, 1])
identify_axes(axd)
mosaic = 'AA\n            BC'
fig = plt.figure()
axd = fig.subplot_mosaic(mosaic, gridspec_kw={'bottom': 0.25, 'top': 0.95, 'left': 0.1, 'right': 0.5, 'wspace': 0.5, 'hspace': 0.5})
identify_axes(axd)
axd = fig.subplot_mosaic(mosaic, gridspec_kw={'bottom': 0.05, 'top': 0.75, 'left': 0.6, 'right': 0.95, 'wspace': 0.5, 'hspace': 0.5})
identify_axes(axd)
mosaic = 'AA\n            BC'
fig = plt.figure(layout='constrained')
(left, right) = fig.subfigures(nrows=1, ncols=2)
axd = left.subplot_mosaic(mosaic)
identify_axes(axd)
axd = right.subplot_mosaic(mosaic)
identify_axes(axd)
axd = plt.figure(layout='constrained').subplot_mosaic('AB', subplot_kw={'projection': 'polar'})
identify_axes(axd)
(fig, axd) = plt.subplot_mosaic('AB;CD', per_subplot_kw={'A': {'projection': 'polar'}, ('C', 'D'): {'xscale': 'log'}})
identify_axes(axd)
(fig, axd) = plt.subplot_mosaic('AB;CD', per_subplot_kw={'AD': {'projection': 'polar'}, 'BC': {'facecolor': '.9'}})
identify_axes(axd)
axd = plt.figure(layout='constrained').subplot_mosaic('AB;CD', subplot_kw={'facecolor': 'xkcd:tangerine'}, per_subplot_kw={'B': {'facecolor': 'xkcd:water blue'}, 'D': {'projection': 'polar', 'facecolor': 'w'}})
identify_axes(axd)
axd = plt.figure(layout='constrained').subplot_mosaic([['main', 'zoom'], ['main', 'BLANK']], empty_sentinel='BLANK', width_ratios=[2, 1])
identify_axes(axd)
inner = [['inner A'], ['inner B']]
outer_nested_mosaic = [['main', inner], ['bottom', 'bottom']]
axd = plt.figure(layout='constrained').subplot_mosaic(outer_nested_mosaic, empty_sentinel=None)
identify_axes(axd, fontsize=36)
mosaic = np.zeros((4, 4), dtype=int)
for j in range(4):
    mosaic[j, j] = j + 1
axd = plt.figure(layout='constrained').subplot_mosaic(mosaic, empty_sentinel=0)
identify_axes(axd)