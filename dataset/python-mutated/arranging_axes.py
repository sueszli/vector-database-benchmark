"""
.. redirect-from:: /tutorials/intermediate/gridspec
.. redirect-from:: /tutorials/intermediate/arranging_axes

.. _arranging_axes:

===================================
Arranging multiple Axes in a Figure
===================================

Often more than one Axes is wanted on a figure at a time, usually
organized into a regular grid.  Matplotlib has a variety of tools for
working with grids of Axes that have evolved over the history of the library.
Here we will discuss the tools we think users should use most often, the tools
that underpin how Axes are organized, and mention some of the older tools.

.. note::

    Matplotlib uses *Axes* to refer to the drawing area that contains
    data, x- and y-axis, ticks, labels, title, etc. See :ref:`figure_parts`
    for more details.  Another term that is often used is "subplot", which
    refers to an Axes that is in a grid with other Axes objects.

Overview
========

Create grid-shaped combinations of Axes
---------------------------------------

`~matplotlib.pyplot.subplots`
    The primary function used to create figures and a grid of Axes.  It
    creates and places all Axes on the figure at once, and returns an
    object array with handles for the Axes in the grid.  See
    `.Figure.subplots`.

or

`~matplotlib.pyplot.subplot_mosaic`
    A simple way to create figures and a grid of Axes, with the added
    flexibility that Axes can also span rows or columns. The Axes are returned
    in a labelled dictionary instead of an array.  See also
    `.Figure.subplot_mosaic` and
    :ref:`mosaic`.

Sometimes it is natural to have more than one distinct group of Axes grids,
in which case Matplotlib has the concept of `.SubFigure`:

`~matplotlib.figure.SubFigure`
    A virtual figure within a figure.

Underlying tools
----------------

Underlying these are the concept of a `~.gridspec.GridSpec` and
a `~.SubplotSpec`:

`~matplotlib.gridspec.GridSpec`
    Specifies the geometry of the grid that a subplot will be
    placed. The number of rows and number of columns of the grid
    need to be set. Optionally, the subplot layout parameters
    (e.g., left, right, etc.) can be tuned.

`~matplotlib.gridspec.SubplotSpec`
    Specifies the location of the subplot in the given `.GridSpec`.

.. _fixed_size_axes:

Adding single Axes at a time
----------------------------

The above functions create all Axes in a single function call.  It is also
possible to add Axes one at a time, and this was originally how Matplotlib
used to work.  Doing so is generally less elegant and flexible, though
sometimes useful for interactive work or to place an Axes in a custom
location:

`~matplotlib.figure.Figure.add_axes`
    Adds a single axes at a location specified by
    ``[left, bottom, width, height]`` in fractions of figure width or height.

`~matplotlib.pyplot.subplot` or `.Figure.add_subplot`
    Adds a single subplot on a figure, with 1-based indexing (inherited from
    Matlab).  Columns and rows can be spanned by specifying a range of grid
    cells.

`~matplotlib.pyplot.subplot2grid`
    Similar to `.pyplot.subplot`, but uses 0-based indexing and two-d python
    slicing to choose cells.

"""
import matplotlib.pyplot as plt
import numpy as np
(w, h) = (4, 3)
margin = 0.5
fig = plt.figure(figsize=(w, h), facecolor='lightblue')
ax = fig.add_axes([margin / w, margin / h, (w - 2 * margin) / w, (h - 2 * margin) / h])
(fig, axs) = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5), layout='constrained')
for row in range(2):
    for col in range(2):
        axs[row, col].annotate(f'axs[{row}, {col}]', (0.5, 0.5), transform=axs[row, col].transAxes, ha='center', va='center', fontsize=18, color='darkgrey')
fig.suptitle('plt.subplots()')

def annotate_axes(ax, text, fontsize=18):
    if False:
        print('Hello World!')
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center', fontsize=fontsize, color='darkgrey')
(fig, axd) = plt.subplot_mosaic([['upper left', 'upper right'], ['lower left', 'lower right']], figsize=(5.5, 3.5), layout='constrained')
for (k, ax) in axd.items():
    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
fig.suptitle('plt.subplot_mosaic()')
(fig, axs) = plt.subplots(2, 2, layout='constrained', figsize=(5.5, 3.5), facecolor='lightblue')
for ax in axs.flat:
    ax.set_aspect(1)
fig.suptitle('Fixed aspect Axes')
(fig, axs) = plt.subplots(2, 2, layout='compressed', figsize=(5.5, 3.5), facecolor='lightblue')
for ax in axs.flat:
    ax.set_aspect(1)
fig.suptitle('Fixed aspect Axes: compressed')
(fig, axd) = plt.subplot_mosaic([['upper left', 'right'], ['lower left', 'right']], figsize=(5.5, 3.5), layout='constrained')
for (k, ax) in axd.items():
    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
fig.suptitle('plt.subplot_mosaic()')
gs_kw = dict(width_ratios=[1.4, 1], height_ratios=[1, 2])
(fig, axd) = plt.subplot_mosaic([['upper left', 'right'], ['lower left', 'right']], gridspec_kw=gs_kw, figsize=(5.5, 3.5), layout='constrained')
for (k, ax) in axd.items():
    annotate_axes(ax, f'axd[{k!r}]', fontsize=14)
fig.suptitle('plt.subplot_mosaic()')
fig = plt.figure(layout='constrained')
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1.5, 1.0])
axs0 = subfigs[0].subplots(2, 2)
subfigs[0].set_facecolor('lightblue')
subfigs[0].suptitle('subfigs[0]\nLeft side')
subfigs[0].supxlabel('xlabel for subfigs[0]')
axs1 = subfigs[1].subplots(3, 1)
subfigs[1].suptitle('subfigs[1]')
subfigs[1].supylabel('ylabel for subfigs[1]')
inner = [['innerA'], ['innerB']]
outer = [['upper left', inner], ['lower left', 'lower right']]
(fig, axd) = plt.subplot_mosaic(outer, layout='constrained')
for (k, ax) in axd.items():
    annotate_axes(ax, f'axd[{k!r}]')
fig = plt.figure(figsize=(5.5, 3.5), layout='constrained')
spec = fig.add_gridspec(ncols=2, nrows=2)
ax0 = fig.add_subplot(spec[0, 0])
annotate_axes(ax0, 'ax0')
ax1 = fig.add_subplot(spec[0, 1])
annotate_axes(ax1, 'ax1')
ax2 = fig.add_subplot(spec[1, 0])
annotate_axes(ax2, 'ax2')
ax3 = fig.add_subplot(spec[1, 1])
annotate_axes(ax3, 'ax3')
fig.suptitle('Manually added subplots using add_gridspec')
fig = plt.figure(figsize=(5.5, 3.5), layout='constrained')
spec = fig.add_gridspec(2, 2)
ax0 = fig.add_subplot(spec[0, :])
annotate_axes(ax0, 'ax0')
ax10 = fig.add_subplot(spec[1, 0])
annotate_axes(ax10, 'ax10')
ax11 = fig.add_subplot(spec[1, 1])
annotate_axes(ax11, 'ax11')
fig.suptitle('Manually added subplots, spanning a column')
fig = plt.figure(layout=None, facecolor='lightblue')
gs = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.75, hspace=0.1, wspace=0.05)
ax0 = fig.add_subplot(gs[:-1, :])
annotate_axes(ax0, 'ax0')
ax1 = fig.add_subplot(gs[-1, :-1])
annotate_axes(ax1, 'ax1')
ax2 = fig.add_subplot(gs[-1, -1])
annotate_axes(ax2, 'ax2')
fig.suptitle('Manual gridspec with right=0.75')
fig = plt.figure(layout='constrained')
gs0 = fig.add_gridspec(1, 2)
gs00 = gs0[0].subgridspec(2, 2)
gs01 = gs0[1].subgridspec(3, 1)
for a in range(2):
    for b in range(2):
        ax = fig.add_subplot(gs00[a, b])
        annotate_axes(ax, f'axLeft[{a}, {b}]', fontsize=10)
        if a == 1 and b == 1:
            ax.set_xlabel('xlabel')
for a in range(3):
    ax = fig.add_subplot(gs01[a])
    annotate_axes(ax, f'axRight[{a}, {b}]')
    if a == 2:
        ax.set_ylabel('ylabel')
fig.suptitle('nested gridspecs')

def squiggle_xy(a, b, c, d, i=np.arange(0.0, 2 * np.pi, 0.05)):
    if False:
        i = 10
        return i + 15
    return (np.sin(i * a) * np.cos(i * b), np.sin(i * c) * np.cos(i * d))
fig = plt.figure(figsize=(8, 8), layout='constrained')
outer_grid = fig.add_gridspec(4, 4, wspace=0, hspace=0)
for a in range(4):
    for b in range(4):
        inner_grid = outer_grid[a, b].subgridspec(3, 3, wspace=0, hspace=0)
        axs = inner_grid.subplots()
        for ((c, d), ax) in np.ndenumerate(axs):
            ax.plot(*squiggle_xy(a + 1, b + 1, c + 1, d + 1))
            ax.set(xticks=[], yticks=[])
for ax in fig.get_axes():
    ss = ax.get_subplotspec()
    ax.spines.top.set_visible(ss.is_first_row())
    ax.spines.bottom.set_visible(ss.is_last_row())
    ax.spines.left.set_visible(ss.is_first_col())
    ax.spines.right.set_visible(ss.is_last_col())
plt.show()