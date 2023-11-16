"""

.. redirect-from:: /tutorials/intermediate/constrainedlayout_guide

.. _constrainedlayout_guide:

========================
Constrained Layout Guide
========================

Use *constrained layout* to fit plots within your figure cleanly.

*Constrained layout* automatically adjusts subplots so that decorations like tick
labels, legends, and colorbars do not overlap, while still preserving the
logical layout requested by the user.

*Constrained layout* is similar to :ref:`Tight
layout<tight_layout_guide>`, but is substantially more
flexible.  It handles colorbars placed on multiple Axes
(:ref:`colorbar_placement`) nested layouts (`~.Figure.subfigures`) and Axes that
span rows or columns (`~.pyplot.subplot_mosaic`), striving to align spines from
Axes in the same row or column.  In addition, :ref:`Compressed layout
<compressed_layout>` will try and move fixed aspect-ratio Axes closer together.
These features are described in this document, as well as some
:ref:`implementation details <cl_notes_on_algorithm>` discussed at the end.

*Constrained layout* typically needs to be activated before any Axes are added to
a figure. Two ways of doing so are

* using the respective argument to `~.pyplot.subplots`,
  `~.pyplot.figure`, `~.pyplot.subplot_mosaic` e.g.::

      plt.subplots(layout="constrained")

* activate it via :ref:`rcParams<customizing-with-dynamic-rc-settings>`, like::

      plt.rcParams['figure.constrained_layout.use'] = True

Those are described in detail throughout the following sections.

.. warning::

    Calling `~.pyplot.tight_layout` will turn off *constrained layout*!

Simple example
==============

With the default Axes positioning, the axes title, axis labels, or tick labels
can sometimes go outside the figure area, and thus get clipped.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
plt.rcParams['savefig.facecolor'] = '0.8'
plt.rcParams['figure.figsize'] = (4.5, 4.0)
plt.rcParams['figure.max_open_warning'] = 50

def example_plot(ax, fontsize=12, hide_labels=False):
    if False:
        return 10
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    if hide_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
(fig, ax) = plt.subplots(layout=None)
example_plot(ax, fontsize=24)
(fig, ax) = plt.subplots(layout='constrained')
example_plot(ax, fontsize=24)
(fig, axs) = plt.subplots(2, 2, layout=None)
for ax in axs.flat:
    example_plot(ax)
(fig, axs) = plt.subplots(2, 2, layout='constrained')
for ax in axs.flat:
    example_plot(ax)
arr = np.arange(100).reshape((10, 10))
norm = mcolors.Normalize(vmin=0.0, vmax=100.0)
pc_kwargs = {'rasterized': True, 'cmap': 'viridis', 'norm': norm}
(fig, ax) = plt.subplots(figsize=(4, 4), layout='constrained')
im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=ax, shrink=0.6)
(fig, axs) = plt.subplots(2, 2, figsize=(4, 4), layout='constrained')
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)
(fig, axs) = plt.subplots(3, 3, figsize=(4, 4), layout='constrained')
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs[1:, 1], shrink=0.8)
fig.colorbar(im, ax=axs[:, -1], shrink=0.6)
(fig, axs) = plt.subplots(2, 2, figsize=(4, 4), layout='constrained')
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)
fig.suptitle('Big Suptitle')
(fig, ax) = plt.subplots(layout='constrained')
ax.plot(np.arange(10), label='This is a plot')
ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
(fig, axs) = plt.subplots(1, 2, figsize=(4, 2), layout='constrained')
axs[0].plot(np.arange(10))
axs[1].plot(np.arange(10), label='This is a plot')
axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
(fig, axs) = plt.subplots(1, 2, figsize=(4, 2), layout='constrained')
axs[0].plot(np.arange(10))
axs[1].plot(np.arange(10), label='This is a plot')
leg = axs[1].legend(loc='center left', bbox_to_anchor=(0.8, 0.5))
leg.set_in_layout(False)
fig.canvas.draw()
leg.set_in_layout(True)
fig.set_layout_engine('none')
try:
    fig.savefig('../../../doc/_static/constrained_layout_1b.png', bbox_inches='tight', dpi=100)
except FileNotFoundError:
    pass
(fig, axs) = plt.subplots(1, 2, figsize=(4, 2), layout='constrained')
axs[0].plot(np.arange(10))
lines = axs[1].plot(np.arange(10), label='This is a plot')
labels = [l.get_label() for l in lines]
leg = fig.legend(lines, labels, loc='center left', bbox_to_anchor=(0.8, 0.5), bbox_transform=axs[1].transAxes)
try:
    fig.savefig('../../../doc/_static/constrained_layout_2b.png', bbox_inches='tight', dpi=100)
except FileNotFoundError:
    pass
(fig, axs) = plt.subplots(2, 2, layout='constrained')
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0, wspace=0)
(fig, axs) = plt.subplots(2, 2, layout='constrained')
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2, wspace=0.2)
(fig, axs) = plt.subplots(2, 3, layout='constrained')
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2, wspace=0.2)
(fig, axs) = plt.subplots(2, 2, layout='constrained', gridspec_kw={'wspace': 0.3, 'hspace': 0.2})
for ax in axs.flat:
    example_plot(ax, hide_labels=True)
fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.0, wspace=0.0)
(fig, axs) = plt.subplots(2, 2, layout='constrained')
pads = [0, 0.05, 0.1, 0.2]
for (pad, ax) in zip(pads, axs.flat):
    pc = ax.pcolormesh(arr, **pc_kwargs)
    fig.colorbar(pc, ax=ax, shrink=0.6, pad=pad)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'pad: {pad}')
fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.2, wspace=0.2)
plt.rcParams['figure.constrained_layout.use'] = True
(fig, axs) = plt.subplots(2, 2, figsize=(3, 3))
for ax in axs.flat:
    example_plot(ax)
plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(layout='constrained')
gs1 = gridspec.GridSpec(2, 1, figure=fig)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
example_plot(ax1)
example_plot(ax2)
fig = plt.figure(layout='constrained')
gs0 = fig.add_gridspec(1, 2)
gs1 = gs0[0].subgridspec(2, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
example_plot(ax1)
example_plot(ax2)
gs2 = gs0[1].subgridspec(3, 1)
for ss in gs2:
    ax = fig.add_subplot(ss)
    example_plot(ax)
    ax.set_title('')
    ax.set_xlabel('')
ax.set_xlabel('x-label', fontsize=12)
fig = plt.figure(figsize=(4, 6), layout='constrained')
gs0 = fig.add_gridspec(6, 2)
ax1 = fig.add_subplot(gs0[:3, 0])
ax2 = fig.add_subplot(gs0[3:, 0])
example_plot(ax1)
example_plot(ax2)
ax = fig.add_subplot(gs0[0:2, 1])
example_plot(ax, hide_labels=True)
ax = fig.add_subplot(gs0[2:4, 1])
example_plot(ax, hide_labels=True)
ax = fig.add_subplot(gs0[4:, 1])
example_plot(ax, hide_labels=True)
fig.suptitle('Overlapping Gridspecs')
fig = plt.figure(layout='constrained')
gs0 = fig.add_gridspec(1, 2, figure=fig, width_ratios=[1, 2])
gs_left = gs0[0].subgridspec(2, 1)
gs_right = gs0[1].subgridspec(2, 2)
for gs in gs_left:
    ax = fig.add_subplot(gs)
    example_plot(ax)
axs = []
for gs in gs_right:
    ax = fig.add_subplot(gs)
    pcm = ax.pcolormesh(arr, **pc_kwargs)
    ax.set_xlabel('x-label')
    ax.set_ylabel('y-label')
    ax.set_title('title')
    axs += [ax]
fig.suptitle('Nested plots using subgridspec')
fig.colorbar(pcm, ax=axs)
fig = plt.figure(layout='constrained')
sfigs = fig.subfigures(1, 2, width_ratios=[1, 2])
axs_left = sfigs[0].subplots(2, 1)
for ax in axs_left.flat:
    example_plot(ax)
axs_right = sfigs[1].subplots(2, 2)
for ax in axs_right.flat:
    pcm = ax.pcolormesh(arr, **pc_kwargs)
    ax.set_xlabel('x-label')
    ax.set_ylabel('y-label')
    ax.set_title('title')
fig.colorbar(pcm, ax=axs_right)
fig.suptitle('Nested plots using subfigures')
(fig, axs) = plt.subplots(1, 2, layout='constrained')
example_plot(axs[0], fontsize=12)
axs[1].set_position([0.2, 0.2, 0.4, 0.4])
(fig, axs) = plt.subplots(2, 2, figsize=(5, 3), sharex=True, sharey=True, layout='constrained')
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='constrained'")
(fig, axs) = plt.subplots(2, 2, figsize=(5, 3), sharex=True, sharey=True, layout='compressed')
for ax in axs.flat:
    ax.imshow(arr)
fig.suptitle("fixed-aspect plots, layout='compressed'")
fig = plt.figure(layout='constrained')
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(2, 2, (2, 4))
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
plt.suptitle('Homogenous nrows, ncols')
fig = plt.figure(layout='constrained')
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 3)
ax3 = plt.subplot(1, 2, 2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
plt.suptitle('Mixed nrows, ncols')
fig = plt.figure(layout='constrained')
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
fig.suptitle('subplot2grid')
from matplotlib._layoutgrid import plot_children
(fig, ax) = plt.subplots(layout='constrained')
example_plot(ax, fontsize=24)
plot_children(fig)
(fig, ax) = plt.subplots(1, 2, layout='constrained')
example_plot(ax[0], fontsize=32)
example_plot(ax[1], fontsize=8)
plot_children(fig)
(fig, ax) = plt.subplots(1, 2, layout='constrained')
im = ax[0].pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=ax[0], shrink=0.6)
im = ax[1].pcolormesh(arr, **pc_kwargs)
plot_children(fig)
(fig, axs) = plt.subplots(2, 2, layout='constrained')
for ax in axs.flat:
    im = ax.pcolormesh(arr, **pc_kwargs)
fig.colorbar(im, ax=axs, shrink=0.6)
plot_children(fig)
fig = plt.figure(layout='constrained')
gs = gridspec.GridSpec(2, 2, figure=fig)
ax = fig.add_subplot(gs[:, 0])
im = ax.pcolormesh(arr, **pc_kwargs)
ax = fig.add_subplot(gs[0, 1])
im = ax.pcolormesh(arr, **pc_kwargs)
ax = fig.add_subplot(gs[1, 1])
im = ax.pcolormesh(arr, **pc_kwargs)
plot_children(fig)
fig = plt.figure(layout='constrained')
gs = fig.add_gridspec(2, 4)
ax00 = fig.add_subplot(gs[0, 0:2])
ax01 = fig.add_subplot(gs[0, 2:])
ax10 = fig.add_subplot(gs[1, 1:3])
example_plot(ax10, fontsize=14)
plot_children(fig)
plt.show()