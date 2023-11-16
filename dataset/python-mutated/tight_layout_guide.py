"""
.. redirect-from:: /tutorial/intermediate/tight_layout_guide

.. _tight_layout_guide:

==================
Tight Layout guide
==================

How to use tight-layout to fit plots within your figure cleanly.

*tight_layout* automatically adjusts subplot params so that the
subplot(s) fits in to the figure area. This is an experimental
feature and may not work for some cases. It only checks the extents
of ticklabels, axis labels, and titles.

An alternative to *tight_layout* is :ref:`constrained_layout
<constrainedlayout_guide>`.

Simple example
==============

With the default Axes positioning, the axes title, axis labels, or tick labels
can sometimes go outside the figure area, and thus get clipped.
"""
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['savefig.facecolor'] = '0.8'

def example_plot(ax, fontsize=12):
    if False:
        print('Hello World!')
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)
plt.close('all')
(fig, ax) = plt.subplots()
example_plot(ax, fontsize=24)
(fig, ax) = plt.subplots()
example_plot(ax, fontsize=24)
plt.tight_layout()
plt.close('all')
(fig, ((ax1, ax2), (ax3, ax4))) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
(fig, ((ax1, ax2), (ax3, ax4))) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()
(fig, ((ax1, ax2), (ax3, ax4))) = plt.subplots(nrows=2, ncols=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.close('all')
fig = plt.figure()
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)
ax3 = plt.subplot(122)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
plt.tight_layout()
plt.close('all')
fig = plt.figure()
ax1 = plt.subplot2grid((3, 3), (0, 0))
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
example_plot(ax1)
example_plot(ax2)
example_plot(ax3)
example_plot(ax4)
plt.tight_layout()
arr = np.arange(100).reshape((10, 10))
plt.close('all')
fig = plt.figure(figsize=(5, 4))
ax = plt.subplot()
im = ax.imshow(arr, interpolation='none')
plt.tight_layout()
import matplotlib.gridspec as gridspec
plt.close('all')
fig = plt.figure()
gs1 = gridspec.GridSpec(2, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
example_plot(ax1)
example_plot(ax2)
gs1.tight_layout(fig)
fig = plt.figure()
gs1 = gridspec.GridSpec(2, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
example_plot(ax1)
example_plot(ax2)
gs1.tight_layout(fig, rect=[0, 0, 0.5, 1.0])
(fig, ax) = plt.subplots(figsize=(4, 3))
lines = ax.plot(range(10), label='A simple plot')
ax.legend(bbox_to_anchor=(0.7, 0.5), loc='center left')
fig.tight_layout()
plt.show()
(fig, ax) = plt.subplots(figsize=(4, 3))
lines = ax.plot(range(10), label='B simple plot')
leg = ax.legend(bbox_to_anchor=(0.7, 0.5), loc='center left')
leg.set_in_layout(False)
fig.tight_layout()
plt.show()
from mpl_toolkits.axes_grid1 import Grid
plt.close('all')
fig = plt.figure()
grid = Grid(fig, rect=111, nrows_ncols=(2, 2), axes_pad=0.25, label_mode='L')
for ax in grid:
    example_plot(ax)
ax.title.set_visible(False)
plt.tight_layout()
plt.close('all')
arr = np.arange(100).reshape((10, 10))
fig = plt.figure(figsize=(4, 4))
im = plt.imshow(arr, interpolation='none')
plt.colorbar(im)
plt.tight_layout()
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.close('all')
arr = np.arange(100).reshape((10, 10))
fig = plt.figure(figsize=(4, 4))
im = plt.imshow(arr, interpolation='none')
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes('right', '5%', pad='3%')
plt.colorbar(im, cax=cax)
plt.tight_layout()