"""
.. redirect-from:: /tutorials/intermediate/legend_guide

.. _legend_guide:

============
Legend guide
============

.. currentmodule:: matplotlib.pyplot

This legend guide extends the `~.Axes.legend` docstring -
please read it before proceeding with this guide.

This guide makes use of some common terms, which are documented here for
clarity:

.. glossary::

    legend entry
        A legend is made up of one or more legend entries. An entry is made up
        of exactly one key and one label.

    legend key
        The colored/patterned marker to the left of each legend label.

    legend label
        The text which describes the handle represented by the key.

    legend handle
        The original object which is used to generate an appropriate entry in
        the legend.


Controlling the legend entries
==============================

Calling :func:`legend` with no arguments automatically fetches the legend
handles and their associated labels. This functionality is equivalent to::

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

The :meth:`~matplotlib.axes.Axes.get_legend_handles_labels` function returns
a list of handles/artists which exist on the Axes which can be used to
generate entries for the resulting legend - it is worth noting however that
not all artists can be added to a legend, at which point a "proxy" will have
to be created (see :ref:`proxy_legend_handles` for further details).

.. note::
    Artists with an empty string as label or with a label starting with an
    underscore, "_", will be ignored.

For full control of what is being added to the legend, it is common to pass
the appropriate handles directly to :func:`legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend(handles=[line_up, line_down])

In the rare case where the labels cannot directly be set on the handles, they
can also be directly passed to :func:`legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend([line_up, line_down], ['Line Up', 'Line Down'])


.. _proxy_legend_handles:

Creating artists specifically for adding to the legend (aka. Proxy artists)
===========================================================================

Not all handles can be turned into legend entries automatically,
so it is often necessary to create an artist which *can*. Legend handles
don't have to exist on the Figure or Axes in order to be used.

Suppose we wanted to create a legend which has an entry for some data which
is represented by a red color:
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
(fig, ax) = plt.subplots()
red_patch = mpatches.Patch(color='red', label='The red data')
ax.legend(handles=[red_patch])
plt.show()
import matplotlib.lines as mlines
(fig, ax) = plt.subplots()
blue_line = mlines.Line2D([], [], color='blue', marker='*', markersize=15, label='Blue stars')
ax.legend(handles=[blue_line])
plt.show()
(fig, ax_dict) = plt.subplot_mosaic([['top', 'top'], ['bottom', 'BLANK']], empty_sentinel='BLANK')
ax_dict['top'].plot([1, 2, 3], label='test1')
ax_dict['top'].plot([3, 2, 1], label='test2')
ax_dict['top'].legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc='lower left', ncols=2, mode='expand', borderaxespad=0.0)
ax_dict['bottom'].plot([1, 2, 3], label='test1')
ax_dict['bottom'].plot([3, 2, 1], label='test2')
ax_dict['bottom'].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
(fig, axs) = plt.subplot_mosaic([['left', 'right']], layout='constrained')
axs['left'].plot([1, 2, 3], label='test1')
axs['left'].plot([3, 2, 1], label='test2')
axs['right'].plot([1, 2, 3], 'C2', label='test3')
axs['right'].plot([3, 2, 1], 'C3', label='test4')
fig.legend(loc='outside upper right')
ucl = ['upper', 'center', 'lower']
lcr = ['left', 'center', 'right']
(fig, ax) = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')
ax.plot([1, 2], [1, 2], label='TEST')
for loc in ['outside upper left', 'outside upper center', 'outside upper right', 'outside lower left', 'outside lower center', 'outside lower right']:
    fig.legend(loc=loc, title=loc)
(fig, ax) = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')
ax.plot([1, 2], [1, 2], label='test')
for loc in ['outside left upper', 'outside right upper', 'outside left lower', 'outside right lower']:
    fig.legend(loc=loc, title=loc)
(fig, ax) = plt.subplots()
(line1,) = ax.plot([1, 2, 3], label='Line 1', linestyle='--')
(line2,) = ax.plot([3, 2, 1], label='Line 2', linewidth=4)
first_legend = ax.legend(handles=[line1], loc='upper right')
ax.add_artist(first_legend)
ax.legend(handles=[line2], loc='lower right')
plt.show()
from matplotlib.legend_handler import HandlerLine2D
(fig, ax) = plt.subplots()
(line1,) = ax.plot([3, 2, 1], marker='o', label='Line 1')
(line2,) = ax.plot([1, 2, 3], marker='o', label='Line 2')
ax.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
from numpy.random import randn
z = randn(10)
(fig, ax) = plt.subplots()
(red_dot,) = ax.plot(z, 'ro', markersize=15)
(white_cross,) = ax.plot(z[:5], 'w+', markeredgewidth=3, markersize=15)
ax.legend([red_dot, (red_dot, white_cross)], ['Attr A', 'Attr A+B'])
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
(fig, ax) = plt.subplots()
(p1,) = ax.plot([1, 2.5, 3], 'r-d')
(p2,) = ax.plot([3, 2, 1], 'k-o')
l = ax.legend([(p1, p2)], ['Two keys'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
import matplotlib.patches as mpatches

class AnyObject:
    pass

class AnyObjectHandler:

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        if False:
            print('Hello World!')
        (x0, y0) = (handlebox.xdescent, handlebox.ydescent)
        (width, height) = (handlebox.width, handlebox.height)
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red', edgecolor='black', hatch='xx', lw=3, transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch
(fig, ax) = plt.subplots()
ax.legend([AnyObject()], ['My first handler'], handler_map={AnyObject: AnyObjectHandler()})
from matplotlib.legend_handler import HandlerPatch

class HandlerEllipse(HandlerPatch):

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        if False:
            for i in range(10):
                print('nop')
        center = (0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent)
        p = mpatches.Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]
c = mpatches.Circle((0.5, 0.5), 0.25, facecolor='green', edgecolor='red', linewidth=3)
(fig, ax) = plt.subplots()
ax.add_patch(c)
ax.legend([c], ['An ellipse, not a rectangle'], handler_map={mpatches.Circle: HandlerEllipse()})