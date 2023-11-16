"""
.. redirect-from:: /tutorials/intermediate/imshow_extent

.. _imshow_extent:

*origin* and *extent* in `~.Axes.imshow`
========================================

:meth:`~.Axes.imshow` allows you to render an image (either a 2D array which
will be color-mapped (based on *norm* and *cmap*) or a 3D RGB(A) array which
will be used as-is) to a rectangular region in data space.  The orientation of
the image in the final rendering is controlled by the *origin* and *extent*
keyword arguments (and attributes on the resulting `.AxesImage` instance) and
the data limits of the axes.

The *extent* keyword arguments controls the bounding box in data coordinates
that the image will fill specified as ``(left, right, bottom, top)`` in **data
coordinates**, the *origin* keyword argument controls how the image fills that
bounding box, and the orientation in the final rendered image is also affected
by the axes limits.

.. hint:: Most of the code below is used for adding labels and informative
   text to the plots. The described effects of *origin* and *extent* can be
   seen in the plots without the need to follow all code details.

   For a quick understanding, you may want to skip the code details below and
   directly continue with the discussion of the results.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def index_to_coordinate(index, extent, origin):
    if False:
        while True:
            i = 10
    'Return the pixel center of an index.'
    (left, right, bottom, top) = extent
    hshift = 0.5 * np.sign(right - left)
    (left, right) = (left + hshift, right - hshift)
    vshift = 0.5 * np.sign(top - bottom)
    (bottom, top) = (bottom + vshift, top - vshift)
    if origin == 'upper':
        (bottom, top) = (top, bottom)
    return {'[0, 0]': (left, bottom), "[M', 0]": (left, top), "[0, N']": (right, bottom), "[M', N']": (right, top)}[index]

def get_index_label_pos(index, extent, origin, inverted_xindex):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the desired position and horizontal alignment of an index label.\n    '
    if extent is None:
        extent = lookup_extent(origin)
    (left, right, bottom, top) = extent
    (x, y) = index_to_coordinate(index, extent, origin)
    is_x0 = index[-2:] == '0]'
    halign = 'left' if is_x0 ^ inverted_xindex else 'right'
    hshift = 0.5 * np.sign(left - right)
    x += hshift * (1 if is_x0 else -1)
    return (x, y, halign)

def get_color(index, data, cmap):
    if False:
        while True:
            i = 10
    'Return the data color of an index.'
    val = {'[0, 0]': data[0, 0], "[0, N']": data[0, -1], "[M', 0]": data[-1, 0], "[M', N']": data[-1, -1]}[index]
    return cmap(val / data.max())

def lookup_extent(origin):
    if False:
        while True:
            i = 10
    'Return extent for label positioning when not given explicitly.'
    if origin == 'lower':
        return (-0.5, 6.5, -0.5, 5.5)
    else:
        return (-0.5, 6.5, 5.5, -0.5)

def set_extent_None_text(ax):
    if False:
        for i in range(10):
            print('nop')
    ax.text(3, 2.5, 'equals\nextent=None', size='large', ha='center', va='center', color='w')

def plot_imshow_with_labels(ax, data, extent, origin, xlim, ylim):
    if False:
        while True:
            i = 10
    'Actually run ``imshow()`` and add extent and index labels.'
    im = ax.imshow(data, origin=origin, extent=extent)
    (left, right, bottom, top) = im.get_extent()
    if xlim is None or top > bottom:
        (upper_string, lower_string) = ('top', 'bottom')
    else:
        (upper_string, lower_string) = ('bottom', 'top')
    if ylim is None or left < right:
        (port_string, starboard_string) = ('left', 'right')
        inverted_xindex = False
    else:
        (port_string, starboard_string) = ('right', 'left')
        inverted_xindex = True
    bbox_kwargs = {'fc': 'w', 'alpha': 0.75, 'boxstyle': 'round4'}
    ann_kwargs = {'xycoords': 'axes fraction', 'textcoords': 'offset points', 'bbox': bbox_kwargs}
    ax.annotate(upper_string, xy=(0.5, 1), xytext=(0, -1), ha='center', va='top', **ann_kwargs)
    ax.annotate(lower_string, xy=(0.5, 0), xytext=(0, 1), ha='center', va='bottom', **ann_kwargs)
    ax.annotate(port_string, xy=(0, 0.5), xytext=(1, 0), ha='left', va='center', rotation=90, **ann_kwargs)
    ax.annotate(starboard_string, xy=(1, 0.5), xytext=(-1, 0), ha='right', va='center', rotation=-90, **ann_kwargs)
    ax.set_title(f'origin: {origin}')
    for index in ['[0, 0]', "[0, N']", "[M', 0]", "[M', N']"]:
        (tx, ty, halign) = get_index_label_pos(index, extent, origin, inverted_xindex)
        facecolor = get_color(index, data, im.get_cmap())
        ax.text(tx, ty, index, color='white', ha=halign, va='center', bbox={'boxstyle': 'square', 'facecolor': facecolor})
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

def generate_imshow_demo_grid(extents, xlim=None, ylim=None):
    if False:
        for i in range(10):
            print('nop')
    N = len(extents)
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6, N * 11.25 / 5)
    gs = GridSpec(N, 5, figure=fig)
    columns = {'label': [fig.add_subplot(gs[j, 0]) for j in range(N)], 'upper': [fig.add_subplot(gs[j, 1:3]) for j in range(N)], 'lower': [fig.add_subplot(gs[j, 3:5]) for j in range(N)]}
    (x, y) = np.ogrid[0:6, 0:7]
    data = x + y
    for origin in ['upper', 'lower']:
        for (ax, extent) in zip(columns[origin], extents):
            plot_imshow_with_labels(ax, data, extent, origin, xlim, ylim)
    columns['label'][0].set_title('extent=')
    for (ax, extent) in zip(columns['label'], extents):
        if extent is None:
            text = 'None'
        else:
            (left, right, bottom, top) = extent
            text = f'left: {left:0.1f}\nright: {right:0.1f}\nbottom: {bottom:0.1f}\ntop: {top:0.1f}\n'
        ax.text(1.0, 0.5, text, transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')
    return columns
generate_imshow_demo_grid(extents=[None])
extents = [(-0.5, 6.5, -0.5, 5.5), (-0.5, 6.5, 5.5, -0.5), (6.5, -0.5, -0.5, 5.5), (6.5, -0.5, 5.5, -0.5)]
columns = generate_imshow_demo_grid(extents)
set_extent_None_text(columns['upper'][1])
set_extent_None_text(columns['lower'][0])
generate_imshow_demo_grid(extents=[None] + extents, xlim=(-2, 8), ylim=(-1, 6))
plt.show()