"""
=========================
Euler number
=========================

This example shows an illustration of the computation of the Euler number [1]_
in 2D and 3D objects.

For 2D objects, the Euler number is the number of objects minus the number of
holes. Notice that if a neighborhood of 8 connected pixels (2-connectivity)
is considered for objects, then this amounts to considering a neighborhood
of 4 connected pixels (1-connectivity) for the complementary set (holes,
background) , and conversely. It is also possible to compute the number of
objects using :func:`skimage.measure.label`, and to deduce the number of holes
from the difference between the two numbers.

For 3D objects, the Euler number is obtained as the number of objects plus the
number of holes, minus the number of tunnels, or loops. If one uses
3-connectivity for an object (considering the 26 surrounding voxels as its
neighborhood), this corresponds to using 1-connectivity for the complementary
set (holes, background), that is considering only 6 neighbors for a given
voxel. The voxels are represented here with blue transparent surfaces.
Inner porosities are represented in red.

.. [1] https://en.wikipedia.org/wiki/Euler_characteristic
"""
from skimage.measure import euler_number, label
import matplotlib.pyplot as plt
import numpy as np
SAMPLE = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
SAMPLE = np.pad(SAMPLE, 1, mode='constant')
(fig, ax) = plt.subplots()
ax.imshow(SAMPLE, cmap=plt.cm.gray)
ax.axis('off')
e4 = euler_number(SAMPLE, connectivity=1)
object_nb_4 = label(SAMPLE, connectivity=1).max()
holes_nb_4 = object_nb_4 - e4
e8 = euler_number(SAMPLE, connectivity=2)
object_nb_8 = label(SAMPLE, connectivity=2).max()
holes_nb_8 = object_nb_8 - e8
ax.set_title(f'Euler number for N4: {e4} ({object_nb_4} objects, {holes_nb_4} holes), \n for N8: {e8} ({object_nb_8} objects, {holes_nb_8} holes)')
plt.show()

def make_ax(grid=False):
    if False:
        print('Hello World!')
    ax = plt.figure().add_subplot(projection='3d')
    ax.grid(grid)
    ax.set_axis_off()
    return ax

def explode(data):
    if False:
        print('Hello World!')
    'visualization to separate voxels\n\n    Data voxels are separated by 0-valued ones so that they appear\n    separated in the matplotlib figure.\n    '
    size = np.array(data.shape) * 2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

def expand_coordinates(indices):
    if False:
        while True:
            i = 10
    '\n    This collapses together pairs of indices, so that\n    the gaps in the volume array will have a zero width.\n    '
    (x, y, z) = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return (x, y, z)

def display_voxels(volume):
    if False:
        return 10
    '\n    volume: (N,M,P) array\n            Represents a binary set of pixels: objects are marked with 1,\n            complementary (porosities) with 0.\n\n    The voxels are actually represented with blue transparent surfaces.\n    Inner porosities are represented in red.\n    '
    red = '#ff0000ff'
    blue = '#1f77b410'
    filled = explode(np.ones(volume.shape))
    fcolors = explode(np.where(volume, blue, red))
    (x, y, z) = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax = make_ax()
    ax.voxels(x, y, z, filled, facecolors=fcolors)
    e26 = euler_number(volume, connectivity=3)
    e6 = euler_number(volume, connectivity=1)
    plt.title(f'Euler number for N26: {e26}, for N6: {e6}')
    plt.show()
n = 7
cube = np.ones((n, n, n), dtype=bool)
c = int(n / 2)
cube[c, :, c] = False
cube[int(3 * n / 4), c - 1, c - 1] = False
cube[int(3 * n / 4), c, c] = False
cube[:, c, int(3 * n / 4)] = False
display_voxels(cube)