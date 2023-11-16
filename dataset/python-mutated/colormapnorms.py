"""

.. redirect-from:: /tutorials/colors/colormapnorms

.. _colormapnorms:

Colormap Normalization
======================

Objects that use colormaps by default linearly map the colors in the
colormap from data values *vmin* to *vmax*.  For example::

    pcm = ax.pcolormesh(x, y, Z, vmin=-1., vmax=1., cmap='RdBu_r')

will map the data in *Z* linearly from -1 to +1, so *Z=0* will
give a color at the center of the colormap *RdBu_r* (white in this
case).

Matplotlib does this mapping in two steps, with a normalization from
the input data to [0, 1] occurring first, and then mapping onto the
indices in the colormap.  Normalizations are classes defined in the
:func:`matplotlib.colors` module.  The default, linear normalization
is :func:`matplotlib.colors.Normalize`.

Artists that map data to color pass the arguments *vmin* and *vmax* to
construct a :func:`matplotlib.colors.Normalize` instance, then call it:

.. code-block:: pycon

   >>> import matplotlib as mpl
   >>> norm = mpl.colors.Normalize(vmin=-1, vmax=1)
   >>> norm(0)
   0.5

However, there are sometimes cases where it is useful to map data to
colormaps in a non-linear fashion.

Logarithmic
-----------

One of the most common transformations is to plot data by taking its logarithm
(to the base-10).  This transformation is useful to display changes across
disparate scales.  Using `.colors.LogNorm` normalizes the data via
:math:`log_{10}`.  In the example below, there are two bumps, one much smaller
than the other. Using `.colors.LogNorm`, the shape and location of each bump
can clearly be seen:

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
N = 100
(X, Y) = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = np.exp(-X ** 2 - Y ** 2)
Z2 = np.exp(-(X * 10) ** 2 - (Y * 10) ** 2)
Z = Z1 + 50 * Z2
(fig, ax) = plt.subplots(2, 1)
pcm = ax[0].pcolor(X, Y, Z, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='max')
pcm = ax[1].pcolor(X, Y, Z, cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[1], extend='max')
plt.show()
delta = 0.1
x = np.arange(-3.0, 4.001, delta)
y = np.arange(-4.0, 3.001, delta)
(X, Y) = np.meshgrid(x, y)
Z1 = np.exp(-X ** 2 - Y ** 2)
Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
Z = (0.9 * Z1 - 0.5 * Z2) * 2
cmap = cm.coolwarm
(fig, (ax1, ax2)) = plt.subplots(ncols=2)
pc = ax1.pcolormesh(Z, cmap=cmap)
fig.colorbar(pc, ax=ax1)
ax1.set_title('Normalize()')
pc = ax2.pcolormesh(Z, norm=colors.CenteredNorm(), cmap=cmap)
fig.colorbar(pc, ax=ax2)
ax2.set_title('CenteredNorm()')
plt.show()
N = 100
(X, Y) = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = np.exp(-X ** 2 - Y ** 2)
Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
Z = (Z1 - Z2) * 2
(fig, ax) = plt.subplots(2, 1)
pcm = ax[0].pcolormesh(X, Y, Z, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-1.0, vmax=1.0, base=10), cmap='RdBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='both')
pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z), shading='auto')
fig.colorbar(pcm, ax=ax[1], extend='both')
plt.show()
N = 100
(X, Y) = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.0)) * X ** 2
(fig, ax) = plt.subplots(2, 1, layout='constrained')
pcm = ax[0].pcolormesh(X, Y, Z1, norm=colors.PowerNorm(gamma=0.5), cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[0], extend='max')
ax[0].set_title('PowerNorm()')
pcm = ax[1].pcolormesh(X, Y, Z1, cmap='PuBu_r', shading='auto')
fig.colorbar(pcm, ax=ax[1], extend='max')
ax[1].set_title('Normalize()')
plt.show()
N = 100
(X, Y) = np.meshgrid(np.linspace(-3, 3, N), np.linspace(-2, 2, N))
Z1 = np.exp(-X ** 2 - Y ** 2)
Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
Z = ((Z1 - Z2) * 2)[:-1, :-1]
(fig, ax) = plt.subplots(2, 2, figsize=(8, 6), layout='constrained')
ax = ax.flatten()
pcm = ax[0].pcolormesh(X, Y, Z, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[0], orientation='vertical')
ax[0].set_title('Default norm')
bounds = np.linspace(-1.5, 1.5, 7)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[1].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')
ax[1].set_title('BoundaryNorm: 7 boundaries')
bounds = np.array([-0.2, -0.1, 0, 0.5, 1])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[2].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[2], extend='both', orientation='vertical')
ax[2].set_title('BoundaryNorm: nonuniform')
bounds = np.linspace(-1.5, 1.5, 7)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')
pcm = ax[3].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r')
fig.colorbar(pcm, ax=ax[3], orientation='vertical')
ax[3].set_title('BoundaryNorm: extend="both"')
plt.show()
dem = cbook.get_sample_data('topobathy.npz')
topo = dem['topo']
longitude = dem['longitude']
latitude = dem['latitude']
(fig, ax) = plt.subplots()
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)
divnorm = colors.TwoSlopeNorm(vmin=-500.0, vcenter=0, vmax=4000)
pcm = ax.pcolormesh(longitude, latitude, topo, rasterized=True, norm=divnorm, cmap=terrain_map, shading='auto')
ax.set_aspect(1 / np.cos(np.deg2rad(49)))
ax.set_title('TwoSlopeNorm(x)')
cb = fig.colorbar(pcm, shrink=0.6)
cb.set_ticks([-500, 0, 1000, 2000, 3000, 4000])
plt.show()

def _forward(x):
    if False:
        while True:
            i = 10
    return np.sqrt(x)

def _inverse(x):
    if False:
        for i in range(10):
            print('nop')
    return x ** 2
N = 100
(X, Y) = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.0)) * X ** 2
(fig, ax) = plt.subplots()
norm = colors.FuncNorm((_forward, _inverse), vmin=0, vmax=20)
pcm = ax.pcolormesh(X, Y, Z1, norm=norm, cmap='PuBu_r', shading='auto')
ax.set_title('FuncNorm(x)')
fig.colorbar(pcm, shrink=0.6)
plt.show()

class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        if False:
            return 10
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if False:
            while True:
                i = 10
        (x, y) = ([self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.0])
        return np.ma.masked_array(np.interp(value, x, y, left=-np.inf, right=np.inf))

    def inverse(self, value):
        if False:
            for i in range(10):
                print('nop')
        (y, x) = ([self.vmin, self.vcenter, self.vmax], [0, 0.5, 1])
        return np.interp(value, x, y, left=-np.inf, right=np.inf)
(fig, ax) = plt.subplots()
midnorm = MidpointNormalize(vmin=-500.0, vcenter=0, vmax=4000)
pcm = ax.pcolormesh(longitude, latitude, topo, rasterized=True, norm=midnorm, cmap=terrain_map, shading='auto')
ax.set_aspect(1 / np.cos(np.deg2rad(49)))
ax.set_title('Custom norm')
cb = fig.colorbar(pcm, shrink=0.6, extend='both')
cb.set_ticks([-500, 0, 1000, 2000, 3000, 4000])
plt.show()