"""
=======================
Colormap normalizations
=======================

Demonstration of using norm to map colormaps onto data in non-linear ways.

.. redirect-from:: /gallery/userdemo/colormap_normalizations
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
N = 100
(X, Y) = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = np.exp(-X ** 2 - Y ** 2)
Z2 = np.exp(-(X * 10) ** 2 - (Y * 10) ** 2)
Z = Z1 + 50 * Z2
(fig, ax) = plt.subplots(2, 1)
pcm = ax[0].pcolor(X, Y, Z, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='max')
pcm = ax[1].pcolor(X, Y, Z, cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='max')
(X, Y) = np.mgrid[0:3:complex(0, N), 0:2:complex(0, N)]
Z1 = (1 + np.sin(Y * 10.0)) * X ** 2
(fig, ax) = plt.subplots(2, 1)
pcm = ax[0].pcolormesh(X, Y, Z1, norm=colors.PowerNorm(gamma=1.0 / 2.0), cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='max')
pcm = ax[1].pcolormesh(X, Y, Z1, cmap='PuBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='max')
(X, Y) = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z = 5 * np.exp(-X ** 2 - Y ** 2)
(fig, ax) = plt.subplots(2, 1)
pcm = ax[0].pcolormesh(X, Y, Z, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=-1.0, vmax=1.0, base=10), cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='both')
pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z), shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='both')
(X, Y) = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
Z1 = np.exp(-X ** 2 - Y ** 2)
Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
Z = (Z1 - Z2) * 2

class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        if False:
            while True:
                i = 10
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if False:
            print('Hello World!')
        (x, y) = ([self.vmin, self.midpoint, self.vmax], [0, 0.5, 1])
        return np.ma.masked_array(np.interp(value, x, y))
(fig, ax) = plt.subplots(2, 1)
pcm = ax[0].pcolormesh(X, Y, Z, norm=MidpointNormalize(midpoint=0.0), cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='both')
pcm = ax[1].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z), shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='both')
(fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
ax = ax.flatten()
bounds = np.linspace(-1, 1, 10)
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[0].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical')
bounds = np.array([-0.25, -0.125, 0, 0.5, 1])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
pcm = ax[1].pcolormesh(X, Y, Z, norm=norm, cmap='RdBu_r', shading='nearest')
fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')
pcm = ax[2].pcolormesh(X, Y, Z, cmap='RdBu_r', vmin=-np.max(Z1), shading='nearest')
fig.colorbar(pcm, ax=ax[2], extend='both', orientation='vertical')
plt.show()