"""
=======================================================
3D voxel / volumetric plot with cylindrical coordinates
=======================================================

Demonstrates using the *x*, *y*, *z* parameters of `.Axes3D.voxels`.
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors

def midpoints(x):
    if False:
        print('Hello World!')
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x
(r, theta, z) = np.mgrid[0:1:11j, 0:np.pi * 2:25j, -0.5:0.5:11j]
x = r * np.cos(theta)
y = r * np.sin(theta)
(rc, thetac, zc) = (midpoints(r), midpoints(theta), midpoints(z))
sphere = (rc - 0.7) ** 2 + (zc + 0.2 * np.cos(thetac * 2)) ** 2 < 0.2 ** 2
hsv = np.zeros(sphere.shape + (3,))
hsv[..., 0] = thetac / (np.pi * 2)
hsv[..., 1] = rc
hsv[..., 2] = zc + 0.5
colors = matplotlib.colors.hsv_to_rgb(hsv)
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, sphere, facecolors=colors, edgecolors=np.clip(2 * colors - 0.5, 0, 1), linewidth=0.5)
plt.show()