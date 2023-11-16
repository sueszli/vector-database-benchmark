"""
================
Trigradient Demo
================

Demonstrates computation of gradient with
`matplotlib.tri.CubicTriInterpolator`.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import CubicTriInterpolator, Triangulation, UniformTriRefiner

def dipole_potential(x, y):
    if False:
        return 10
    'The electric dipole potential V, at position *x*, *y*.'
    r_sq = x ** 2 + y ** 2
    theta = np.arctan2(y, x)
    z = np.cos(theta) / r_sq
    return (np.max(z) - z) / (np.max(z) - np.min(z))
n_angles = 30
n_radii = 10
min_radius = 0.2
radii = np.linspace(min_radius, 0.95, n_radii)
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += np.pi / n_angles
x = (radii * np.cos(angles)).flatten()
y = (radii * np.sin(angles)).flatten()
V = dipole_potential(x, y)
triang = Triangulation(x, y)
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1)) < min_radius)
refiner = UniformTriRefiner(triang)
(tri_refi, z_test_refi) = refiner.refine_field(V, subdiv=3)
tci = CubicTriInterpolator(triang, -V)
(Ex, Ey) = tci.gradient(triang.x, triang.y)
E_norm = np.sqrt(Ex ** 2 + Ey ** 2)
(fig, ax) = plt.subplots()
ax.set_aspect('equal')
ax.use_sticky_edges = False
ax.margins(0.07)
ax.triplot(triang, color='0.8')
levels = np.arange(0.0, 1.0, 0.01)
ax.tricontour(tri_refi, z_test_refi, levels=levels, cmap='hot', linewidths=[2.0, 1.0, 1.0, 1.0])
ax.quiver(triang.x, triang.y, Ex / E_norm, Ey / E_norm, units='xy', scale=10.0, zorder=3, color='blue', width=0.007, headwidth=3.0, headlength=4.0)
ax.set_title('Gradient plot: an electrical dipole')
plt.show()