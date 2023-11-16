"""
==========================
Tricontour Smooth Delaunay
==========================

Demonstrates high-resolution tricontouring of a random set of points;
a `matplotlib.tri.TriAnalyzer` is used to improve the plot quality.

The initial data points and triangular grid for this demo are:

- a set of random points is instantiated, inside [-1, 1] x [-1, 1] square
- A Delaunay triangulation of these points is then computed, of which a
  random subset of triangles is masked out by the user (based on
  *init_mask_frac* parameter). This simulates invalidated data.

The proposed generic procedure to obtain a high resolution contouring of such
a data set is the following:

1. Compute an extended mask with a `matplotlib.tri.TriAnalyzer`, which will
   exclude badly shaped (flat) triangles from the border of the
   triangulation. Apply the mask to the triangulation (using set_mask).
2. Refine and interpolate the data using a `matplotlib.tri.UniformTriRefiner`.
3. Plot the refined data with `~.axes.Axes.tricontour`.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import TriAnalyzer, Triangulation, UniformTriRefiner

def experiment_res(x, y):
    if False:
        i = 10
        return i + 15
    'An analytic function representing experiment results.'
    x = 2 * x
    r1 = np.sqrt((0.5 - x) ** 2 + (0.5 - y) ** 2)
    theta1 = np.arctan2(0.5 - x, 0.5 - y)
    r2 = np.sqrt((-x - 0.2) ** 2 + (-y - 0.2) ** 2)
    theta2 = np.arctan2(-x - 0.2, -y - 0.2)
    z = 4 * (np.exp((r1 / 10) ** 2) - 1) * 30 * np.cos(3 * theta1) + (np.exp((r2 / 10) ** 2) - 1) * 30 * np.cos(5 * theta2) + 2 * (x ** 2 + y ** 2)
    return (np.max(z) - z) / (np.max(z) - np.min(z))
n_test = 200
subdiv = 3
init_mask_frac = 0.0
min_circle_ratio = 0.01
random_gen = np.random.RandomState(seed=19680801)
x_test = random_gen.uniform(-1.0, 1.0, size=n_test)
y_test = random_gen.uniform(-1.0, 1.0, size=n_test)
z_test = experiment_res(x_test, y_test)
tri = Triangulation(x_test, y_test)
ntri = tri.triangles.shape[0]
mask_init = np.zeros(ntri, dtype=bool)
masked_tri = random_gen.randint(0, ntri, int(ntri * init_mask_frac))
mask_init[masked_tri] = True
tri.set_mask(mask_init)
mask = TriAnalyzer(tri).get_flat_tri_mask(min_circle_ratio)
tri.set_mask(mask)
refiner = UniformTriRefiner(tri)
(tri_refi, z_test_refi) = refiner.refine_field(z_test, subdiv=subdiv)
z_expected = experiment_res(tri_refi.x, tri_refi.y)
flat_tri = Triangulation(x_test, y_test)
flat_tri.set_mask(~mask)
plot_tri = True
plot_masked_tri = True
plot_refi_tri = False
plot_expected = False
levels = np.arange(0.0, 1.0, 0.025)
(fig, ax) = plt.subplots()
ax.set_aspect('equal')
ax.set_title('Filtering a Delaunay mesh\n(application to high-resolution tricontouring)')
ax.tricontour(tri_refi, z_test_refi, levels=levels, cmap='Blues', linewidths=[2.0, 0.5, 1.0, 0.5])
if plot_expected:
    ax.tricontour(tri_refi, z_expected, levels=levels, cmap='Blues', linestyles='--')
if plot_refi_tri:
    ax.triplot(tri_refi, color='0.97')
if plot_tri:
    ax.triplot(tri, color='0.7')
if plot_masked_tri:
    ax.triplot(flat_tri, color='red')
plt.show()