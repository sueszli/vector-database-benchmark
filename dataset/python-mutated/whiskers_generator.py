import numpy as np
from copy import deepcopy

def sample_spherical(npoints, ndim=3):
    if False:
        while True:
            i = 10
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
phi = np.linspace(0, np.pi, num=6)
print(phi)
theta = np.linspace(0, 2 * np.pi, num=9)
r = 1
xc = r * np.outer(np.sin(theta), np.cos(phi))
yc = r * np.outer(np.sin(theta), np.sin(phi))
zc = r * np.outer(np.cos(theta), np.ones_like(phi))
eps = 1e-10
xcc = xc.reshape(-1)
ycc = yc.reshape(-1)
zcc = zc.reshape(-1)
samples = list(zip(xcc, ycc, zcc))
unique_samples = []
removed = []
print(len(samples))
removed_counter = 0
other_samples = deepcopy(samples)
for (i, s) in enumerate(samples):
    add_flag = True
    del other_samples[0]
    for (j, ss) in enumerate(other_samples):
        dist = np.linalg.norm(np.array(s) - np.array(ss))
        if dist < eps:
            removed_counter += 1
            add_flag = False
            break
    if add_flag:
        unique_samples.append(s)
    else:
        removed.append(s)
print(f'removed counter = {removed_counter}')
print(len(unique_samples))
print(unique_samples[0])
unique_samples = np.array(unique_samples)
removed = np.array(removed)
x = unique_samples[:, 0].reshape(-1, 1)
y = unique_samples[:, 1].reshape(-1, 1)
z = unique_samples[:, 2].reshape(-1, 1)
with open('whiskers.npy', 'wb') as f:
    np.save(f, np.array(list(zip(x.reshape(-1), y.reshape(-1), z.reshape(-1)))))
ax = plt.axes(projection='3d', aspect='auto')
ax.plot_wireframe(xc, yc, zc, color='k', rstride=1, cstride=1)
ax.scatter(x, y, z, s=100, c='b', zorder=10)
plt.show()
with open('whiskers.npy', 'rb') as f:
    p = np.load(f)
    print(len(p))
    print(p[0])