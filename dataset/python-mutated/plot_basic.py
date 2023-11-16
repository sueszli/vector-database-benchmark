"""
================
Basic matplotlib
================

A basic example of 3D Graph visualization using `mpl_toolkits.mplot_3d`.

"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
G = nx.cycle_graph(20)
pos = nx.spring_layout(G, dim=3, seed=779)
node_xyz = np.array([pos[v] for v in sorted(G)])
edge_xyz = np.array([(pos[u], pos[v]) for (u, v) in G.edges()])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*node_xyz.T, s=100, ec='w')
for vizedge in edge_xyz:
    ax.plot(*vizedge.T, color='tab:gray')

def _format_axes(ax):
    if False:
        for i in range(10):
            print('nop')
    'Visualization options for the 3D axes.'
    ax.grid(False)
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
_format_axes(ax)
fig.tight_layout()
plt.show()