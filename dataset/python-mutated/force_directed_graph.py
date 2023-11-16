"""
Plot clusters of data points and a graph of connections
"""
from vispy import app, scene, color
import numpy as np
npts = 400
nedges = 900
ngroups = 7
np.random.seed(127396)
pos = np.empty((npts, 2), dtype='float32')
colors = np.empty((npts, 3), dtype='float32')
edges = np.empty((nedges, 2), dtype='uint32')
types = np.empty(npts, dtype=int)
pos[:] = np.random.normal(size=pos.shape, scale=4.0)
grpsize = npts // ngroups
ptr = 0
typ = 0
while ptr < npts:
    size = np.random.random() * grpsize + grpsize // 2
    types[int(ptr):int(ptr + size)] = typ
    typ += 1
    ptr = ptr + size
conn = []
connset = set()
while len(conn) < nedges:
    (i, j) = np.random.randint(npts, size=2)
    if i == j:
        continue
    p = 0.7 if types[i] == types[j] else 0.01
    if np.random.random() < p:
        if (i, j) in connset:
            continue
        connset.add((i, j))
        connset.add((j, i))
        conn.append([i, j])
edges[:] = conn
cmap = color.get_colormap('cubehelix')
typ_colors = np.array([cmap.map(x)[0, :3] for x in np.linspace(0.2, 0.8, typ)])
colors[:] = typ_colors[types]
colors *= 1.1 ** np.random.normal(size=colors.shape)
colors = np.clip(colors, 0, 1)
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'panzoom'
view.camera.aspect = 1
lines = scene.Line(pos=pos, connect=edges, antialias=False, method='gl', color=(1, 1, 1, 0.2), parent=view.scene)
markers = scene.Markers(pos=pos, face_color=colors, symbol='o', parent=view.scene)
view.camera.set_range()
i = 1

def update(ev):
    if False:
        while True:
            i = 10
    global pos, edges, lines, markers, view, force, dist, i
    dx = np.empty((npts, npts, 2), dtype='float32')
    dx[:] = pos[:, np.newaxis, :]
    dx -= pos[np.newaxis, :, :]
    dist = (dx ** 2).sum(axis=2) ** 0.5
    dist[dist == 0] = 1.0
    ndx = dx / dist[..., np.newaxis]
    force = np.zeros((npts, npts, 2), dtype='float32')
    force -= 0.1 * ndx / dist[..., np.newaxis] ** 2
    s = 0.1
    mask = np.zeros((npts, npts, 1), dtype='float32')
    mask[edges[:, 0], edges[:, 1]] = s
    mask[edges[:, 1], edges[:, 0]] = s
    force += dx * dist[..., np.newaxis] * mask
    force[np.arange(npts), np.arange(npts)] = 0
    force = force.sum(axis=0)
    pos += np.clip(force, -3, 3) * 0.09
    lines.set_data(pos=pos)
    markers.set_data(pos=pos, face_color=colors)
    i += 1
timer = app.Timer(interval=0, connect=update, start=True)
if __name__ == '__main__':
    app.run()