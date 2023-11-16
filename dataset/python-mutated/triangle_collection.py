import numpy as np
from vispy import app, gloo
from vispy.geometry import Triangulation
from vispy.visuals.collections import PathCollection, TriangleCollection
canvas = app.Canvas(size=(800, 800), show=True, keys='interactive')
gloo.set_viewport(0, 0, canvas.size[0], canvas.size[1])
gloo.set_state('translucent', depth_test=False)

def triangulate(P):
    if False:
        print('Hello World!')
    n = len(P)
    S = np.repeat(np.arange(n + 1), 2)[1:-1]
    S[-2:] = (n - 1, 0)
    S = S.reshape(len(S) // 2, 2)
    T = Triangulation(P[:, :2], S)
    T.triangulate()
    points = T.pts
    triangles = T.tris.ravel()
    P = np.zeros((len(points), 3), dtype=np.float32)
    P[:, :2] = points
    return (P, triangles)

def star(inner=0.5, outer=1.0, n=5):
    if False:
        for i in range(10):
            print('nop')
    R = np.array([inner, outer] * n)
    T = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    P = np.zeros((2 * n, 3))
    P[:, 0] = R * np.cos(T)
    P[:, 1] = R * np.sin(T)
    return P
paths = PathCollection('agg', color='shared')
triangles = TriangleCollection('raw', color='shared')
P0 = star()
(P1, tris) = triangulate(P0)
n = 1000
for i in range(n):
    c = i / float(n)
    (x, y) = np.random.uniform(-1, +1, 2)
    s = 25 / 800.0
    triangles.append(P1 * s + (x, y, i / 1000.0), tris, color=(1, 0, 0, 0.5))
    paths.append(P0 * s + (x, y, (i - 1) / 1000.0), closed=True, color=(0, 0, 0, 0.5))
paths['linewidth'] = 1.0
paths['viewport'] = (0, 0, 800, 800)

@canvas.connect
def on_draw(e):
    if False:
        i = 10
        return i + 15
    gloo.clear('white')
    triangles.draw()
    paths.draw()

@canvas.connect
def on_resize(event):
    if False:
        while True:
            i = 10
    (width, height) = event.size
    gloo.set_viewport(0, 0, width, height)
    paths['viewport'] = (0, 0, width, height)
app.run()