"""
Isosurface Visual
=================

This example demonstrates the use of the Isosurface visual.
"""
import sys
import numpy as np
from vispy import app, scene
canvas = scene.SceneCanvas(keys='interactive')
view = canvas.central_widget.add_view()

def psi(i, j, k, offset=(25, 25, 50)):
    if False:
        i = 10
        return i + 15
    x = i - offset[0]
    y = j - offset[1]
    z = k - offset[2]
    th = np.arctan2(z, (x ** 2 + y ** 2) ** 0.5)
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    a0 = 1
    ps = 1.0 / 81.0 * 1.0 / (6.0 * np.pi) ** 0.5 * (1.0 / a0) ** (3 / 2) * (r / a0) ** 2 * np.exp(-r / (3 * a0)) * (3 * np.cos(th) ** 2 - 1)
    return ps
print('Generating scalar field..')
data = np.abs(np.fromfunction(psi, (50, 50, 100)))
surface = scene.visuals.Isosurface(data, level=data.max() / 4.0, color=(0.5, 0.6, 1, 1), shading='smooth', parent=view.scene)
surface.transform = scene.transforms.STTransform(translate=(-25, -25, -50))
axis = scene.visuals.XYZAxis(parent=view.scene)
cam = scene.TurntableCamera(elevation=30, azimuth=30)
cam.set_range((-10, 10), (-10, 10), (-10, 10))
view.camera = cam
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()