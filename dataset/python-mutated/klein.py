"""
Demonstration of the Klein bottle rendering using Mesh.
"""
import sys
from vispy import app, scene
from vispy.geometry.parametric import surface

def klein(u, v):
    if False:
        i = 10
        return i + 15
    from math import pi, cos, sin
    if u < pi:
        x = 3 * cos(u) * (1 + sin(u)) + 2 * (1 - cos(u) / 2) * cos(u) * cos(v)
        z = -8 * sin(u) - 2 * (1 - cos(u) / 2) * sin(u) * cos(v)
    else:
        x = 3 * cos(u) * (1 + sin(u)) + 2 * (1 - cos(u) / 2) * cos(v + pi)
        z = -8 * sin(u)
    y = -2 * (1 - cos(u) / 2) * sin(v)
    return (x / 5, y / 5, z / 5)
canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.ArcballCamera(parent=view.scene)
(vertices, indices) = surface(klein, urepeat=3)
indices = indices.reshape(len(indices) // 3, 3)
mesh = scene.visuals.Mesh(vertices=vertices['position'], faces=indices, color='white', parent=view.scene, shading='smooth')
if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()