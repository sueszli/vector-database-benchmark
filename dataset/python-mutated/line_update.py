"""
Animated Line Visual
====================

Demonstration of animated Line visual.
"""
import sys
import numpy as np
from vispy import app, scene
N = 200
pos = np.zeros((N, 2), dtype=np.float32)
x_lim = [50.0, 750.0]
y_lim = [-2.0, 2.0]
pos[:, 0] = np.linspace(x_lim[0], x_lim[1], N)
pos[:, 1] = np.random.normal(size=N)
color = np.ones((N, 4), dtype=np.float32)
color[:, 0] = np.linspace(0, 1, N)
color[:, 1] = color[::-1, 0]
canvas = scene.SceneCanvas(keys='interactive', show=True)
grid = canvas.central_widget.add_grid(spacing=0)
viewbox = grid.add_view(row=0, col=1, camera='panzoom')
x_axis = scene.AxisWidget(orientation='bottom')
x_axis.stretch = (1, 0.1)
grid.add_widget(x_axis, row=1, col=1)
x_axis.link_view(viewbox)
y_axis = scene.AxisWidget(orientation='left')
y_axis.stretch = (0.1, 1)
grid.add_widget(y_axis, row=0, col=0)
y_axis.link_view(viewbox)
line = scene.Line(pos, color, parent=viewbox.scene)
viewbox.camera.set_range()

def update(ev):
    if False:
        i = 10
        return i + 15
    global pos, color, line
    pos[:, 1] = np.random.normal(size=N)
    color = np.roll(color, 1, axis=0)
    line.set_data(pos=pos, color=color)
timer = app.Timer()
timer.connect(update)
timer.start(0)
if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()