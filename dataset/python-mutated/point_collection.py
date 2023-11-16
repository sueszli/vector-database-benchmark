import sys
import numpy as np
from vispy import app, gloo
from vispy.visuals.collections import PointCollection
from vispy.visuals.transforms import PanZoomTransform
canvas = app.Canvas(size=(800, 600), show=True, keys='interactive')
gloo.set_viewport(0, 0, canvas.size[0], canvas.size[1])
gloo.set_state('translucent', depth_test=False)
panzoom = PanZoomTransform(canvas)
points = PointCollection('agg', color='shared', transform=panzoom)
points.append(np.random.normal(0.0, 0.5, (10000, 3)), itemsize=5000)
points['color'] = ((1, 0, 0, 1), (0, 0, 1, 1))
points.update.connect(canvas.update)

@canvas.connect
def on_draw(event):
    if False:
        while True:
            i = 10
    gloo.clear('white')
    points.draw()

@canvas.connect
def on_resize(event):
    if False:
        print('Hello World!')
    (width, height) = event.size
    gloo.set_viewport(0, 0, width, height)
if __name__ == '__main__' and sys.flags.interactive == 0:
    app.run()