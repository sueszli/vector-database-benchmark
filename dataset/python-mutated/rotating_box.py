"""
Demonstration of Cube
"""
import sys
from vispy import app, gloo
from vispy.visuals import BoxVisual, transforms

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            while True:
                i = 10
        app.Canvas.__init__(self, 'Cube', keys='interactive', size=(400, 400))
        self.cube = BoxVisual(1.0, 0.5, 0.25, color='red', edge_color='k')
        self.theta = 0
        self.phi = 0
        self.cube_transform = transforms.MatrixTransform()
        self.cube.transform = self.cube_transform
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_resize(self, event):
        if False:
            return 10
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.cube.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.clear('white', depth=True)
        self.cube.draw()

    def on_timer(self, event):
        if False:
            print('Hello World!')
        self.theta += 0.5
        self.phi += 0.5
        self.cube_transform.reset()
        self.cube_transform.rotate(self.theta, (0, 0, 1))
        self.cube_transform.rotate(self.phi, (0, 1, 0))
        self.cube_transform.scale((100, 100, 0.001))
        self.cube_transform.translate((200, 200))
        self.update()
if __name__ == '__main__':
    win = Canvas()
    win.show()
    if sys.flags.interactive != 1:
        win.app.run()