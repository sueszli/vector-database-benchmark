"""
Simple demonstration of mouse drawing and editing of a line plot.
This demo extends the Line visual from scene adding mouse events that allow
modification and creation of line points with the mouse.
Vispy takes care of coordinate transforms from screen to ViewBox - the
demo works on different zoom levels.
"""
import numpy as np
from vispy import app, scene

class EditLineVisual(scene.visuals.Line):
    """
    Mouse editing extension to the Line visual.
    This class adds mouse picking for line points, mouse_move handling for
    dragging existing points, and
    adding new points when clicking into empty space.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        scene.visuals.Line.__init__(self, *args, **kwargs)
        self.unfreeze()
        self.markers = scene.visuals.Markers(parent=self)
        self.marker_colors = np.ones((len(self.pos), 4), dtype=np.float32)
        self.markers.set_data(pos=self.pos, symbol='s', edge_color='red', size=6)
        self.selected_point = None
        self.selected_index = -1
        self.gridsize = 10
        self.freeze()

    def on_draw(self, event):
        if False:
            print('Hello World!')
        scene.visuals.Line.draw(self)
        self.markers.draw()

    def print_mouse_event(self, event, what):
        if False:
            print('Hello World!')
        ' print mouse events for debugging purposes '
        print('%s - pos: %r, button: %s,  delta: %r' % (what, event.pos, event.button, event.delta))

    def select_point(self, pos_scene, radius=5):
        if False:
            while True:
                i = 10
        '\n        Get line point close to mouse pointer and its index\n\n        Parameters\n        ----------\n        event : the mouse event being processed\n        radius : scalar\n            max. distance in pixels between mouse and line point to be accepted\n        return: (numpy.array, int)\n            picked point and index of the point in the pos array\n        '
        mouse_radius = 6
        index = 0
        for p in self.pos:
            if np.linalg.norm(pos_scene[:3] - p) < mouse_radius:
                return (p, index)
            index += 1
        return (None, -1)

    def update_markers(self, selected_index=-1, highlight_color=(1, 0, 0, 1)):
        if False:
            while True:
                i = 10
        ' update marker colors, and highlight a marker with a given color '
        self.marker_colors.fill(1)
        shape = 'o'
        size = 6
        if 0 <= selected_index < len(self.marker_colors):
            self.marker_colors[selected_index] = highlight_color
            shape = 's'
            size = 8
        self.markers.set_data(pos=self.pos, symbol=shape, edge_color='red', size=size, face_color=self.marker_colors)

    def on_mouse_press(self, pos_scene):
        if False:
            print('Hello World!')
        (self.selected_point, self.selected_index) = self.select_point(pos_scene)
        if self.selected_point is None:
            print('adding point', len(self.pos))
            self._pos = np.append(self.pos, [pos_scene[:3]], axis=0)
            self.set_data(pos=self.pos)
            self.marker_colors = np.ones((len(self.pos), 4), dtype=np.float32)
            self.selected_point = self.pos[-1]
            self.selected_index = len(self.pos) - 1
        self.update_markers(self.selected_index)

    def on_mouse_release(self, event):
        if False:
            while True:
                i = 10
        self.print_mouse_event(event, 'Mouse release')
        self.selected_point = None
        self.update_markers()

    def on_mouse_move(self, pos_scene):
        if False:
            while True:
                i = 10
        if self.selected_point is not None:
            self.selected_point[0] = round(pos_scene[0] / self.gridsize) * self.gridsize
            self.selected_point[1] = round(pos_scene[1] / self.gridsize) * self.gridsize
            self.set_data(pos=self.pos)
            self.update_markers(self.selected_index)

    def highlight_markers(self, pos_scene):
        if False:
            i = 10
            return i + 15
        (hl_point, hl_index) = self.select_point(pos_scene)
        self.update_markers(hl_index, highlight_color=(0.5, 0.5, 1.0, 1.0))
        self.update()

class Canvas(scene.SceneCanvas):
    """ A simple test canvas for testing the EditLineVisual """

    def __init__(self):
        if False:
            return 10
        scene.SceneCanvas.__init__(self, keys='interactive', size=(800, 800))
        n = 7
        self.unfreeze()
        self.pos = np.zeros((n, 3), dtype=np.float32)
        self.pos[:, 0] = np.linspace(-50, 50, n)
        self.pos[:, 1] = np.random.normal(size=n, scale=10, loc=0)
        self.line = EditLineVisual(pos=self.pos, color='w', width=3, antialias=True, method='gl')
        self.view = self.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(-100, -100, 200, 200), aspect=1.0)
        self.view.camera._viewbox.events.mouse_move.disconnect(self.view.camera.viewbox_mouse_event)
        self.view.add(self.line)
        self.show()
        self.selected_point = None
        scene.visuals.GridLines(parent=self.view.scene)
        self.freeze()

    def on_mouse_press(self, event):
        if False:
            for i in range(10):
                print('nop')
        tr = self.scene.node_transform(self.line)
        pos = tr.map(event.pos)
        self.line.on_mouse_press(pos)

    def on_mouse_move(self, event):
        if False:
            print('Hello World!')
        tr = self.scene.node_transform(self.line)
        pos = tr.map(event.pos)
        if event.button == 1:
            self.line.on_mouse_move(pos)
        else:
            self.line.highlight_markers(pos)
if __name__ == '__main__':
    win = Canvas()
    app.run()