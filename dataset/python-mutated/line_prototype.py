import sys
import numpy as np
from vispy import app, gloo, visuals
from vispy.visuals.filters import Clipper, ColorFilter
from vispy.visuals.shaders import MultiProgram
from vispy.visuals.collections import PointCollection
from vispy.visuals.transforms import STTransform
from vispy.scene import SceneCanvas
from vispy.scene.visuals import create_visual_node

class LineVisual(visuals.Visual):
    """Example of a very simple GL-line visual.

    This shows the minimal set of methods that need to be reimplemented to 
    make a new visual class.

    """

    def __init__(self, pos=None, color=(1, 1, 1, 1)):
        if False:
            for i in range(10):
                print('nop')
        vcode = '\n        attribute vec2 a_pos;\n        \n        void main() {\n            gl_Position = $transform(vec4(a_pos, 0., 1.)); \n            gl_PointSize = 10.;\n        }\n        '
        fcode = '\n        void main() {\n            gl_FragColor = $color;\n        }\n        '
        visuals.Visual.__init__(self, vcode=vcode, fcode=fcode)
        self.pos_buf = gloo.VertexBuffer()
        self.shared_program['a_pos'] = self.pos_buf
        self.shared_program.frag['color'] = color
        self._need_upload = False
        self._draw_mode = 'line_strip'
        self.set_gl_state('translucent', depth_test=False)
        if pos is not None:
            self.set_data(pos)

    def set_data(self, pos):
        if False:
            print('Hello World!')
        self._pos = pos
        self._need_upload = True

    def _prepare_transforms(self, view=None):
        if False:
            for i in range(10):
                print('nop')
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view=None):
        if False:
            print('Hello World!')
        'This method is called immediately before each draw.\n\n        The *view* argument indicates which view is about to be drawn.\n        '
        if self._need_upload:
            self.pos_buf.set_data(self._pos)
            self._need_upload = False

class PointVisual(LineVisual):
    """Another simple visual class. 

    Due to the simplicity of these example classes, it was only necessary to
    subclass from LineVisual and set the draw mode to 'points'. A more
    fully-featured PointVisual class might not follow this approach.
    """

    def __init__(self, pos=None, color=(1, 1, 1, 1)):
        if False:
            i = 10
            return i + 15
        LineVisual.__init__(self, pos, color)
        self._draw_mode = 'points'

class PlotLineVisual(visuals.CompoundVisual):
    """An example compound visual that draws lines and points.

    To the user, the compound visual behaves exactly like a normal visual--it
    has a transform system, draw() and bounds() methods, etc. Internally, the
    compound visual automatically manages proxying these transforms and methods
    to its sub-visuals.
    """

    def __init__(self, pos=None, line_color=(1, 1, 1, 1), point_color=(1, 1, 1, 1)):
        if False:
            i = 10
            return i + 15
        self._line = LineVisual(pos, color=line_color)
        self._point = PointVisual(pos, color=point_color)
        visuals.CompoundVisual.__init__(self, [self._line, self._point])

class PointCollectionVisual(visuals.Visual):
    """Thin wrapper around a point collection.

    Note: This is currently broken!
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        prog = MultiProgram(vcode='', fcode='')
        self.points = PointCollection('agg', color='shared', program=prog)
        visuals.Visual.__init__(self, program=prog)

    def _prepare_draw(self, view):
        if False:
            while True:
                i = 10
        if self.points._need_update:
            self.points._update()
        self._draw_mode = self.points._mode
        self._index_buffer = self.points._indices_buffer

    def append(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.points.append(*args, **kwargs)

    def _prepare_transforms(self, view=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def color(self):
        if False:
            return 10
        return self.points['color']

    @color.setter
    def color(self, c):
        if False:
            while True:
                i = 10
        self.points['color'] = c

class PanZoomTransform(STTransform):

    def __init__(self, canvas=None, aspect=None, **kwargs):
        if False:
            print('Hello World!')
        self._aspect = aspect
        self.attach(canvas)
        STTransform.__init__(self, **kwargs)

    def attach(self, canvas):
        if False:
            print('Hello World!')
        ' Attach this tranform to a canvas '
        self._canvas = canvas
        canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        canvas.events.mouse_move.connect(self.on_mouse_move)

    def on_mouse_move(self, event):
        if False:
            while True:
                i = 10
        if event.is_dragging:
            dxy = event.pos - event.last_event.pos
            button = event.press_event.button
            if button == 1:
                self.move(dxy)
            elif button == 2:
                center = event.press_event.pos
                if self._aspect is None:
                    self.zoom(np.exp(dxy * (0.01, -0.01)), center)
                else:
                    s = dxy[1] * -0.01
                    self.zoom(np.exp(np.array([s, s])), center)

    def on_mouse_wheel(self, event):
        if False:
            i = 10
            return i + 15
        self.zoom(np.exp(event.delta * (0.01, -0.01)), event.pos)
canvas = app.Canvas(keys='interactive', size=(900, 600), show=True, title='Visual Canvas')
pos = np.random.normal(size=(1000, 2), loc=0, scale=50).astype('float32')
pos[0] = [0, 0]
line = LineVisual(pos=pos)
line.transforms.canvas = canvas
line.transform = STTransform(scale=(2, 1), translate=(20, 20))
panzoom = PanZoomTransform(canvas)
line.transforms.scene_transform = panzoom
panzoom.changed.connect(lambda ev: canvas.update())
line.attach(ColorFilter((1, 1, 0.5, 0.7)))
tr = line.transforms.get_transform('framebuffer', 'canvas')
line.attach(Clipper((20, 20, 260, 260), transform=tr), view=line)
shadow = line.view()
shadow.transforms.canvas = canvas
shadow.transform = STTransform(scale=(2, 1), translate=(25, 25))
shadow.transforms.scene_transform = panzoom
shadow.attach(ColorFilter((0, 0, 0, 0.6)), view=shadow)
tr = shadow.transforms.get_transform('framebuffer', 'canvas')
shadow.attach(Clipper((20, 20, 260, 260), transform=tr), view=shadow)
view = line.view()
view.transforms.canvas = canvas
view.transform = STTransform(scale=(2, 0.5), translate=(450, 150))
tr = view.transforms.get_transform('framebuffer', 'canvas')
view.attach(Clipper((320, 20, 260, 260), transform=tr), view=view)
plot = PlotLineVisual(pos, (0.5, 1, 0.5, 0.2), (0.5, 1, 1, 0.3))
plot.transforms.canvas = canvas
plot.transform = STTransform(translate=(80, 450), scale=(1.5, 1))
tr = plot.transforms.get_transform('framebuffer', 'canvas')
plot.attach(Clipper((20, 320, 260, 260), transform=tr), view=plot)
view2 = plot.view()
view2.transforms.canvas = canvas
view2.transform = STTransform(scale=(1.5, 1), translate=(450, 400))
tr = view2.transforms.get_transform('framebuffer', 'canvas')
view2.attach(Clipper((320, 320, 260, 260), transform=tr), view=view2)
shadow2 = plot.view()
shadow2.transforms.canvas = canvas
shadow2.transform = STTransform(scale=(1.5, 1), translate=(455, 405))
shadow2.attach(ColorFilter((0, 0, 0, 0.6)), view=shadow2)
tr = shadow2.transforms.get_transform('framebuffer', 'canvas')
shadow2.attach(Clipper((320, 320, 260, 260), transform=tr), view=shadow2)
collection = PointCollectionVisual()
collection.transforms.canvas = canvas
collection.transform = STTransform(translate=(750, 150))
collection.append(np.random.normal(loc=0, scale=20, size=(10000, 3)), itemsize=5000)
collection.color = ((1, 0.5, 0.5, 1), (0.5, 0.5, 1, 1))
shadow3 = collection.view()
shadow3.transforms.canvas = canvas
shadow3.transform = STTransform(scale=(1, 1), translate=(752, 152))
shadow3.attach(ColorFilter((0, 0, 0, 0.6)), view=shadow3)
order = [shadow, line, view, plot, shadow2, view2, shadow3, collection]

@canvas.connect
def on_draw(event):
    if False:
        i = 10
        return i + 15
    canvas.context.clear((0.3, 0.3, 0.3, 1.0))
    for v in order:
        v.draw()

def on_resize(event):
    if False:
        print('Hello World!')
    vp = (0, 0, canvas.physical_size[0], canvas.physical_size[1])
    canvas.context.set_viewport(*vp)
    for v in order:
        v.transforms.configure(canvas=canvas, viewport=vp)
canvas.events.resize.connect(on_resize)
on_resize(None)
Line = create_visual_node(LineVisual)
canvas2 = SceneCanvas(keys='interactive', title='Scene Canvas', show=True)
v = canvas2.central_widget.add_view(margin=10)
v.border_color = (1, 1, 1, 1)
v.bgcolor = (0.3, 0.3, 0.3, 1)
v.camera = 'panzoom'
line2 = Line(pos, parent=v.scene)

def mouse(ev):
    if False:
        while True:
            i = 10
    print(ev)
v.events.mouse_press.connect(mouse)
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()