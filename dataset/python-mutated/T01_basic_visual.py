"""
Tutorial: Creating Visuals
--------------------------

This tutorial is intended to guide developers who are interested in creating 
new subclasses of Visual. In most cases, this will not be necessary because
vispy's base library of visuals will be sufficient to create complex scenes as
needed. However, there are cases where a particular visual effect is desired 
that is not supported in the base library, or when a custom visual is needed to
optimize performance for a specific use case.

The purpose of a Visual is to encapsulate a single drawable object. This
drawable can be as simple or complex as desired. Some of the simplest visuals 
draw points, lines, or triangles, whereas more complex visuals invove multiple
drawing stages or make use of sub-visuals to construct larger objects.

In this example we will create a very simple Visual that draws a rectangle.
Visuals are defined by:

1. Creating a subclass of vispy.visuals.Visual that specifies the GLSL code
   and buffer objects to use.
2. Defining a _prepare_transforms() method that will be called whenever the
   user (or scenegraph) assigns a new set of transforms to the visual.


"""
from vispy import app, gloo, visuals, scene
import numpy as np
vertex_shader = '\nvoid main() {\n   gl_Position = $transform(vec4($position, 0, 1));\n}\n'
fragment_shader = '\nvoid main() {\n  gl_FragColor = $color;\n}\n'

class MyRectVisual(visuals.Visual):
    """Visual that draws a red rectangle.

    Parameters
    ----------
    x : float
        x coordinate of rectangle origin
    y : float
        y coordinate of rectangle origin
    w : float
        width of rectangle
    h : float
        height of rectangle

    All parameters are specified in the local (arbitrary) coordinate system of
    the visual. How this coordinate system translates to the canvas will 
    depend on the transformation functions used during drawing.
    """

    def __init__(self, x, y, w, h):
        if False:
            while True:
                i = 10
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self.vbo = gloo.VertexBuffer(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], dtype=np.float32))
        self.shared_program.vert['position'] = self.vbo
        self.shared_program.frag['color'] = (1, 0, 0, 1)
        self._draw_mode = 'triangles'

    def _prepare_transforms(self, view):
        if False:
            return 10
        view.view_program.vert['transform'] = view.get_transform()
MyRect = scene.visuals.create_visual_node(MyRectVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True)
rects = [MyRect(100, 100, 200, 300, parent=canvas.scene), MyRect(500, 100, 200, 300, parent=canvas.scene)]
tr = visuals.transforms.MatrixTransform()
tr.rotate(5, (0, 0, 1))
rects[1].transform = tr
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()