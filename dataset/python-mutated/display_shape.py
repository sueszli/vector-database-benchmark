"""
Draw a Quad
===========

Simple example demonstrating showing a quad using a gloo Program.

"""
from vispy import gloo
from vispy import app
import numpy as np
vPosition = np.array([[-0.8, -0.8, 0.0], [+0.7, -0.7, 0.0], [-0.7, +0.7, 0.0], [+0.8, +0.8, 0.0]], np.float32)
VERT_SHADER = ' // simple vertex shader\nattribute vec3 a_position;\nvoid main (void) {\n    gl_Position = vec4(a_position, 1.0);\n}\n'
FRAG_SHADER = ' // simple fragment shader\nuniform vec4 u_color;\nvoid main()\n{\n    gl_FragColor = u_color;\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(keys='interactive')
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._program['u_color'] = (0.2, 1.0, 0.4, 1)
        self._program['a_position'] = gloo.VertexBuffer(vPosition)
        gloo.set_clear_color('white')
        self.show()

    def on_resize(self, event):
        if False:
            i = 10
            return i + 15
        (width, height) = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        if False:
            return 10
        gloo.clear()
        self._program.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()