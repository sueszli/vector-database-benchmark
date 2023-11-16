"""
Simplest Shader Program
=======================

"""
import sys
from vispy import gloo
from vispy import app
import numpy as np
VERT_SHADER = '\nattribute vec2 a_position;\nuniform float u_size;\n\nvoid main() {\n    gl_Position = vec4(a_position, 0.0, 1.0);\n    gl_PointSize = u_size;\n}\n'
FRAG_SHADER = '\nvoid main() {\n    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            while True:
                i = 10
        app.Canvas.__init__(self, keys='interactive')
        ps = self.pixel_scale
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        data = np.random.uniform(-0.5, 0.5, size=(20, 2))
        self.program['a_position'] = data.astype(np.float32)
        self.program['u_size'] = 20.0 * ps
        self.show()

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        (width, height) = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        gloo.clear('white')
        self.program.draw('points')
if __name__ == '__main__':
    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()