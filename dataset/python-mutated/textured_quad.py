"""
Show a textured quad
====================
"""
import numpy as np
from vispy import gloo, app
from vispy.gloo import Program
vertex = '\n    attribute vec2 position;\n    attribute vec2 texcoord;\n    varying vec2 v_texcoord;\n    void main()\n    {\n        gl_Position = vec4(position, 0.0, 1.0);\n        v_texcoord = texcoord;\n    } '
fragment = '\n    uniform sampler2D texture;\n    varying vec2 v_texcoord;\n    void main()\n    {\n        gl_FragColor = texture2D(texture, v_texcoord);\n    } '

def checkerboard(grid_num=8, grid_size=32):
    if False:
        i = 10
        return i + 15
    row_even = grid_num // 2 * [0, 1]
    row_odd = grid_num // 2 * [1, 0]
    Z = np.row_stack(grid_num // 2 * (row_even, row_odd)).astype(np.uint8)
    return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        app.Canvas.__init__(self, size=(512, 512), title='Textured quad', keys='interactive')
        self.program = Program(vertex, fragment, count=4)
        self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.program['texcoord'] = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.program['texture'] = checkerboard()
        gloo.set_viewport(0, 0, *self.physical_size)
        self.show()

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.set_clear_color('white')
        gloo.clear(color=True)
        self.program.draw('triangle_strip')

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.set_viewport(0, 0, *event.physical_size)
if __name__ == '__main__':
    c = Canvas()
    app.run()