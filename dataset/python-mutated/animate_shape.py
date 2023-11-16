"""
Animate a Shape
===============

Example demonstrating showing a quad using
a Texture2D and VertexBuffer and a timer to control the drawing.

"""
import time
import numpy as np
from vispy import gloo
from vispy import app
im1 = np.zeros((100, 100, 3), 'float32')
im1[:50, :, 0] = 1.0
im1[:, :50, 1] = 1.0
im1[50:, 50:, 2] = 1.0
vertex_data = np.zeros(4, dtype=[('a_position', np.float32, 3), ('a_texcoord', np.float32, 2)])
vertex_data['a_position'] = np.array([[-0.8, -0.8, 0.0], [+0.7, -0.7, 0.0], [-0.7, +0.7, 0.0], [+0.8, +0.8, 0.0]])
vertex_data['a_texcoord'] = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
indices = np.array([0, 1, 2, 1, 2, 3], np.uint16)
indices_buffer = gloo.IndexBuffer(indices)
client_indices_buffer = gloo.IndexBuffer(indices)
VERT_SHADER = ' // simple vertex shader\n\nattribute vec3 a_position;\nattribute vec2 a_texcoord;\nuniform float sizeFactor;\n\nvoid main (void) {\n    // Pass tex coords\n    gl_TexCoord[0] = vec4(a_texcoord.x, a_texcoord.y, 0.0, 0.0);\n    // Calculate position\n    gl_Position = sizeFactor*vec4(a_position.x, a_position.y, a_position.z,\n                                                        1.0/sizeFactor);\n}\n'
FRAG_SHADER = ' // simple fragment shader\nuniform sampler2D texture1;\n\nvoid main()\n{\n    gl_FragColor = texture2D(texture1, gl_TexCoord[0].st);\n}\n\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, keys='interactive')
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._vbo = gloo.VertexBuffer(vertex_data)
        self._program['texture1'] = gloo.Texture2D(im1)
        self._program.bind(self._vbo)
        gloo.set_clear_color('white')
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_resize(self, event):
        if False:
            print('Hello World!')
        (width, height) = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        gloo.clear()
        self._program['sizeFactor'] = 0.5 + np.sin(time.time() * 3) * 0.2
        self._program.draw('triangles', indices_buffer)
if __name__ == '__main__':
    canvas = Canvas()
    app.run()