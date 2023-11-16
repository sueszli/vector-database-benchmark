"""
Use FrameBuffers
================

Minimal example demonstrating the use of frame buffer objects (FBO).
This example blurs the output image.

"""
from vispy import gloo
from vispy import app
import numpy as np
vPosition = np.array([[-0.8, -0.8, 0.0], [+0.7, -0.7, 0.0], [-0.7, +0.7, 0.0], [+0.8, +0.8, 0.0]], np.float32)
vPosition_full = np.array([[-1.0, -1.0, 0.0], [+1.0, -1.0, 0.0], [-1.0, +1.0, 0.0], [+1.0, +1.0, 0.0]], np.float32)
vTexcoord = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], np.float32)
VERT_SHADER1 = '\nattribute vec3 a_position;\nvoid main (void) {\n    gl_Position = vec4(a_position, 1.0);\n}\n'
FRAG_SHADER1 = '\nuniform vec4 u_color;\nvoid main()\n{\n    gl_FragColor = u_color;\n}\n'
VERT_SHADER2 = '\nattribute vec3 a_position;\nattribute vec2 a_texcoord;\nvarying vec2 v_texcoord;\nvoid main (void) {\n    // Pass tex coords\n    v_texcoord = a_texcoord;\n    // Calculate position\n    gl_Position = vec4(a_position.x, a_position.y, a_position.z, 1.0);\n}\n'
FRAG_SHADER2 = '\nuniform sampler2D u_texture1;\nvarying vec2 v_texcoord;\nconst float c_zero = 0.0;\nconst int c_sze = 5;\nvoid main()\n{\n    float scalefactor = 1.0 / float(c_sze * c_sze * 4 + 1);\n    gl_FragColor = vec4(c_zero, c_zero, c_zero, 1.0);\n    for (int y=-c_sze; y<=c_sze; y++) {\n        for (int x=-c_sze; x<=c_sze; x++) {\n            vec2 step = vec2(x,y) * 0.01;\n            vec3 color = texture2D(u_texture1, v_texcoord.st+step).rgb;\n            gl_FragColor.rgb += color * scalefactor;\n        }\n    }\n}\n'
SIZE = 50

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        app.Canvas.__init__(self, keys='interactive', size=(560, 420))
        shape = (self.physical_size[1], self.physical_size[0])
        self._rendertex = gloo.Texture2D(shape + (3,))
        self._fbo = gloo.FrameBuffer(self._rendertex, gloo.RenderBuffer(shape))
        self._program1 = gloo.Program(VERT_SHADER1, FRAG_SHADER1)
        self._program1['u_color'] = (0.9, 1.0, 0.4, 1)
        self._program1['a_position'] = gloo.VertexBuffer(vPosition)
        self._program2 = gloo.Program(VERT_SHADER2, FRAG_SHADER2)
        self._program2['a_position'] = gloo.VertexBuffer(vPosition)
        self._program2['a_texcoord'] = gloo.VertexBuffer(vTexcoord)
        self._program2['u_texture1'] = self._rendertex
        self.show()

    def on_resize(self, event):
        if False:
            while True:
                i = 10
        (width, height) = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        with self._fbo:
            gloo.set_clear_color((0.0, 0.0, 0.5, 1))
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, *self.physical_size)
            self._program1.draw('triangle_strip')
        gloo.set_clear_color('white')
        gloo.clear(color=True, depth=True)
        self._program2.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()