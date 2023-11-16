"""
Use multiple textures
=====================

We create two textures. One that shows a red, green and blue band in
the horizontal direction and one that does the same in the vertical
direction. In the fragment shader the colors from both textures are
added.

"""
import numpy as np
from vispy import gloo
from vispy import app
(W, H) = (30, 30)
im1 = np.zeros((W, H, 3), np.float32)
im2 = np.zeros((W, H, 3), np.float32)
im1[:10, :, 0] = 1.0
im1[10:20, :, 1] = 1.0
im1[20:, :, 2] = 1.0
im2[:, :10, 0] = 1.0
im2[:, 10:20, 1] = 1.0
im1[:, 20:, 2] = 1.0
data = np.zeros(4, dtype=[('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
data['a_texcoord'] = np.array([[1, 0], [1, 1.2], [0, 0], [0, 1.2]])
VERT_SHADER = '\nattribute vec2 a_position;\nattribute vec2 a_texcoord;\nvarying vec2 v_texcoord;\n\nvoid main (void)\n{\n    v_texcoord = a_texcoord;\n    gl_Position = vec4(a_position, 0.0, 1.0);\n}\n'
FRAG_SHADER = '\nuniform sampler2D u_tex1;\nuniform sampler2D u_tex2;\nvarying vec2 v_texcoord;\n\nvoid main()\n{\n    vec3 clr1 = texture2D(u_tex1, v_texcoord).rgb;\n    vec3 clr2 = texture2D(u_tex2, v_texcoord).rgb;\n    gl_FragColor.rgb = clr1 + clr2;\n    gl_FragColor.a = 1.0;\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, size=(500, 500), keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['u_tex1'] = gloo.Texture2D(im1, interpolation='linear')
        self.program['u_tex2'] = gloo.Texture2D(im2, interpolation='linear')
        self.program.bind(gloo.VertexBuffer(data))
        gloo.set_clear_color('white')
        self.show()

    def on_resize(self, event):
        if False:
            while True:
                i = 10
        (width, height) = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear(color=True, depth=True)
        self.program.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()