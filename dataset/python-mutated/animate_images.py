"""
Show an Image
=============

Draw a new array of random image pixels on every draw cycle.

"""
import numpy as np
from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app
(W, H) = (64, 48)
img_array = np.random.uniform(0, 1, (W, H)).astype(np.float32)
data = np.zeros(4, dtype=[('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[0, 0], [W, 0], [0, H], [W, H]])
data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
VERT_SHADER = '\n// Uniforms\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_projection;\nuniform float u_antialias;\n\n// Attributes\nattribute vec2 a_position;\nattribute vec2 a_texcoord;\n\n// Varyings\nvarying vec2 v_texcoord;\n\n// Main\nvoid main (void)\n{\n    v_texcoord = a_texcoord;\n    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);\n}\n'
FRAG_SHADER = '\nuniform sampler2D u_texture;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    gl_FragColor = texture2D(u_texture, v_texcoord);\n    gl_FragColor.a = 1.0;\n}\n\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            return 10
        app.Canvas.__init__(self, keys='interactive', size=(W * 5, H * 5))
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(img_array, interpolation='linear')
        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(data))
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection
        gloo.set_clear_color('white')
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_resize(self, event):
        if False:
            print('Hello World!')
        (width, height) = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = ortho(0, width, 0, height, -100, 100)
        self.program['u_projection'] = self.projection
        r = width / float(height)
        R = W / float(H)
        if r < R:
            (w, h) = (width, width / R)
            (x, y) = (0, int((height - h) / 2))
        else:
            (w, h) = (height * R, height)
            (x, y) = (int((width - w) / 2), 0)
        data['a_position'] = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        self.program.bind(gloo.VertexBuffer(data))

    def on_draw(self, event):
        if False:
            print('Hello World!')
        gloo.clear(color=True, depth=True)
        img_array[...] = np.random.uniform(0, 1, (W, H)).astype(np.float32)
        self.texture.set_data(img_array)
        self.program.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()