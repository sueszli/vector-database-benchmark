"""
Animate an Image
================

Use a timer to trigger updating an image.

This example demonstrates a 3D Texture. The volume contains noise that
is smoothed in the z-direction. Shown is one slice through that volume
to give the effect of "morphing" noise.
"""
import numpy as np
from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app
from vispy.visuals.shaders import ModularProgram
(D, H, W) = (30, 60, 90)
img_array = np.random.uniform(0, 0.1, (D, H, W, 3)).astype(np.float32)
img_array[...] += np.linspace(0, 0.9, D)[:, np.newaxis, np.newaxis, np.newaxis]
img_array[..., 1] *= np.linspace(0, 1, H)[np.newaxis, :, np.newaxis]
img_array[..., 0] *= np.linspace(0, 1, W)[np.newaxis, np.newaxis, :]
data = np.zeros(4, dtype=[('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)])
data['a_position'] = np.array([[0, 0], [W, 0], [0, H], [W, H]])
data['a_texcoord'] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
VERT_SHADER = '\n// Uniforms\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_projection;\n\n// Attributes\nattribute vec2 a_position;\nattribute vec2 a_texcoord;\n\n// Varyings\nvarying vec2 v_texcoord;\n\n// Main\nvoid main (void)\n{\n    v_texcoord = a_texcoord;\n    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);\n}\n'
FRAG_SHADER = '\nuniform $sampler_type u_texture;\nuniform float i;\nvarying vec2 v_texcoord;\nvoid main()\n{\n    // step through gradient with i, note that slice (depth) comes last here!\n    gl_FragColor = $sample(u_texture, vec3(v_texcoord, i));\n    gl_FragColor.a = 1.0;\n}\n\n'

class Canvas(app.Canvas):

    def __init__(self, emulate3d=True):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, keys='interactive', size=(W * 5, H * 5))
        if emulate3d:
            tex_cls = gloo.TextureEmulated3D
        else:
            tex_cls = gloo.Texture3D
        self.texture = tex_cls(img_array, interpolation='nearest', wrapping='clamp_to_edge')
        self.program = ModularProgram(VERT_SHADER, FRAG_SHADER)
        self.program.frag['sampler_type'] = self.texture.glsl_sampler_type
        self.program.frag['sample'] = self.texture.glsl_sample
        self.program['u_texture'] = self.texture
        self.program['i'] = 0.0
        self.program.bind(gloo.VertexBuffer(data))
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection
        self.i = 0
        gloo.set_clear_color('white')
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def on_resize(self, event):
        if False:
            while True:
                i = 10
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

    def on_timer(self, event):
        if False:
            while True:
                i = 10
        self.i = (self.i + 1.0 / 120.0) % 1.0
        self.update()

    def on_draw(self, event):
        if False:
            return 10
        gloo.clear(color=True, depth=True)
        self.program['i'] = 1.9 * np.abs(0.5 - self.i)
        self.program.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas(emulate3d=True)
    app.run()