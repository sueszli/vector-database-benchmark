"""
Example using texture internalformat for higher precision
=========================================================

Generates a gradient texture with high dynamic range and renders it
with a fragment shader that tests for quantization errors by comparing
adjacent texels and decomposes the gradient values into high and low
significance bits, mapping them to separate display color channels.

Pressing the 'f' key cycles through a list of different texture
formats to show the different levels of precision available.
"""
import numpy as np
from vispy import gloo
from vispy import app
(W, H) = (1024, 1024)
data = np.zeros((H, W, 3), np.float32)
for i in range(W):
    data[:, i, :] = i ** 2
for i in range(H):
    data[i, :, :] *= i ** 2
data *= 1.0 / data.max()
quad = np.zeros(4, dtype=[('a_position', np.float32, 2), ('a_texcoord', np.float32, 2)])
quad['a_position'] = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
quad['a_texcoord'] = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
vert_shader = '\nattribute vec2 a_position;\nattribute vec2 a_texcoord;\nvarying vec2 v_texcoord;\n\nvoid main()\n{\n   v_texcoord = a_texcoord;\n   gl_Position = vec4(a_position, 0.0, 1.0);\n}\n'
frag_shader = '\nuniform sampler2D u_texture;\nvarying vec2 v_texcoord;\n\nvoid main()\n{\n   float ndiff;\n   // an adjacent texel is 1/W further over in normalized texture coordinates\n   vec2 v_texcoord2 = vec2(clamp(v_texcoord.x + 1.0/%(W)d., 0.0, 1.0),\n                           v_texcoord.y);\n   vec4 texel1 = texture2D(u_texture, v_texcoord);\n   vec4 texel2 = texture2D(u_texture, v_texcoord2);\n\n   // test for quantized binning of adjacent texels\n   if (texel1.r == texel2.r && v_texcoord2.x < 1.0 && v_texcoord.y > 0.0)\n      ndiff = 1.0;\n   else\n      ndiff = 0.0;\n\n   gl_FragColor = vec4(\n      fract(texel1.r * 255.0),  // render low-significance bits as red\n      texel1.r,                 // render high-significance bits as green\n      ndiff,                    // flag quantized bands as blue\n      1);\n}\n' % dict(W=W)

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, size=(W, H), keys='interactive')
        self._internalformats = ['rgb8', 'rgb16', 'rgb16f', 'rgb32f']
        self.program = gloo.Program(vert_shader, frag_shader)
        self.program.bind(gloo.VertexBuffer(quad))
        self._internalformat = -1
        self.texture = gloo.Texture2D(shape=(H, W, 3), interpolation='nearest')
        gloo.set_viewport(0, 0, *self.physical_size)
        self.toggle_internalformat()
        self.show()

    def on_key_press(self, event):
        if False:
            i = 10
            return i + 15
        if event.key == 'F':
            self.toggle_internalformat()

    def toggle_internalformat(self):
        if False:
            for i in range(10):
                print('nop')
        self._internalformat = (self._internalformat + 1) % len(self._internalformats)
        internalformat = self._internalformats[self._internalformat]
        print('Requesting texture internalformat %s' % internalformat)
        self.texture.resize(data.shape, format='rgb', internalformat=internalformat)
        self.texture.set_data(data)
        self.program['u_texture'] = self.texture
        self.update()

    def on_resize(self, event):
        if False:
            return 10
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        if False:
            return 10
        gloo.clear(color=True, depth=True)
        self.program.draw('triangle_strip')
if __name__ == '__main__':
    c = Canvas()
    app.run()