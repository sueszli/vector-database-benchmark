"""
Example demonstrating simulation of fireworks using point sprites.
(adapted from the "OpenGL ES 2.0 Programming Guide")

This example demonstrates a series of explosions that last one second. The
visualization during the explosion is highly optimized using a Vertex Buffer
Object (VBO). After each explosion, vertex data for the next explosion are
calculated, such that each explostion is unique.
"""
import time
import numpy as np
from vispy import gloo, app
radius = 32
im1 = np.random.normal(0.8, 0.3, (radius * 2 + 1, radius * 2 + 1)).astype(np.float32)
L = np.linspace(-radius, radius, 2 * radius + 1)
(X, Y) = np.meshgrid(L, L)
im1 *= np.array(X ** 2 + Y ** 2 <= radius * radius, dtype='float32')
N = 10000
data = np.zeros(N, [('a_lifetime', np.float32), ('a_startPosition', np.float32, 3), ('a_endPosition', np.float32, 3)])
VERT_SHADER = '\nuniform float u_time;\nuniform vec3 u_centerPosition;\nattribute float a_lifetime;\nattribute vec3 a_startPosition;\nattribute vec3 a_endPosition;\nvarying float v_lifetime;\n\nvoid main () {\n    if (u_time <= a_lifetime)\n    {\n        gl_Position.xyz = a_startPosition + (u_time * a_endPosition);\n        gl_Position.xyz += u_centerPosition;\n        gl_Position.y -= 1.0 * u_time * u_time;\n        gl_Position.w = 1.0;\n    }\n    else\n        gl_Position = vec4(-1000, -1000, 0, 0);\n\n    v_lifetime = 1.0 - (u_time / a_lifetime);\n    v_lifetime = clamp(v_lifetime, 0.0, 1.0);\n    gl_PointSize = (v_lifetime * v_lifetime) * 40.0;\n}\n'
FRAG_SHADER = '\n#version 120\nprecision highp float;\nuniform sampler2D texture1;\nuniform vec4 u_color;\nvarying float v_lifetime;\nuniform highp sampler2D s_texture;\n\nvoid main()\n{\n    highp vec4 texColor;\n    texColor = texture2D(s_texture, gl_PointCoord);\n    gl_FragColor = vec4(u_color) * texColor;\n    gl_FragColor.a *= v_lifetime;\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))
        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._program.bind(gloo.VertexBuffer(data))
        self._program['s_texture'] = gloo.Texture2D(im1)
        self._new_explosion()
        gloo.set_state(blend=True, clear_color='black', blend_func=('src_alpha', 'one'))
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_resize(self, event):
        if False:
            print('Hello World!')
        (width, height) = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear()
        self._program['u_time'] = time.time() - self._starttime
        self._program.draw('points')
        if time.time() - self._starttime > 1.5:
            self._new_explosion()

    def _new_explosion(self):
        if False:
            for i in range(10):
                print('nop')
        centerpos = np.random.uniform(-0.5, 0.5, (3,))
        self._program['u_centerPosition'] = centerpos
        alpha = 1.0 / N ** 0.08
        color = np.random.uniform(0.1, 0.9, (3,))
        self._program['u_color'] = tuple(color) + (alpha,)
        data['a_lifetime'] = np.random.normal(2.0, 0.5, (N,))
        data['a_startPosition'] = np.random.normal(0.0, 0.2, (N, 3))
        data['a_endPosition'] = np.random.normal(0.0, 1.2, (N, 3))
        self._starttime = time.time()
if __name__ == '__main__':
    c = Canvas()
    app.run()