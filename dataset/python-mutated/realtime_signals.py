"""
Multiple real-time digital signals with GLSL-based clipping.
"""
from vispy import gloo
from vispy import app
import numpy as np
import math
nrows = 16
ncols = 20
m = nrows * ncols
n = 1000
amplitudes = 0.1 + 0.2 * np.random.rand(m, 1).astype(np.float32)
y = amplitudes * np.random.randn(m, n).astype(np.float32)
color = np.repeat(np.random.uniform(size=(m, 3), low=0.5, high=0.9), n, axis=0).astype(np.float32)
index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n), np.repeat(np.tile(np.arange(nrows), ncols), n), np.tile(np.arange(n), m)].astype(np.float32)
VERT_SHADER = '\n#version 120\n\n// y coordinate of the position.\nattribute float a_position;\n\n// row, col, and time index.\nattribute vec3 a_index;\nvarying vec3 v_index;\n\n// 2D scaling factor (zooming).\nuniform vec2 u_scale;\n\n// Size of the table.\nuniform vec2 u_size;\n\n// Number of samples per signal.\nuniform float u_n;\n\n// Color.\nattribute vec3 a_color;\nvarying vec4 v_color;\n\n// Varying variables used for clipping in the fragment shader.\nvarying vec2 v_position;\nvarying vec4 v_ab;\n\nvoid main() {\n    float nrows = u_size.x;\n    float ncols = u_size.y;\n\n    // Compute the x coordinate from the time index.\n    float x = -1 + 2*a_index.z / (u_n-1);\n    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);\n\n    // Find the affine transformation for the subplots.\n    vec2 a = vec2(1./ncols, 1./nrows)*.9;\n    vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,\n                  -1 + 2*(a_index.y+.5) / nrows);\n    // Apply the static subplot transformation + scaling.\n    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);\n\n    v_color = vec4(a_color, 1.);\n    v_index = a_index;\n\n    // For clipping test in the fragment shader.\n    v_position = gl_Position.xy;\n    v_ab = vec4(a, b);\n}\n'
FRAG_SHADER = '\n#version 120\n\nvarying vec4 v_color;\nvarying vec3 v_index;\n\nvarying vec2 v_position;\nvarying vec4 v_ab;\n\nvoid main() {\n    gl_FragColor = v_color;\n\n    // Discard the fragments between the signals (emulate glMultiDrawArrays).\n    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))\n        discard;\n\n    // Clipping test.\n    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);\n    if ((test.x > 1) || (test.y > 1))\n        discard;\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            return 10
        app.Canvas.__init__(self, title='Use your wheel to zoom!', keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1.0, 1.0)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = n
        gloo.set_viewport(0, 0, *self.physical_size)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.show()

    def on_resize(self, event):
        if False:
            while True:
                i = 10
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_mouse_wheel(self, event):
        if False:
            while True:
                i = 10
        dx = np.sign(event.delta[1]) * 0.05
        (scale_x, scale_y) = self.program['u_scale']
        (scale_x_new, scale_y_new) = (scale_x * math.exp(2.5 * dx), scale_y * math.exp(0.0 * dx))
        self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
        self.update()

    def on_timer(self, event):
        if False:
            print('Hello World!')
        'Add some data at the end of each signal (real-time signals).'
        k = 10
        y[:, :-k] = y[:, k:]
        y[:, -k:] = amplitudes * np.random.randn(m, k)
        self.program['a_position'].set_data(y.ravel().astype(np.float32))
        self.update()
        self.context.flush()

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear()
        self.program.draw('line_strip')
if __name__ == '__main__':
    c = Canvas()
    app.run()