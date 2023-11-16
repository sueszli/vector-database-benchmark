"""Computing a Voronoi diagram on the GPU. Shows how to use uniform arrays.

Original version by Xavier Olive (xoolive).

"""
import numpy as np
from vispy import app
from vispy import gloo
VS_voronoi = '\nattribute vec2 a_position;\n\nvoid main() {\n    gl_Position = vec4(a_position, 0., 1.);\n}\n'
FS_voronoi = '\nuniform vec2 u_seeds[32];\nuniform vec3 u_colors[32];\nuniform vec2 u_screen;\n\nvoid main() {\n    float dist = distance(u_screen * u_seeds[0], gl_FragCoord.xy);\n    vec3 color = u_colors[0];\n    for (int i = 1; i < 32; i++) {\n        float current = distance(u_screen * u_seeds[i], gl_FragCoord.xy);\n        if (current < dist) {\n            color = u_colors[i];\n            dist = current;\n        }\n    }\n    gl_FragColor = vec4(color, 1.0);\n}\n'
VS_seeds = '\nattribute vec2 a_position;\nuniform float u_ps;\n\nvoid main() {\n    gl_Position = vec4(2. * a_position - 1., 0., 1.);\n    gl_PointSize = 10. * u_ps;\n}\n'
FS_seeds = '\nvarying vec3 v_color;\nvoid main() {\n    gl_FragColor = vec4(1., 1., 1., 1.);\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            return 10
        app.Canvas.__init__(self, size=(600, 600), title='Voronoi diagram', keys='interactive')
        self.ps = self.pixel_scale
        self.seeds = np.random.uniform(0, 1.0 * self.ps, size=(32, 2)).astype(np.float32)
        self.colors = np.random.uniform(0.3, 0.8, size=(32, 3)).astype(np.float32)
        self.idx = 0
        self.program_v = gloo.Program(VS_voronoi, FS_voronoi)
        self.program_v['a_position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        for i in range(32):
            self.program_v['u_seeds[%d]' % i] = self.seeds[i, :]
            self.program_v['u_colors[%d]' % i] = self.colors[i, :]
        self.program_s = gloo.Program(VS_seeds, FS_seeds)
        self.program_s['a_position'] = self.seeds
        self.program_s['u_ps'] = self.ps
        self.activate_zoom()
        self.show()

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear()
        self.program_v.draw('triangle_strip')
        self.program_s.draw('points')

    def on_resize(self, event):
        if False:
            i = 10
            return i + 15
        self.activate_zoom()

    def activate_zoom(self):
        if False:
            print('Hello World!')
        (self.width, self.height) = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program_v['u_screen'] = self.physical_size

    def on_mouse_move(self, event):
        if False:
            while True:
                i = 10
        (x, y) = event.pos
        (x, y) = (x / float(self.width), 1 - y / float(self.height))
        self.program_v['u_seeds[%d]' % self.idx] = (x * self.ps, y * self.ps)
        self.seeds[self.idx, :] = (x, y)
        self.program_s['a_position'].set_data(self.seeds)
        self.update()

    def on_mouse_press(self, event):
        if False:
            return 10
        self.idx = (self.idx + 1) % 32
if __name__ == '__main__':
    c = Canvas()
    app.run()