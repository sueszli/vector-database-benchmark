import numpy as np
from vispy import gloo, app
from vispy.gloo import Program, VertexBuffer
from vispy.util.transforms import ortho
vertex = '\n#version 120\n\nuniform mat4  u_model;\nuniform mat4  u_view;\nuniform mat4  u_projection;\nuniform float u_linewidth;\nuniform float u_antialias;\n\nattribute vec3  a_position;\nattribute vec4  a_fg_color;\nattribute float a_size;\n\nvarying vec4  v_fg_color;\nvarying float v_size;\n\nvoid main (void)\n{\n    v_size = a_size;\n    v_fg_color = a_fg_color;\n    if( a_fg_color.a > 0.0)\n    {\n        gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);\n        gl_PointSize = v_size + u_linewidth + 2.*1.5*u_antialias;\n    }\n    else\n    {\n        gl_Position = u_projection * u_view * u_model * vec4(-1.,-1.,0.,1.);\n        gl_PointSize = 0.0;\n    }\n}\n'
fragment = '\n#version 120\n\nuniform float u_linewidth;\nuniform float u_antialias;\nvarying vec4  v_fg_color;\nvarying vec4  v_bg_color;\nvarying float v_size;\nfloat disc(vec2 P, float size)\n{\n    return length((P.xy - vec2(0.5,0.5))*size);\n}\nvoid main()\n{\n    if( v_fg_color.a <= 0.0)\n        discard;\n    float actual_size = v_size + u_linewidth + 2*1.5*u_antialias;\n    float t = u_linewidth/2.0 - u_antialias;\n    float r = disc(gl_PointCoord, actual_size);\n    float d = abs(r - v_size/2.0) - t;\n    if( d < 0.0 )\n    {\n         gl_FragColor = v_fg_color;\n    }\n    else if( abs(d) > 2.5*u_antialias )\n    {\n         discard;\n    }\n    else\n    {\n        d /= u_antialias;\n        gl_FragColor = vec4(v_fg_color.rgb, exp(-d*d)*v_fg_color.a);\n    }\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, title='Rain [Move mouse]', size=(512, 512), keys='interactive')
        n = 500
        self.data = np.zeros(n, [('a_position', np.float32, 2), ('a_fg_color', np.float32, 4), ('a_size', np.float32)])
        self.index = 0
        self.program = Program(vertex, fragment)
        self.vdata = VertexBuffer(self.data)
        self.program.bind(self.vdata)
        self.program['u_antialias'] = 1.0
        self.program['u_linewidth'] = 1.0
        self.program['u_model'] = np.eye(4, dtype=np.float32)
        self.program['u_view'] = np.eye(4, dtype=np.float32)
        self.activate_zoom()
        gloo.set_clear_color('white')
        gloo.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.timer = app.Timer('auto', self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        gloo.clear()
        self.program.draw('points')

    def on_resize(self, event):
        if False:
            while True:
                i = 10
        self.activate_zoom()

    def activate_zoom(self):
        if False:
            for i in range(10):
                print('nop')
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = ortho(0, self.size[0], 0, self.size[1], -1, +1)
        self.program['u_projection'] = projection

    def on_timer(self, event):
        if False:
            print('Hello World!')
        self.data['a_fg_color'][..., 3] -= 0.01
        self.data['a_size'] += 1.0
        self.vdata.set_data(self.data)
        self.update()

    def on_mouse_move(self, event):
        if False:
            return 10
        (x, y) = event.pos
        h = self.size[1]
        self.data['a_position'][self.index] = (x, h - y)
        self.data['a_size'][self.index] = 5
        self.data['a_fg_color'][self.index] = (0, 0, 0, 1)
        self.index = (self.index + 1) % 500
if __name__ == '__main__':
    canvas = Canvas()
    app.run()