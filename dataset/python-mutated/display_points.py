"""
Draw 2D points
==============

Simple example plotting 2D points.

"""
from vispy import gloo
from vispy import app
import numpy as np
VERT_SHADER = '\nattribute vec2  a_position;\nattribute vec3  a_color;\nattribute float a_size;\n\nvarying vec4 v_fg_color;\nvarying vec4 v_bg_color;\nvarying float v_radius;\nvarying float v_linewidth;\nvarying float v_antialias;\n\nvoid main (void) {\n    v_radius = a_size;\n    v_linewidth = 1.0;\n    v_antialias = 1.0;\n    v_fg_color  = vec4(0.0,0.0,0.0,0.5);\n    v_bg_color  = vec4(a_color,    1.0);\n\n    gl_Position = vec4(a_position, 0.0, 1.0);\n    gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);\n}\n'
FRAG_SHADER = '\n#version 120\n\nvarying vec4 v_fg_color;\nvarying vec4 v_bg_color;\nvarying float v_radius;\nvarying float v_linewidth;\nvarying float v_antialias;\nvoid main()\n{\n    float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);\n    float t = v_linewidth/2.0-v_antialias;\n    float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);\n    float d = abs(r - v_radius) - t;\n    if( d < 0.0 )\n        gl_FragColor = v_fg_color;\n    else\n    {\n        float alpha = d/v_antialias;\n        alpha = exp(-alpha*alpha);\n        if (r > v_radius)\n            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);\n        else\n            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);\n    }\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, keys='interactive')
        ps = self.pixel_scale
        n = 10000
        v_position = 0.25 * np.random.randn(n, 2).astype(np.float32)
        v_color = np.random.uniform(0, 1, (n, 3)).astype(np.float32)
        v_size = np.random.uniform(2 * ps, 12 * ps, (n, 1)).astype(np.float32)
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_color'] = gloo.VertexBuffer(v_color)
        self.program['a_position'] = gloo.VertexBuffer(v_position)
        self.program['a_size'] = gloo.VertexBuffer(v_size)
        gloo.set_state(clear_color='white', blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.show()

    def on_resize(self, event):
        if False:
            return 10
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear(color=True, depth=True)
        self.program.draw('points')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()