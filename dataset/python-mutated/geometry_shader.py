"""
Use a Geometry Shader
=====================

Simple geometry shader: Takes one point as input emits one triangle as output.

NOTE: This example is currently not processed in CI.

"""
import numpy as np
from vispy import gloo
from vispy import app
gloo.gl.use_gl('gl+')
position = np.random.normal(loc=0, scale=0.3, size=(1000, 2)).astype('float32')
VERT_SHADER = '\n#version 330\n\nin vec2 a_position;\n\nvoid main (void) {\n    gl_Position = vec4(a_position, 0, 1);\n    gl_PointSize = 3.0;\n}\n'
GEOM_SHADER = '\n#version 330\n\nlayout (points) in;\nlayout (triangle_strip, max_vertices=3) out;\n\nvoid main(void) {\n    vec4 p = gl_in[0].gl_Position;\n    \n    gl_Position = p;\n    EmitVertex();\n    gl_Position = p + vec4(0.06, 0.03, 0, 0);\n    EmitVertex();\n    gl_Position = p + vec4(0.03, 0.06, 0, 0);\n    EmitVertex();\n    EndPrimitive();\n}\n'
FRAG_SHADER = '\n#version 330\n\nout vec4 frag_color;\n\nvoid main()\n{\n    frag_color = vec4(0,0,0,0.5);\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, keys='interactive', size=(400, 400))
        self.program = gloo.Program()
        self.program.set_shaders(vert=VERT_SHADER, geom=GEOM_SHADER, frag=FRAG_SHADER)
        self.program['a_position'] = gloo.VertexBuffer(position)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_clear_color('white')
        self.context.set_state('translucent', cull_face=False, depth_test=False)
        self.show()

    def on_resize(self, event):
        if False:
            return 10
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])

    def on_draw(self, event):
        if False:
            return 10
        self.context.clear()
        self.program.draw('points')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()