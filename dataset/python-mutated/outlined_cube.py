"""
Show a rotating cube with an outline
====================================
"""
import numpy as np
from vispy import gloo, app
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import create_cube
vertex = '\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_projection;\nuniform vec4 u_color;\n\nattribute vec3 position;\nattribute vec2 texcoord;\nattribute vec3 normal;\nattribute vec4 color;\n\nvarying vec4 v_color;\nvoid main()\n{\n    v_color = u_color * color;\n    gl_Position = u_projection * u_view * u_model * vec4(position,1.0);\n}\n'
fragment = '\nvarying vec4 v_color;\nvoid main()\n{\n    gl_FragColor = v_color;\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        app.Canvas.__init__(self, size=(512, 512), title='Rotating cube', keys='interactive')
        self.timer = app.Timer('auto', self.on_timer)
        (V, I, outline) = create_cube()
        vertices = VertexBuffer(V)
        self.faces = IndexBuffer(I)
        self.outline = IndexBuffer(outline)
        self.program = Program(vertex, fragment)
        self.program.bind(vertices)
        view = translate((0, 0, -5))
        model = np.eye(4, dtype=np.float32)
        self.program['u_model'] = model
        self.program['u_view'] = view
        (self.phi, self.theta) = (0, 0)
        self.activate_zoom()
        gloo.set_state(clear_color=(0.3, 0.3, 0.35, 1.0), depth_test=True, polygon_offset=(1, 1), line_width=0.75, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.timer.start()
        self.show()

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        gloo.clear(color=True, depth=True)
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program['u_color'] = (1, 1, 1, 1)
        self.program.draw('triangles', self.faces)
        gloo.set_state(blend=True, depth_mask=False, polygon_offset_fill=False)
        self.program['u_color'] = (0, 0, 0, 1)
        self.program.draw('lines', self.outline)
        gloo.set_state(depth_mask=True)

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
        projection = perspective(45.0, self.size[0] / float(self.size[1]), 2.0, 10.0)
        self.program['u_projection'] = projection

    def on_timer(self, event):
        if False:
            return 10
        self.theta += 0.5
        self.phi += 0.5
        self.program['u_model'] = np.dot(rotate(self.theta, (0, 0, 1)), rotate(self.phi, (0, 1, 0)))
        self.update()
if __name__ == '__main__':
    c = Canvas()
    app.run()