"""
Show a rotating colored cube
============================
"""
import numpy as np
from vispy import app, gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import create_cube
vertex = '\nuniform mat4 model;\nuniform mat4 view;\nuniform mat4 projection;\n\nattribute vec3 position;\nattribute vec2 texcoord;\nattribute vec3 normal;\nattribute vec4 color;\n\nvarying vec4 v_color;\nvoid main()\n{\n    v_color = color;\n    gl_Position = projection * view * model * vec4(position,1.0);\n}\n'
fragment = '\nvarying vec4 v_color;\nvoid main()\n{\n    gl_FragColor = v_color;\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, size=(512, 512), title='Colored cube', keys='interactive')
        (V, I, _) = create_cube()
        vertices = VertexBuffer(V)
        self.indices = IndexBuffer(I)
        self.program = Program(vertex, fragment)
        self.program.bind(vertices)
        view = translate((0, 0, -5))
        model = np.eye(4, dtype=np.float32)
        self.program['model'] = model
        self.program['view'] = view
        (self.phi, self.theta) = (0, 0)
        gloo.set_state(clear_color=(0.3, 0.3, 0.35, 1.0), depth_test=True)
        self.activate_zoom()
        self.timer = app.Timer('auto', self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        gloo.clear(color=True, depth=True)
        self.program.draw('triangles', self.indices)

    def on_resize(self, event):
        if False:
            i = 10
            return i + 15
        self.activate_zoom()

    def activate_zoom(self):
        if False:
            while True:
                i = 10
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]), 2.0, 10.0)
        self.program['projection'] = projection

    def on_timer(self, event):
        if False:
            return 10
        self.theta += 0.5
        self.phi += 0.5
        self.program['model'] = np.dot(rotate(self.theta, (0, 0, 1)), rotate(self.phi, (0, 1, 0)))
        self.update()
if __name__ == '__main__':
    c = Canvas()
    app.run()