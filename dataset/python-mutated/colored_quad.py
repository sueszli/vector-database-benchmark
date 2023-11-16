"""
Show color quad Image
=====================

Create a new drawing using triangle_strip

"""
from vispy import app, gloo
from vispy.gloo import Program
vertex = '\n    attribute vec4 color;\n    attribute vec2 position;\n    varying vec4 v_color;\n    void main()\n    {\n        gl_Position = vec4(position, 0.0, 1.0);\n        v_color = color;\n    } '
fragment = '\n    varying vec4 v_color;\n    void main()\n    {\n        gl_FragColor = v_color;\n    } '

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(size=(512, 512), title='Colored quad', keys='interactive')
        self.program = Program(vertex, fragment, count=4)
        self.program['color'] = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
        self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        gloo.set_viewport(0, 0, *self.physical_size)
        self.show()

    def on_draw(self, event):
        if False:
            for i in range(10):
                print('nop')
        gloo.clear()
        self.program.draw('triangle_strip')

    def on_resize(self, event):
        if False:
            return 10
        gloo.set_viewport(0, 0, *event.physical_size)
c = Canvas()
if __name__ == '__main__':
    app.run()