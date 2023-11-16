"""
Rotating Quad
=============

Use a Timer to animate a quad

"""
from vispy import gloo, app
from vispy.gloo import Program
vertex = '\n    uniform float theta;\n    attribute vec4 color;\n    attribute vec2 position;\n    varying vec4 v_color;\n    void main()\n    {\n        float ct = cos(theta);\n        float st = sin(theta);\n        float x = 0.75* (position.x*ct - position.y*st);\n        float y = 0.75* (position.x*st + position.y*ct);\n        gl_Position = vec4(x, y, 0.0, 1.0);\n        v_color = color;\n    } '
fragment = '\n    varying vec4 v_color;\n    void main()\n    {\n        gl_FragColor = v_color;\n    } '

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(size=(512, 512), title='Rotating quad', keys='interactive')
        self.program = Program(vertex, fragment, count=4)
        self.program['color'] = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
        self.program['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.program['theta'] = 0.0
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_clear_color('white')
        self.timer = app.Timer('auto', self.on_timer)
        self.clock = 0
        self.timer.start()
        self.show()

    def on_draw(self, event):
        if False:
            print('Hello World!')
        gloo.clear()
        self.program.draw('triangle_strip')

    def on_resize(self, event):
        if False:
            print('Hello World!')
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_timer(self, event):
        if False:
            return 10
        self.clock += 0.001 * 1000.0 / 60.0
        self.program['theta'] = self.clock
        self.update()
if __name__ == '__main__':
    c = Canvas()
    app.run()