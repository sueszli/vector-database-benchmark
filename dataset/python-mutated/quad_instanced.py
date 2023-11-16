"""
Displaying quads using Instanced rendering
==========================================

This example is a modification of examples/tutorial/gl/quad.py which
uses instanced rendering to generate many copies of the same quad.
"""
import numpy as np
from vispy import app, use
from vispy.gloo import gl
use(gl='gl+')
vertex_code = '\n    uniform float scale;\n    attribute vec4 color;\n    attribute vec2 position;\n    attribute vec2 instance_offset;\n    varying vec4 v_color;\n    void main()\n    {\n        gl_Position = vec4(scale*position + instance_offset, 0.0, 1.0);\n        v_color = color;\n    } '
fragment_code = '\n    varying vec4 v_color;\n    void main()\n    {\n        gl_FragColor = v_color;\n    } '

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            while True:
                i = 10
        app.Canvas.__init__(self, size=(512, 512), title='Quad (GL)', keys='interactive')

    def on_initialize(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.data = np.zeros(4, [('position', np.float32, 2), ('color', np.float32, 4)])
        self.data['color'] = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
        self.data['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.n_instances = 1000
        self.instances = np.empty(self.n_instances, [('instance_offset', np.float32, 2)])
        self.instances['instance_offset'] = (np.random.rand(self.n_instances, 2) - 0.5) * 2
        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(vertex, vertex_code)
        gl.glShaderSource(fragment, fragment_code)
        gl.glCompileShader(vertex)
        gl.glCompileShader(fragment)
        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)
        gl.glDetachShader(program, vertex)
        gl.glDetachShader(program, fragment)
        gl.glUseProgram(program)
        buf = gl.glCreateBuffer()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.data, gl.GL_DYNAMIC_DRAW)
        stride = self.data.strides[0]
        instance_offset = 0
        loc = gl.glGetAttribLocation(program, 'position')
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, False, stride, instance_offset)
        instance_offset = self.data.dtype['position'].itemsize
        loc = gl.glGetAttribLocation(program, 'color')
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, False, stride, instance_offset)
        buf = gl.glCreateBuffer()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.instances, gl.GL_STATIC_DRAW)
        stride = self.instances.strides[0]
        instance_offset = 0
        loc = gl.glGetAttribLocation(program, 'instance_offset')
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, stride, instance_offset)
        gl.glVertexAttribDivisor(loc, 1)
        loc = gl.glGetUniformLocation(program, 'scale')
        gl.glUniform1f(loc, 0.01)

    def on_draw(self, event):
        if False:
            print('Hello World!')
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, self.n_instances)

    def on_resize(self, event):
        if False:
            return 10
        gl.glViewport(0, 0, *event.physical_size)
if __name__ == '__main__':
    c = Canvas()
    c.show()
    app.run()