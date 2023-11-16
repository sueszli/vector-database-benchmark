""" Visualization of traveling through space.
"""
import time
import numpy as np
from vispy import gloo
from vispy import app
from vispy.util.transforms import perspective
vertex = '\n#version 120\n\nuniform mat4 u_model;\nuniform mat4 u_view;\nuniform mat4 u_projection;\nuniform float u_time_offset;\nuniform float u_pixel_scale;\n\nattribute vec3  a_position;\nattribute float a_offset;\n\nvarying float v_pointsize;\n\nvoid main (void) {\n\n    vec3 pos = a_position;\n    pos.z = pos.z - a_offset - u_time_offset;\n    vec4 v_eye_position = u_view * u_model * vec4(pos, 1.0);\n    gl_Position = u_projection * v_eye_position;\n\n    // stackoverflow.com/questions/8608844/...\n    //  ... resizing-point-sprites-based-on-distance-from-the-camera\n    float radius = 1;\n    vec4 corner = vec4(radius, radius, v_eye_position.z, v_eye_position.w);\n    vec4  proj_corner = u_projection * corner;\n    gl_PointSize = 100.0 * u_pixel_scale * proj_corner.x / proj_corner.w;\n    v_pointsize = gl_PointSize;\n}\n'
fragment = '\n#version 120\nvarying float v_pointsize;\nvoid main()\n{\n    float x = 2.0*gl_PointCoord.x - 1.0;\n    float y = 2.0*gl_PointCoord.y - 1.0;\n    float a = 0.9 - (x*x + y*y);\n    a = a * min(1.0, v_pointsize/1.5);\n    gl_FragColor = vec4(1.0, 1.0, 1.0, a);\n}\n'
N = 100000
SIZE = 100
SPEED = 4.0
NBLOCKS = 10

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            print('Hello World!')
        app.Canvas.__init__(self, title='Spacy', keys='interactive', size=(800, 600))
        self.program = gloo.Program(vertex, fragment)
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.activate_zoom()
        self.timer = app.Timer('auto', connect=self.update, start=True)
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_pixel_scale'] = self.pixel_scale
        self.program['a_position'] = np.zeros((N, 3), np.float32)
        self.program['a_offset'] = np.zeros((N, 1), np.float32)
        self._timeout = 0
        self._active_block = 0
        for i in range(NBLOCKS):
            self._generate_stars()
        self._timeout = time.time() + SPEED
        gloo.set_state(clear_color='black', depth_test=False, blend=True, blend_equation='func_add', blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.show()

    def on_key_press(self, event):
        if False:
            i = 10
            return i + 15
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()

    def on_resize(self, event):
        if False:
            print('Hello World!')
        self.activate_zoom()

    def activate_zoom(self):
        if False:
            return 10
        (width, height) = self.size
        gloo.set_viewport(0, 0, *self.physical_size)
        far = SIZE * (NBLOCKS - 2)
        self.projection = perspective(25.0, width / float(height), 1.0, far)
        self.program['u_projection'] = self.projection

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        factor = (self._timeout - time.time()) / SPEED
        self.program['u_time_offset'] = -(1 - factor) * SIZE
        gloo.clear()
        self.program.draw('points')
        if factor < 0:
            self._generate_stars()

    def on_close(self, event):
        if False:
            print('Hello World!')
        self.timer.stop()

    def _generate_stars(self):
        if False:
            return 10
        blocksize = N // NBLOCKS
        self._active_block += 1
        if self._active_block >= NBLOCKS:
            self._active_block = 0
        pos = np.zeros((blocksize, 3), 'float32')
        pos[:, :2] = np.random.normal(0.0, SIZE / 2.0, (blocksize, 2))
        pos[:, 2] = np.random.uniform(0, SIZE, (blocksize,))
        start_index = self._active_block * blocksize
        self.program['a_position'].set_subdata(pos, offset=start_index)
        for i in range(NBLOCKS):
            val = i - self._active_block
            if val < 0:
                val += NBLOCKS
            values = np.ones((blocksize, 1), 'float32') * val * SIZE
            start_index = i * blocksize
            self.program['a_offset'].set_subdata(values, offset=start_index)
        self._timeout += SPEED
if __name__ == '__main__':
    c = Canvas()
    app.run()