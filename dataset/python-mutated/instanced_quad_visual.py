"""
Custom Visual for instanced rendering of a colored quad
=======================================================

# this example is based on the tutorial: T01_basic_visual.py
"""
from vispy import app, gloo, visuals, scene, use
import numpy as np
use(gl='gl+')
vertex_shader = '\n// both these attributes will be defined on an instance basis (not per vertex)\nattribute vec2 shift;\nattribute vec4 color;\n\nvarying vec4 v_color;\nvoid main() {\n    v_color = color;\n    gl_Position = $transform(vec4($position + shift, 0, 1));\n}\n'
fragment_shader = '\nvarying vec4 v_color;\n\nvoid main() {\n  gl_FragColor = v_color;\n}\n'

class InstancedRectVisual(visuals.Visual):

    def __init__(self, x, y, w, h):
        if False:
            while True:
                i = 10
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)
        self.vbo = gloo.VertexBuffer(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], dtype=np.float32))
        self.shared_program.vert['position'] = self.vbo
        self._draw_mode = 'triangles'
        self.shifts = gloo.VertexBuffer(np.random.rand(100, 2).astype(np.float32) * 500, divisor=1)
        self.shared_program['shift'] = self.shifts
        self.color = gloo.VertexBuffer(np.random.rand(20, 4).astype(np.float32), divisor=5)
        self.shared_program['color'] = self.color

    def _prepare_transforms(self, view):
        if False:
            for i in range(10):
                print('nop')
        view.view_program.vert['transform'] = view.get_transform()
InstancedRect = scene.visuals.create_visual_node(InstancedRectVisual)
canvas = scene.SceneCanvas(keys='interactive', show=True)
rect = InstancedRect(0, 0, 20, 40, parent=canvas.scene)
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        app.run()