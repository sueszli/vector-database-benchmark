"""
Show vector field flow
"""
from __future__ import division
from vispy import app, scene, visuals, gloo
from vispy.util import ptime
import numpy as np

class VectorFieldVisual(visuals.Visual):
    vertex = '\n    uniform sampler2D field;\n    attribute vec2 index;\n    uniform vec2 shape;\n    uniform vec2 field_shape;\n    uniform float spacing;\n    varying float dist;  // distance along path for this vertex\n    varying vec2 ij;\n    uniform sampler2D offset;\n    uniform float seg_len;\n    uniform int n_iter;  // iterations to integrate along field per vertex\n    uniform vec2 attractor;\n    varying vec4 base_color;\n    uniform sampler2D color;\n    \n    void main() {\n        // distance along one line\n        dist = index.y * seg_len;\n        \n        vec2 local;\n        ij = vec2(mod(index.x, shape.x), floor(index.x / shape.x));\n        // *off* is a random offset to the starting location, which prevents\n        // the appearance of combs in the field \n        vec2 off = texture2D(offset, ij / shape).xy - 0.5;\n        local = spacing * (ij + off);\n        vec2 uv;\n        vec2 dir;\n        vec2 da;\n        int index_y = int(index.y);\n        for( int i=0; i<index.y; i+=1 ) {\n            for ( int j=0; j<n_iter; j += 1 ) {\n                uv = local / field_shape;\n                dir = texture2D(field, uv).xy;\n                \n                // add influence of variable attractor (mouse)\n                da = attractor - local;\n                float al = 0.1 * length(da);\n                da /= 0.5 * (1 + al*al);\n                \n                dir += da;\n                \n                // maybe pick a more accurate integration method?\n                local += seg_len * dir / n_iter;\n            }\n        }\n        base_color = texture2D(color, uv);\n        \n        gl_Position = $transform(vec4(local, 0, 1));\n    }\n    '
    fragment = '\n    uniform float time;\n    uniform float speed;\n    varying float dist;\n    varying vec2 ij;\n    uniform sampler2D offset;\n    uniform vec2 shape;\n    uniform float nseg;\n    uniform float seg_len;\n    varying vec4 base_color;\n    \n    void main() {\n        float totlen = nseg * seg_len;\n        float phase = texture2D(offset, ij / shape).b;\n        float alpha;\n        \n        // vary alpha along the length of the line to give the appearance of\n        // motion\n        alpha = mod((dist / totlen) + phase - time * speed, 1);\n        \n        // add a cosine envelope to fade in and out smoothly at the ends\n        alpha *= (1 - cos(2 * 3.141592 * dist / totlen)) * 0.5;\n        \n        gl_FragColor = vec4(base_color.rgb, base_color.a * alpha);\n    }\n    '

    def __init__(self, field, spacing=10, segments=3, seg_len=0.5, color=(1, 1, 1, 0.3)):
        if False:
            for i in range(10):
                print('nop')
        self._time = 0.0
        self._last_time = ptime.time()
        rows = int(field.shape[0] / spacing)
        cols = int(field.shape[1] / spacing)
        index = np.empty((rows * cols, int(segments) * 2, 2), dtype=np.float32)
        index[:, :, 0] = np.arange(rows * cols)[:, np.newaxis]
        index[:, ::2, 1] = np.arange(segments)[np.newaxis, :]
        index[:, 1::2, 1] = np.arange(segments)[np.newaxis, :] + 1
        self._index = gloo.VertexBuffer(index)
        if not isinstance(color, np.ndarray):
            color = np.array([[list(color)]], dtype='float32')
        self._color = gloo.Texture2D(color)
        offset = np.random.uniform(256, size=(rows, cols, 3)).astype(np.ubyte)
        self._offset = gloo.Texture2D(offset, format='rgb')
        self._field = gloo.Texture2D(field, format='rg', internalformat='rg32f', interpolation='linear')
        self._field_shape = field.shape[:2]
        visuals.Visual.__init__(self, vcode=self.vertex, fcode=self.fragment)
        self.timer = app.Timer(interval='auto', connect=self.update_time, start=False)
        self.freeze()
        self.shared_program['field'] = self._field
        self.shared_program['field_shape'] = self._field.shape[:2]
        self.shared_program['shape'] = (rows, cols)
        self.shared_program['index'] = self._index
        self.shared_program['spacing'] = spacing
        self.shared_program['t'] = self._time
        self.shared_program['offset'] = self._offset
        self.shared_program['speed'] = 1
        self.shared_program['color'] = self._color
        self.shared_program['seg_len'] = seg_len
        self.shared_program['nseg'] = segments
        self.shared_program['n_iter'] = 1
        self.shared_program['attractor'] = (0, 0)
        self.shared_program['time'] = 0
        self._draw_mode = 'lines'
        self.set_gl_state('translucent', depth_test=False)
        self.timer.start()

    def _prepare_transforms(self, view):
        if False:
            i = 10
            return i + 15
        view.view_program.vert['transform'] = view.get_transform()

    def _prepare_draw(self, view):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _compute_bounds(self, axis, view):
        if False:
            while True:
                i = 10
        if axis > 1:
            return (0, 0)
        return (0, self._field_shape[axis])

    def update_time(self, ev):
        if False:
            while True:
                i = 10
        t = ptime.time()
        self._time += t - self._last_time
        self._last_time = t
        self.shared_program['time'] = self._time
        self.update()
VectorField = scene.visuals.create_visual_node(VectorFieldVisual)

def fn(y, x):
    if False:
        i = 10
        return i + 15
    dx = x - 50
    dy = y - 30
    hyp = (dx ** 2 + dy ** 2) ** 0.5 + 0.01
    return np.array([100 * dy / hyp ** 1.7, -100 * dx / hyp ** 1.8])
field = np.fromfunction(fn, (100, 100)).transpose(1, 2, 0).astype('float32')
field[..., 0] += 10 * np.cos(np.linspace(0, 2 * 3.1415, 100))
color = np.zeros((100, 100, 4), dtype='float32')
color[..., :2] = (field + 5) / 10.0
color[..., 2] = 0.5
color[..., 3] = 0.5
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view(camera='panzoom')
vfield = VectorField(field[..., :2], spacing=0.5, segments=30, seg_len=0.05, parent=view.scene, color=color)
view.camera.set_range()

@canvas.connect
def on_mouse_move(event):
    if False:
        print('Hello World!')
    if 3 in event.buttons:
        tr = canvas.scene.node_transform(vfield)
        vfield.shared_program['attractor'] = tr.map(event.pos)[:2]
if __name__ == '__main__':
    app.run()