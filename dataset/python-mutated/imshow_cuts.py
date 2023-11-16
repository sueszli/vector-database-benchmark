"""
Show an image using gloo, with on-mouseover cross-section visualizations.
"""
import numpy as np
from vispy import app
from vispy.gloo import set_viewport, clear, set_state, Program

def func(x, y):
    if False:
        return 10
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
x = np.linspace(-3.0, 3.0, 512).astype(np.float32)
y = np.linspace(-3.0, 3.0, 512).astype(np.float32)
(X, Y) = np.meshgrid(x, y)
idxs = func(X, Y)
(vmin, vmax) = (idxs.min(), idxs.max())
idxs = (idxs - vmin) / (vmax - vmin)
colormaps = np.ones((16, 512, 4)).astype(np.float32)
values = np.linspace(0, 1, 512)[1:-1]
colormaps[0, 0] = (0, 0, 1, 1)
colormaps[0, -1] = (0, 1, 0, 1)
colormaps[0, 1:-1, 0] = np.interp(values, [0.0, 0.33, 0.66, 1.0], [0.0, 1.0, 1.0, 1.0])
colormaps[0, 1:-1, 1] = np.interp(values, [0.0, 0.33, 0.66, 1.0], [0.0, 0.0, 1.0, 1.0])
colormaps[0, 1:-1, 2] = np.interp(values, [0.0, 0.33, 0.66, 1.0], [0.0, 0.0, 0.0, 1.0])
colormaps[1, 0] = (0, 0, 1, 1)
colormaps[1, -1] = (0, 1, 0, 1)
colormaps[1, 1:-1, 0] = np.interp(values, [0.0, 1.0], [0.0, 1.0])
colormaps[1, 1:-1, 1] = np.interp(values, [0.0, 1.0], [0.0, 1.0])
colormaps[1, 1:-1, 2] = np.interp(values, [0.0, 1.0], [0.0, 1.0])
lines_vertex = '\nattribute vec2 position;\nattribute vec4 color;\nvarying vec4 v_color;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0 );\n    v_color = color;\n}\n'
lines_fragment = '\nvarying vec4 v_color;\nvoid main()\n{\n    gl_FragColor = v_color;\n}\n'
image_vertex = '\nattribute vec2 position;\nattribute vec2 texcoord;\n\nvarying vec2 v_texcoord;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0 );\n    v_texcoord = texcoord;\n}\n'
image_fragment = '\nuniform float vmin;\nuniform float vmax;\nuniform float cmap;\nuniform float n_colormaps;\n\nuniform sampler2D image;\nuniform sampler2D colormaps;\n\nvarying vec2 v_texcoord;\nvoid main()\n{\n    float value = texture2D(image, v_texcoord).r;\n    float index = (cmap+0.5) / n_colormaps;\n\n    if( value < vmin ) {\n        gl_FragColor = texture2D(colormaps, vec2(0.0,index));\n    } else if( value > vmax ) {\n        gl_FragColor = texture2D(colormaps, vec2(1.0,index));\n    } else {\n        value = (value-vmin)/(vmax-vmin);\n        value = 1.0/512.0 + 510.0/512.0*value;\n        gl_FragColor = texture2D(colormaps, vec2(value,index));\n    }\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        app.Canvas.__init__(self, size=(512, 512), keys='interactive')
        self.image = Program(image_vertex, image_fragment, 4)
        self.image['position'] = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))
        self.image['texcoord'] = ((0, 0), (0, +1), (+1, 0), (+1, +1))
        self.image['vmin'] = +0.0
        self.image['vmax'] = +1.0
        self.image['cmap'] = 0
        self.image['colormaps'] = colormaps
        self.image['n_colormaps'] = colormaps.shape[0]
        self.image['image'] = idxs.astype('float32')
        self.image['image'].interpolation = 'linear'
        set_viewport(0, 0, *self.physical_size)
        self.lines = Program(lines_vertex, lines_fragment)
        self.lines['position'] = np.zeros((4 + 4 + 514 + 514, 2), np.float32)
        color = np.zeros((4 + 4 + 514 + 514, 4), np.float32)
        color[1:1 + 2, 3] = 0.25
        color[5:5 + 2, 3] = 0.25
        color[9:9 + 512, 3] = 0.5
        color[523:523 + 512, 3] = 0.5
        self.lines['color'] = color
        set_state(clear_color='white', blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.show()

    def on_resize(self, event):
        if False:
            i = 10
            return i + 15
        set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        clear(color=True, depth=True)
        self.image.draw('triangle_strip')
        self.lines.draw('line_strip')

    def on_mouse_move(self, event):
        if False:
            print('Hello World!')
        (x, y) = event.pos
        (w, h) = self.size
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        yf = 1 - y / (h / 2.0)
        xf = x / (w / 2.0) - 1
        x_norm = int(x * 512 // w)
        y_norm = int(y * 512 // h)
        P = np.zeros((4 + 4 + 514 + 514, 2), np.float32)
        x_baseline = P[:4]
        y_baseline = P[4:8]
        x_profile = P[8:522]
        y_profile = P[522:]
        x_baseline[...] = ((-1, yf), (-1, yf), (1, yf), (1, yf))
        y_baseline[...] = ((xf, -1), (xf, -1), (xf, 1), (xf, 1))
        x_profile[1:-1, 0] = np.linspace(-1, 1, 512)
        x_profile[1:-1, 1] = yf + 0.15 * idxs[y_norm, :]
        x_profile[0] = x_profile[1]
        x_profile[-1] = x_profile[-2]
        y_profile[1:-1, 0] = xf + 0.15 * idxs[:, x_norm]
        y_profile[1:-1, 1] = np.linspace(-1, 1, 512)
        y_profile[0] = y_profile[1]
        y_profile[-1] = y_profile[-2]
        self.lines['position'] = P
        self.update()
if __name__ == '__main__':
    canvas = Canvas()
    app.run()