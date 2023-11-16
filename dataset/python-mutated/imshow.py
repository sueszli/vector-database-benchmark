"""
Show an image using gloo.
"""
import numpy as np
from vispy import app
from vispy.gloo import clear, set_clear_color, set_viewport, Program

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
img_vertex = '\nattribute vec2 position;\nattribute vec2 texcoord;\n\nvarying vec2 v_texcoord;\nvoid main()\n{\n    gl_Position = vec4(position, 0.0, 1.0 );\n    v_texcoord = texcoord;\n}\n'
img_fragment = '\nuniform float vmin;\nuniform float vmax;\nuniform float cmap;\n\nuniform sampler2D image;\nuniform sampler2D colormaps;\nuniform vec2 colormaps_shape;\n\nvarying vec2 v_texcoord;\nvoid main()\n{\n    float value = texture2D(image, v_texcoord).r;\n    float index = (cmap+0.5) / colormaps_shape.y;\n\n    if( value < vmin ) {\n        gl_FragColor = texture2D(colormaps, vec2(0.0,index));\n    } else if( value > vmax ) {\n        gl_FragColor = texture2D(colormaps, vec2(1.0,index));\n    } else {\n        value = (value-vmin)/(vmax-vmin);\n        value = 1.0/512.0 + 510.0/512.0*value;\n        gl_FragColor = texture2D(colormaps, vec2(value,index));\n    }\n}\n'

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        app.Canvas.__init__(self, size=(512, 512), keys='interactive')
        self.image = Program(img_vertex, img_fragment, 4)
        self.image['position'] = ((-1, -1), (-1, +1), (+1, -1), (+1, +1))
        self.image['texcoord'] = ((0, 0), (0, +1), (+1, 0), (+1, +1))
        self.image['vmin'] = +0.1
        self.image['vmax'] = +0.9
        self.image['cmap'] = 0
        self.image['colormaps'] = colormaps
        self.image['colormaps'].interpolation = 'linear'
        self.image['colormaps_shape'] = (colormaps.shape[1], colormaps.shape[0])
        self.image['image'] = idxs.astype('float32')
        self.image['image'].interpolation = 'linear'
        set_clear_color('black')
        self.show()

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        (width, height) = event.physical_size
        set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        if False:
            print('Hello World!')
        clear(color=True, depth=True)
        self.image.draw('triangle_strip')
if __name__ == '__main__':
    canvas = Canvas()
    app.run()