"""Using Colorbars with the Canvas with the Mandlebrot set"""
from vispy import app
from vispy import gloo
from vispy.visuals.transforms import STTransform
from vispy.visuals import ColorBarVisual, ImageVisual
from vispy.color import Color, get_colormap
import numpy as np
ESCAPE_MAGNITUDE = 2
MIN_MAGNITUDE = 0.002
MAX_ITERATIONS = 50
colormap = get_colormap('hot')

def get_num_escape_turns(x, y):
    if False:
        print('Hello World!')
    'Returns the number of iterations it took to escape\n       as normalized values.\n       Parameters\n       ----------\n\n       x: float\n        the x coordinates of the point\n\n       y: float\n        the y coordinates of the point\n\n       Returns\n       -------\n       float: [0, 1]\n       * 0 if it took 0 iterations to escape\n       * 1 if did not escape in MAX_ITERATIONS iterations\n       * a linearly interpolated number between 0 and 1 if the point took\n         anywhere between 0 to MAX_ITERATIONS to escape\n\n    '
    c = complex(x, y)
    z = complex(x, y)
    num_iterations = 0
    while MIN_MAGNITUDE < np.absolute(z) < ESCAPE_MAGNITUDE and num_iterations < MAX_ITERATIONS:
        z = z ** 2 + c
        num_iterations += 1
    return float(num_iterations) / float(MAX_ITERATIONS)

def get_mandlebrot_escape_values(width, height):
    if False:
        return 10
    'Constructs the Mandlebro set for a grid of dimensions (width, height)\n\n    Parameters\n    ----------\n    width: int\n        width of the resulting grid\n    height: int\n        height of the resulting grid\n\n    Returns\n    -------\n    A grid of floating point values containing the output of\n    get_num_escape_turns function for each point\n    '
    x_vals = np.linspace(-3, 2, width)
    y_vals = np.linspace(-1.5, 1.5, height)
    grid = np.meshgrid(x_vals, y_vals)
    v_get_num_escape_turns = np.vectorize(get_num_escape_turns)
    return v_get_num_escape_turns(*grid).astype(np.float32)

def get_vertical_bar(pos, size):
    if False:
        for i in range(10):
            print('nop')
    '\n    Constructs the vertical bar that represents the\n    color values for the Mandlebrot set\n\n    Returns\n    -------\n    A vispy.visual.ColorBarVisual object that represents the\n    data of the Mandlebrot set\n    '
    vertical = ColorBarVisual(pos=pos, size=size, label='iterations to escape', cmap=colormap, orientation='left')
    vertical.label.font_size = 15
    vertical.label.color = 'white'
    vertical.clim = (0, MAX_ITERATIONS)
    vertical.ticks[0].font_size = 10
    vertical.ticks[1].font_size = 10
    vertical.ticks[0].color = 'white'
    vertical.ticks[1].color = 'white'
    vertical.border_width = 1
    vertical.border_color = Color('#ababab')
    return vertical

class Canvas(app.Canvas):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        img_dim = np.array([700, 500])
        colorbar_pos = np.array([100, 300])
        colorbar_size = np.array([400, 20])
        image_pos = np.array([200, 80])
        app.Canvas.__init__(self, size=(800, 600), keys='interactive')
        img_data = get_mandlebrot_escape_values(img_dim[0], img_dim[1])
        self.image = ImageVisual(img_data, cmap=colormap)
        self.image.transform = STTransform(scale=1.1, translate=image_pos)
        self.vertical_bar = get_vertical_bar(colorbar_pos, colorbar_size)
        self.show()

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.image.transforms.configure(canvas=self, viewport=vp)
        self.vertical_bar.transforms.configure(canvas=self, viewport=vp)

    def on_draw(self, event):
        if False:
            while True:
                i = 10
        gloo.clear(color=colormap[0.0])
        self.image.draw()
        self.vertical_bar.draw()
if __name__ == '__main__':
    win = Canvas()
    app.run()