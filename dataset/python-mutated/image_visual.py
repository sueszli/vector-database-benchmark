"""
Simple demonstration of ImageVisual.
"""
import numpy as np
import vispy.app
from vispy import gloo
from vispy import visuals
from vispy.visuals.transforms import STTransform

class Canvas(vispy.app.Canvas):

    def __init__(self):
        if False:
            while True:
                i = 10
        vispy.app.Canvas.__init__(self, keys='interactive', size=(800, 800))
        self.image = visuals.ImageVisual(get_image(), method='subdivide')
        s = 700.0 / max(self.image.size)
        t = 0.5 * (700.0 - self.image.size[0] * s) + 50
        self.image.transform = STTransform(scale=(s, s), translate=(t, 50))
        self.show()

    def on_draw(self, ev):
        if False:
            i = 10
            return i + 15
        gloo.clear(color='black', depth=True)
        self.image.draw()

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        self.image.transforms.configure(canvas=self, viewport=vp)

def get_image():
    if False:
        i = 10
        return i + 15
    'Load an image from the demo-data repository if possible. Otherwise,\n    just return a randomly generated image.\n    '
    from vispy.io import load_data_file, read_png
    try:
        return read_png(load_data_file('mona_lisa/mona_lisa_sm.png'))
    except Exception as exc:
        print('Error loading demo image data: %r' % exc)
    image = np.random.normal(size=(100, 100, 3))
    image[20:80, 20:80] += 3.0
    image[50] += 3.0
    image[:, 50] += 3.0
    image = ((image - image.min()) * (253.0 / (image.max() - image.min()))).astype(np.ubyte)
    return image
if __name__ == '__main__':
    win = Canvas()
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()