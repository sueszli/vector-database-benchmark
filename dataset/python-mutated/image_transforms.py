"""
Simple demonstration of ImageVisual.
"""
import numpy as np
import vispy.app
from vispy import gloo
from vispy import visuals
from vispy.visuals.transforms import MatrixTransform, STTransform, arg_to_array, LogTransform, PolarTransform, BaseTransform
from image_visual import get_image

class Canvas(vispy.app.Canvas):

    def __init__(self):
        if False:
            return 10
        vispy.app.Canvas.__init__(self, keys='interactive', size=(800, 800))
        image = get_image()
        self.images = [visuals.ImageVisual(image, method='impostor') for i in range(4)]
        s = 100.0 / max(self.images[0].size)
        tx = 0.5 * (100 - self.images[0].size[0] * s)
        ty = 0.5 * (100 - self.images[0].size[1] * s)
        base_tr = STTransform(scale=(s, s), translate=(tx, ty))
        self.images[0].transform = STTransform(scale=(30, 30), translate=(600, 600)) * SineTransform() * STTransform(scale=(0.1, 0.1), translate=(-5, -5)) * base_tr
        tr = MatrixTransform()
        tr.rotate(40, (0, 0, 1))
        tr.rotate(30, (1, 0, 0))
        tr.translate((0, -20, -60))
        p = MatrixTransform()
        p.set_perspective(0.5, 1, 0.1, 1000)
        tr = p * tr
        tr1 = STTransform(translate=(200, 600)) * tr * STTransform(translate=(-50, -50)) * base_tr
        self.images[1].transform = tr1
        tr2 = STTransform(scale=(3, -100), translate=(200, 50)) * LogTransform((0, 2, 0)) * STTransform(scale=(1, -0.01), translate=(-50, 1.1)) * base_tr
        self.images[2].transform = tr2
        tr3 = STTransform(scale=(400, 400), translate=(570, 400)) * PolarTransform() * STTransform(scale=(np.pi / 150, -0.005), translate=(-3.3 * np.pi / 4.0, 0.7)) * base_tr
        self.images[3].transform = tr3
        text = visuals.TextVisual(text=['logarithmic', 'polar', 'perspective', 'custom (sine)'], pos=[(100, 20), (500, 20), (100, 410), (500, 410)], color='k', font_size=16)
        self.visuals = self.images + [text]
        self.show()

    def on_draw(self, ev):
        if False:
            i = 10
            return i + 15
        gloo.clear(color='w', depth=True)
        for vis in self.visuals:
            vis.draw()

    def on_resize(self, event):
        if False:
            for i in range(10):
                print('nop')
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        for vis in self.visuals:
            vis.transforms.configure(canvas=self, viewport=vp)

class SineTransform(BaseTransform):
    """
    Add sine wave to y-value for wavy effect.
    """
    glsl_map = '\n        vec4 sineTransform(vec4 pos) {\n            return vec4(pos.x, pos.y + sin(pos.x), pos.z, 1);\n        }'
    glsl_imap = '\n        vec4 sineTransform(vec4 pos) {\n            return vec4(pos.x, pos.y - sin(pos.x), pos.z, 1);\n        }'
    Linear = False

    @arg_to_array
    def map(self, coords):
        if False:
            for i in range(10):
                print('nop')
        ret = coords.copy()
        ret[..., 1] += np.sin(ret[..., 0])
        return ret

    @arg_to_array
    def imap(self, coords):
        if False:
            i = 10
            return i + 15
        ret = coords.copy()
        ret[..., 1] -= np.sin(ret[..., 0])
        return ret

    def inverse(self):
        if False:
            print('Hello World!')
        return InvSineTransform()

class InvSineTransform(BaseTransform):
    glsl_map = SineTransform.glsl_imap
    glsl_imap = SineTransform.glsl_map
    Linear = False
    map = SineTransform.imap
    imap = SineTransform.map

    def inverse(self):
        if False:
            return 10
        return SineTransform()
if __name__ == '__main__':
    win = Canvas()
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()