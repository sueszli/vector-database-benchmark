from .base_filter import Filter
from ..transforms import NullTransform
from ...geometry import Rect

class Clipper(Filter):
    """Clips visual output to a rectangular region."""
    FRAG_SHADER = '\n        void clip() {\n            vec4 pos = $fb_to_clip(gl_FragCoord);\n            if( pos.x < $view.x || pos.x > $view.y ||\n                pos.y < $view.z || pos.y > $view.w ) {\n                discard;\n            }\n        }\n    '

    def __init__(self, bounds=(0, 0, 1, 1), transform=None):
        if False:
            return 10
        super(Clipper, self).__init__(fcode=self.FRAG_SHADER, fhook='pre', fpos=1)
        self.bounds = bounds
        if transform is None:
            transform = NullTransform()
        self._transform = None
        self.transform = transform

    @property
    def bounds(self):
        if False:
            return 10
        'The clipping boundaries.\n\n        This must be a tuple (x, y, w, h) in a clipping coordinate system\n        that is defined by the `transform` property.\n        '
        return self._bounds

    @bounds.setter
    def bounds(self, b):
        if False:
            while True:
                i = 10
        self._bounds = Rect(b).normalized()
        b = self._bounds
        self.fshader['view'] = (b.left, b.right, b.bottom, b.top)

    @property
    def transform(self):
        if False:
            return 10
        'The transform that maps from framebuffer coordinates to clipping\n        coordinates.\n        '
        return self._transform

    @transform.setter
    def transform(self, tr):
        if False:
            print('Hello World!')
        if tr is self._transform:
            return
        self._transform = tr
        self.fshader['fb_to_clip'] = tr