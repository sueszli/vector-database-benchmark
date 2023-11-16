from .element import Element
from .transform import Transform

class Transformable(Element):
    """Transformable SVG element"""

    def __init__(self, content=None, parent=None):
        if False:
            return 10
        Element.__init__(self, content, parent)
        if isinstance(content, str):
            self._transform = Transform()
            self._computed_transform = self._transform
        else:
            self._transform = Transform(content.get('transform', None))
            self._computed_transform = self._transform
            if parent:
                self._computed_transform = self._transform + self.parent.transform

    @property
    def transform(self):
        if False:
            for i in range(10):
                print('nop')
        return self._computed_transform