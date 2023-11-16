from UM.Scene.SceneNodeDecorator import SceneNodeDecorator

class ZOffsetDecorator(SceneNodeDecorator):
    """A decorator that stores the amount an object has been moved below the platform."""

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self._z_offset = 0.0

    def setZOffset(self, offset: float) -> None:
        if False:
            return 10
        self._z_offset = offset

    def getZOffset(self) -> float:
        if False:
            print('Hello World!')
        return self._z_offset

    def __deepcopy__(self, memo) -> 'ZOffsetDecorator':
        if False:
            while True:
                i = 10
        copied_decorator = ZOffsetDecorator()
        copied_decorator.setZOffset(self.getZOffset())
        return copied_decorator