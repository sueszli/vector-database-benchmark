from UM.Scene.SceneNodeDecorator import SceneNodeDecorator

class SliceableObjectDecorator(SceneNodeDecorator):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()

    def isSliceable(self) -> bool:
        if False:
            return 10
        return True

    def __deepcopy__(self, memo) -> 'SliceableObjectDecorator':
        if False:
            i = 10
            return i + 15
        return type(self)()