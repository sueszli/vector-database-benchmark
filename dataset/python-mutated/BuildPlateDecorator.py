from UM.Scene.SceneNodeDecorator import SceneNodeDecorator
from cura.Scene.CuraSceneNode import CuraSceneNode

class BuildPlateDecorator(SceneNodeDecorator):
    """Make a SceneNode build plate aware CuraSceneNode objects all have this decorator."""

    def __init__(self, build_plate_number: int=-1) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._build_plate_number = build_plate_number
        self.setBuildPlateNumber(build_plate_number)

    def setBuildPlateNumber(self, nr: int) -> None:
        if False:
            while True:
                i = 10
        self._build_plate_number = nr
        if isinstance(self._node, CuraSceneNode):
            self._node.transformChanged()
        if self._node:
            for child in self._node.getChildren():
                child.callDecoration('setBuildPlateNumber', nr)

    def getBuildPlateNumber(self) -> int:
        if False:
            return 10
        return self._build_plate_number

    def __deepcopy__(self, memo):
        if False:
            print('Hello World!')
        return BuildPlateDecorator()