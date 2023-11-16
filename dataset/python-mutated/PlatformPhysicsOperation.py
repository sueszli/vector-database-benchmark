from UM.Math.Vector import Vector
from UM.Operations.Operation import Operation
from UM.Operations.GroupedOperation import GroupedOperation
from UM.Scene.SceneNode import SceneNode

class PlatformPhysicsOperation(Operation):
    """A specialised operation designed specifically to modify the previous operation."""

    def __init__(self, node: SceneNode, translation: Vector) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._node = node
        self._old_transformation = node.getLocalTransformation()
        self._translation = translation
        self._always_merge = True

    def undo(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._node.setTransformation(self._old_transformation)

    def redo(self) -> None:
        if False:
            i = 10
            return i + 15
        self._node.translate(self._translation, SceneNode.TransformSpace.World)

    def mergeWith(self, other: Operation) -> GroupedOperation:
        if False:
            return 10
        group = GroupedOperation()
        group.addOperation(other)
        group.addOperation(self)
        return group

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'PlatformPhysicsOp.(trans.={0})'.format(self._translation)