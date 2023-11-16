from typing import Optional
from UM.Scene.SceneNode import SceneNode
from UM.Operations import Operation

class SetParentOperation(Operation.Operation):
    """An operation that parents a scene node to another scene node."""

    def __init__(self, node: SceneNode, parent_node: Optional[SceneNode]) -> None:
        if False:
            print('Hello World!')
        'Initialises this SetParentOperation.\n\n        :param node: The node which will be reparented.\n        :param parent_node: The node which will be the parent.\n        '
        super().__init__()
        self._node = node
        self._parent = parent_node
        self._old_parent = node.getParent()

    def undo(self) -> None:
        if False:
            while True:
                i = 10
        'Undoes the set-parent operation, restoring the old parent.'
        self._set_parent(self._old_parent)

    def redo(self) -> None:
        if False:
            return 10
        'Re-applies the set-parent operation.'
        self._set_parent(self._parent)

    def _set_parent(self, new_parent: Optional[SceneNode]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the parent of the node while applying transformations to the world-transform of the node stays the same.\n\n        :param new_parent: The new parent. Note: this argument can be None, which would hide the node from the scene.\n        '
        if new_parent:
            current_parent = self._node.getParent()
            if current_parent:
                old_parent = new_parent.callDecoration('getOldParent')
                if old_parent:
                    new_parent.callDecoration('getNode').setParent(old_parent)
                depth_difference = current_parent.getDepth() - new_parent.getDepth()
                child_transformation = self._node.getLocalTransformation()
                if depth_difference > 0:
                    parent_transformation = current_parent.getLocalTransformation()
                    self._node.setTransformation(parent_transformation.multiply(child_transformation))
                else:
                    parent_transformation = new_parent.getLocalTransformation()
                    result = parent_transformation.getInverse().multiply(child_transformation, copy=True)
                    self._node.setTransformation(result)
        self._node.setParent(new_parent)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns a programmer-readable representation of this operation.\n\n        :return: A programmer-readable representation of this operation.\n        '
        return 'SetParentOperation(node = {0}, parent_node={1})'.format(self._node, self._parent)