from typing import TYPE_CHECKING
from UM.Settings.ContainerRegistry import ContainerRegistry
from cura.Machines.ContainerNode import ContainerNode
if TYPE_CHECKING:
    from cura.Machines.QualityNode import QualityNode

class IntentNode(ContainerNode):
    """This class represents an intent profile in the container tree.

    This class has no more subnodes.
    """

    def __init__(self, container_id: str, quality: 'QualityNode') -> None:
        if False:
            while True:
                i = 10
        super().__init__(container_id)
        self.quality = quality
        self.intent_category = ContainerRegistry.getInstance().findContainersMetadata(id=container_id)[0].get('intent_category', 'default')