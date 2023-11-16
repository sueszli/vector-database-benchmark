from typing import Any, Dict, Optional
from UM.ConfigurationErrorMessage import ConfigurationErrorMessage
from UM.Settings.ContainerRegistry import ContainerRegistry
from UM.Logger import Logger
from UM.Settings.InstanceContainer import InstanceContainer

class ContainerNode:
    """A node in the container tree. It represents one container.

    The container it represents is referenced by its container_id. During normal use of the tree, this container is
    not constructed. Only when parts of the tree need to get loaded in the container stack should it get constructed.
    """

    def __init__(self, container_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Creates a new node for the container tree.\n\n        :param container_id: The ID of the container that this node should represent.\n        '
        self.container_id = container_id
        self._container = None
        self.children_map = {}

    def getMetadata(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Gets the metadata of the container that this node represents.\n\n        Getting the metadata from the container directly is about 10x as fast.\n\n        :return: The metadata of the container in this node.\n        '
        return ContainerRegistry.getInstance().findContainersMetadata(id=self.container_id)[0]

    def getMetaDataEntry(self, entry: str, default: Any=None) -> Any:
        if False:
            print('Hello World!')
        'Get an entry from the metadata of the container that this node contains.\n\n        This is just a convenience function.\n\n        :param entry: The metadata entry key to return.\n        :param default: If the metadata is not present or the container is not found, the value of this default is\n        returned.\n\n        :return: The value of the metadata entry, or the default if it was not present.\n        '
        container_metadata = ContainerRegistry.getInstance().findContainersMetadata(id=self.container_id)
        if len(container_metadata) == 0:
            return default
        return container_metadata[0].get(entry, default)

    @property
    def container(self) -> Optional[InstanceContainer]:
        if False:
            i = 10
            return i + 15
        "The container that this node's container ID refers to.\n\n        This can be used to finally instantiate the container in order to put it in the container stack.\n\n        :return: A container.\n        "
        if not self._container:
            container_list = ContainerRegistry.getInstance().findInstanceContainers(id=self.container_id)
            if len(container_list) == 0:
                Logger.log('e', 'Failed to lazy-load container [{container_id}]. Cannot find it.'.format(container_id=self.container_id))
                error_message = ConfigurationErrorMessage.getInstance()
                error_message.addFaultyContainers(self.container_id)
                return None
            self._container = container_list[0]
        return self._container

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return '%s[%s]' % (self.__class__.__name__, self.container_id)