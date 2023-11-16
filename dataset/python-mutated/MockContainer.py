from typing import Optional
from UM.Settings.Interfaces import ContainerInterface
import UM.PluginObject
from UM.Signal import Signal

class MockContainer(ContainerInterface, UM.PluginObject.PluginObject):
    """Fake container class to add to the container registry.

    This allows us to test the container registry without testing the container
    class. If something is wrong in the container class it won't influence this
    test.
    """

    def __init__(self, metadata=None):
        if False:
            return 10
        'Initialise a new definition container.\n\n        The container will have the specified ID and all metadata in the\n        provided dictionary.\n        '
        super().__init__()
        if metadata is None:
            self._metadata = {}
        else:
            self._metadata = metadata
        self._plugin_id = 'MockContainerPlugin'

    def getId(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the ID that was provided at initialisation.\n\n        :return: The ID of the container.\n        '
        return self._metadata['id']

    def getMetaData(self):
        if False:
            return 10
        'Gets all metadata of this container.\n\n        This returns the metadata dictionary that was provided in the\n        constructor of this mock container.\n\n        :return: The metadata for this container.\n        '
        return self._metadata

    def getMetaDataEntry(self, entry, default=None):
        if False:
            print('Hello World!')
        'Gets a metadata entry from the metadata dictionary.\n\n        :param key: The key of the metadata entry.\n        :return: The value of the metadata entry, or None if there is no such\n        entry.\n        '
        if entry in self._metadata:
            return self._metadata[entry]
        return default

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets a human-readable name for this container.\n\n        :return: The name from the metadata, or "MockContainer" if there was no\n        name provided.\n        '
        return self._metadata.get('name', 'MockContainer')

    @property
    def isEnabled(self):
        if False:
            for i in range(10):
                print('nop')
        'Get whether a container stack is enabled or not.\n\n        :return: Always returns True.\n        '
        return True

    def isReadOnly(self):
        if False:
            for i in range(10):
                print('nop')
        'Get whether the container item is stored on a read only location in the filesystem.\n\n        :return: Always returns False\n        '
        return False

    def getPath(self):
        if False:
            while True:
                i = 10
        'Mock get path'
        return '/path/to/the/light/side'

    def setPath(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Mock set path'
        pass

    def getAllKeys(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _trustHook(self, file_name: Optional[str]) -> bool:
        if False:
            return 10
        return True

    def setProperty(self, key, property_name, property_value, container=None, set_from_cache=False):
        if False:
            print('Hello World!')
        pass

    def getProperty(self, key, property_name, context=None):
        if False:
            i = 10
            return i + 15
        if key in self.items:
            return self.items[key]
        return None

    def getValue(self, key):
        if False:
            i = 10
            return i + 15
        'Get the value of a container item.\n\n        Since this mock container cannot contain any items, it always returns None.\n\n        :return: Always returns None.\n        '
        pass

    def hasProperty(self, key, property_name):
        if False:
            for i in range(10):
                print('nop')
        'Get whether the container item has a specific property.\n\n        This method is not implemented in the mock container.\n        '
        return key in self.items

    def serialize(self, ignored_metadata_keys=None):
        if False:
            while True:
                i = 10
        'Serializes the container to a string representation.\n\n        This method is not implemented in the mock container.\n        '
        raise NotImplementedError()

    def deserialize(self, serialized, file_name: Optional[str]=None):
        if False:
            print('Hello World!')
        'Deserializes the container from a string representation.\n\n        This method is not implemented in the mock container.\n        '
        raise NotImplementedError()

    @classmethod
    def getConfigurationTypeFromSerialized(cls, serialized: str):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @classmethod
    def getVersionFromSerialized(cls, serialized):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def isDirty(self):
        if False:
            return 10
        return True

    def setDirty(self, dirty):
        if False:
            return 10
        pass
    metaDataChanged = Signal()
    propertyChanged = Signal()
    containersChanged = Signal()