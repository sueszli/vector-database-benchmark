from abc import ABC, abstractmethod
from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto

class RegistryStore(ABC):
    """
    A registry store is a storage backend for the Feast registry.
    """

    @abstractmethod
    def get_registry_proto(self) -> RegistryProto:
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieves the registry proto from the registry path. If there is no file at that path,\n        raises a FileNotFoundError.\n\n        Returns:\n            Returns either the registry proto stored at the registry path, or an empty registry proto.\n        '
        pass

    @abstractmethod
    def update_registry_proto(self, registry_proto: RegistryProto):
        if False:
            return 10
        '\n        Overwrites the current registry proto with the proto passed in. This method\n        writes to the registry path.\n\n        Args:\n            registry_proto: the new RegistryProto\n        '
        pass

    @abstractmethod
    def teardown(self):
        if False:
            print('Hello World!')
        '\n        Tear down the registry.\n        '
        pass

class NoopRegistryStore(RegistryStore):

    def get_registry_proto(self) -> RegistryProto:
        if False:
            return 10
        pass

    def update_registry_proto(self, registry_proto: RegistryProto):
        if False:
            while True:
                i = 10
        pass

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        pass