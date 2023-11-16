from pathlib import Path
from feast.infra.registry.registry_store import RegistryStore
from feast.protos.feast.core.Registry_pb2 import Registry as RegistryProto
from feast.repo_config import RegistryConfig

class FooRegistryStore(RegistryStore):

    def __init__(self, registry_config: RegistryConfig, repo_path: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.registry_proto = RegistryProto()

    def get_registry_proto(self):
        if False:
            print('Hello World!')
        return self.registry_proto

    def update_registry_proto(self, registry_proto: RegistryProto):
        if False:
            for i in range(10):
                print('nop')
        self.registry_proto = registry_proto

    def teardown(self):
        if False:
            return 10
        pass