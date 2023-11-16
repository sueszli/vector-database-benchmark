from typing import Any, Dict, Set
NodeMetadataValue = Any
PROTECTED_KEYS: Set[str] = {'val', 'stack_trace', 'nn_module_stack', 'debug_handle', 'tensor_meta'}

class NodeMetadata:

    def __init__(self, data: Dict[str, Any]) -> None:
        if False:
            return 10
        self.data: Dict[str, Any] = data.copy()

    def __getitem__(self, key: str) -> NodeMetadataValue:
        if False:
            return 10
        return self.data[key]

    def __setitem__(self, key: str, value: NodeMetadataValue) -> NodeMetadataValue:
        if False:
            for i in range(10):
                print('nop')
        if key in PROTECTED_KEYS:
            raise RuntimeError(f'Could not override node key: {key}')
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return key in self.data

    def copy(self) -> 'NodeMetadata':
        if False:
            print('Hello World!')
        return NodeMetadata(self.data.copy())