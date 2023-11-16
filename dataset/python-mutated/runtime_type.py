from enum import Enum
from typing import TYPE_CHECKING, Type
from lightning.app.runners import CloudRuntime, MultiProcessRuntime
if TYPE_CHECKING:
    from lightning.app.runners.runtime import Runtime

class RuntimeType(Enum):
    MULTIPROCESS = 'multiprocess'
    CLOUD = 'cloud'

    def get_runtime(self) -> Type['Runtime']:
        if False:
            i = 10
            return i + 15
        if self == RuntimeType.MULTIPROCESS:
            return MultiProcessRuntime
        if self == RuntimeType.CLOUD:
            return CloudRuntime
        raise ValueError('Unknown runtime type')