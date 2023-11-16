from ._deprecation_msg import deprecation_msg
from ray.util.annotations import Deprecated

@Deprecated(message=deprecation_msg)
class HuggingFaceCheckpoint:

    def __new__(cls: type, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise DeprecationWarning
__all__ = ['HuggingFaceCheckpoint']