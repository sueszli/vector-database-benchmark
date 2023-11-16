import abc
from typing import List, Tuple
from ray.rllib.utils.typing import NetworkType

class RLModuleWithTargetNetworksInterface(abc.ABC):
    """An RLModule Mixin for adding an interface for target networks.

    This is used for identifying the target networks that are used for stabilizing
    the updates of the current trainable networks of this RLModule.
    """

    @abc.abstractmethod
    def get_target_network_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        if False:
            print('Hello World!')
        'Returns a list of (target, current) networks.\n\n        This is used for identifying the target networks that are used for stabilizing\n        the updates of the current trainable networks of this RLModule.\n\n        Returns:\n            A list of (target, current) networks.\n        '