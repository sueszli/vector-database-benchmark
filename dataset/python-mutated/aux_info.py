from __future__ import annotations
from typing import TYPE_CHECKING, Generic, Optional, TypeVar
from qlib.typehint import final
from .simulator import StateType
if TYPE_CHECKING:
    from .utils.env_wrapper import EnvWrapper
__all__ = ['AuxiliaryInfoCollector']
AuxInfoType = TypeVar('AuxInfoType')

class AuxiliaryInfoCollector(Generic[StateType, AuxInfoType]):
    """Override this class to collect customized auxiliary information from environment."""
    env: Optional[EnvWrapper] = None

    @final
    def __call__(self, simulator_state: StateType) -> AuxInfoType:
        if False:
            while True:
                i = 10
        return self.collect(simulator_state)

    def collect(self, simulator_state: StateType) -> AuxInfoType:
        if False:
            return 10
        'Override this for customized auxiliary info.\n        Usually useful in Multi-agent RL.\n\n        Parameters\n        ----------\n        simulator_state\n            Retrieved with ``simulator.get_state()``.\n\n        Returns\n        -------\n        Auxiliary information.\n        '
        raise NotImplementedError('collect is not implemented!')