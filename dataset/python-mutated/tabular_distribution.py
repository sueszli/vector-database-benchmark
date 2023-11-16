"""A tabular representation of a distribution for a game."""
from typing import Dict, Optional
from open_spiel.python.mfg import distribution
import pyspiel
DistributionDict = Dict[str, float]

class TabularDistribution(distribution.ParametricDistribution):
    """Distribution that uses a dictionary to store the values of the states."""

    def __init__(self, game: pyspiel.Game):
        if False:
            return 10
        self._distribution: DistributionDict = {}
        super().__init__(game)

    def value(self, state: pyspiel.State) -> float:
        if False:
            print('Hello World!')
        return self.value_str(self.state_to_str(state))

    def value_str(self, state_str: str, default_value: Optional[float]=None) -> float:
        if False:
            return 10
        'Returns the probability of the distribution on the state string given.\n\n    Args:\n      state_str: A string.\n      default_value: If not None, return this value if the state is not in the\n        support of the distribution.\n\n    Returns:\n      A `float`.\n\n    Raises:\n      ValueError: If the state has not been seen by the distribution and no\n        default value has been passed to the method.\n    '
        if default_value is None:
            try:
                return self._distribution[state_str]
            except KeyError as e:
                raise ValueError(f'Distribution not computed for state {state_str}') from e
        return self._distribution.get(state_str, default_value)

    def get_params(self) -> DistributionDict:
        if False:
            return 10
        return self._distribution

    def set_params(self, params: DistributionDict):
        if False:
            while True:
                i = 10
        self._distribution = params

    def state_to_str(self, state: pyspiel.State) -> str:
        if False:
            i = 10
            return i + 15
        return state.observation_string(pyspiel.PlayerId.DEFAULT_PLAYER_ID)

    @property
    def distribution(self) -> DistributionDict:
        if False:
            while True:
                i = 10
        return self._distribution