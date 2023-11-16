from dataclasses import InitVar, dataclass
from typing import Any, Final, List, Mapping
from airbyte_cdk.sources.declarative.interpolation.jinja import JinjaInterpolation
from airbyte_cdk.sources.declarative.types import Config
FALSE_VALUES: Final[List[Any]] = ['False', 'false', '{}', '[]', '()', '', '0', '0.0', {}, False, [], (), set()]

@dataclass
class InterpolatedBoolean:
    f'\n    Wrapper around a string to be evaluated to a boolean value.\n    The string will be evaluated as False if it interpolates to a value in {FALSE_VALUES}\n\n    Attributes:\n        condition (str): The string representing the condition to evaluate to a boolean\n    '
    condition: str
    parameters: InitVar[Mapping[str, Any]]

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        self._default = 'False'
        self._interpolation = JinjaInterpolation()
        self._parameters = parameters

    def eval(self, config: Config, **additional_parameters):
        if False:
            return 10
        "\n        Interpolates the predicate condition string using the config and other optional arguments passed as parameter.\n\n        :param config: The user-provided configuration as specified by the source's spec\n        :param additional_parameters: Optional parameters used for interpolation\n        :return: The interpolated string\n        "
        if isinstance(self.condition, bool):
            return self.condition
        else:
            evaluated = self._interpolation.eval(self.condition, config, self._default, parameters=self._parameters, **additional_parameters)
            if evaluated in FALSE_VALUES:
                return False
            return True