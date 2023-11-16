from dataclasses import InitVar, dataclass
from typing import Any, Mapping, Optional, Union
from airbyte_cdk.sources.declarative.interpolation.jinja import JinjaInterpolation
from airbyte_cdk.sources.declarative.types import Config

@dataclass
class InterpolatedString:
    """
    Wrapper around a raw string to be interpolated with the Jinja2 templating engine

    Attributes:
        string (str): The string to evalute
        default (Optional[str]): The default value to return if the evaluation returns an empty string
        parameters (Mapping[str, Any]): Additional runtime parameters to be used for string interpolation
    """
    string: str
    parameters: InitVar[Mapping[str, Any]]
    default: Optional[str] = None

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            return 10
        self.default = self.default or self.string
        self._interpolation = JinjaInterpolation()
        self._parameters = parameters

    def eval(self, config: Config, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Interpolates the input string using the config and other optional arguments passed as parameter.\n\n        :param config: The user-provided configuration as specified by the source's spec\n        :param kwargs: Optional parameters used for interpolation\n        :return: The interpolated string\n        "
        return self._interpolation.eval(self.string, config, self.default, parameters=self._parameters, **kwargs)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, InterpolatedString):
            return False
        return self.string == other.string and self.default == other.default

    @classmethod
    def create(cls, string_or_interpolated: Union['InterpolatedString', str], *, parameters: Mapping[str, Any]):
        if False:
            return 10
        '\n        Helper function to obtain an InterpolatedString from either a raw string or an InterpolatedString.\n\n        :param string_or_interpolated: either a raw string or an InterpolatedString.\n        :param parameters: parameters propagated from parent component\n        :return: InterpolatedString representing the input string.\n        '
        if isinstance(string_or_interpolated, str):
            return InterpolatedString(string=string_or_interpolated, parameters=parameters)
        else:
            return string_or_interpolated