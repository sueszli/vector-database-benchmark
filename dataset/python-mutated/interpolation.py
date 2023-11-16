from abc import ABC, abstractmethod
from typing import Optional
from airbyte_cdk.sources.declarative.types import Config

class Interpolation(ABC):
    """
    Strategy for evaluating the interpolated value of a string at runtime using Jinja.
    """

    @abstractmethod
    def eval(self, input_str: str, config: Config, default: Optional[str]=None, **additional_options):
        if False:
            print('Hello World!')
        "\n        Interpolates the input string using the config, and additional options passed as parameter.\n\n        :param input_str: The string to interpolate\n        :param config: The user-provided configuration as specified by the source's spec\n        :param default: Default value to return if the evaluation returns an empty string\n        :param additional_options: Optional parameters used for interpolation\n        :return: The interpolated string\n        "
        pass