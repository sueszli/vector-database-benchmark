import re
from dataclasses import InitVar, dataclass
from typing import Any, Mapping, Optional, Union
import requests
from airbyte_cdk.sources.declarative.interpolation.interpolated_string import InterpolatedString
from airbyte_cdk.sources.declarative.requesters.error_handlers.backoff_strategies.header_helper import get_numeric_value_from_header
from airbyte_cdk.sources.declarative.requesters.error_handlers.backoff_strategy import BackoffStrategy
from airbyte_cdk.sources.declarative.types import Config

@dataclass
class WaitTimeFromHeaderBackoffStrategy(BackoffStrategy):
    """
    Extract wait time from http header

    Attributes:
        header (str): header to read wait time from
        regex (Optional[str]): optional regex to apply on the header to extract its value
    """
    header: Union[InterpolatedString, str]
    parameters: InitVar[Mapping[str, Any]]
    config: Config
    regex: Optional[str] = None

    def __post_init__(self, parameters: Mapping[str, Any]):
        if False:
            i = 10
            return i + 15
        self.regex = re.compile(self.regex) if self.regex else None
        self.header = InterpolatedString.create(self.header, parameters=parameters)

    def backoff(self, response: requests.Response, attempt_count: int) -> Optional[float]:
        if False:
            for i in range(10):
                print('nop')
        header = self.header.eval(config=self.config)
        header_value = get_numeric_value_from_header(response, header, self.regex)
        return header_value