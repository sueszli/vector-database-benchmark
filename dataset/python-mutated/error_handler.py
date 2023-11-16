from abc import abstractmethod
from dataclasses import dataclass
from typing import Union
import requests
from airbyte_cdk.sources.declarative.requesters.error_handlers.response_status import ResponseStatus

@dataclass
class ErrorHandler:
    """
    Defines whether a request was successful and how to handle a failure.
    """

    @property
    @abstractmethod
    def max_retries(self) -> Union[int, None]:
        if False:
            return 10
        '\n        Specifies maximum amount of retries for backoff policy. Return None for no limit.\n        '
        pass

    @property
    @abstractmethod
    def max_time(self) -> Union[int, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifies maximum total waiting time (in seconds) for backoff policy. Return None for no limit.\n        '
        pass

    @abstractmethod
    def interpret_response(self, response: requests.Response) -> ResponseStatus:
        if False:
            return 10
        '\n        Evaluate response status describing whether a failing request should be retried or ignored.\n\n        :param response: response to evaluate\n        :return: response status\n        '
        pass