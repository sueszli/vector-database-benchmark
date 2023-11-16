from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional
import requests
from airbyte_cdk.sources.declarative.types import Record

@dataclass
class PaginationStrategy:
    """
    Defines how to get the next page token
    """

    @property
    @abstractmethod
    def initial_token(self) -> Optional[Any]:
        if False:
            return 10
        '\n        Return the initial value of the token\n        '

    @abstractmethod
    def next_page_token(self, response: requests.Response, last_records: List[Record]) -> Optional[Any]:
        if False:
            return 10
        '\n        :param response: response to process\n        :param last_records: records extracted from the response\n        :return: next page token. Returns None if there are no more pages to fetch\n        '
        pass

    @abstractmethod
    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Reset the pagination's inner state\n        "

    @abstractmethod
    def get_page_size(self) -> Optional[int]:
        if False:
            return 10
        '\n        :return: page size: The number of records to fetch in a page. Returns None if unspecified\n        '