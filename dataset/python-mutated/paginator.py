from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional
import requests
from airbyte_cdk.sources.declarative.requesters.request_options.request_options_provider import RequestOptionsProvider
from airbyte_cdk.sources.declarative.types import Record

@dataclass
class Paginator(ABC, RequestOptionsProvider):
    """
    Defines the token to use to fetch the next page of records from the API.

    If needed, the Paginator will set request options to be set on the HTTP request to fetch the next page of records.
    If the next_page_token is the path to the next page of records, then it should be accessed through the `path` method
    """

    @abstractmethod
    def reset(self) -> None:
        if False:
            return 10
        "\n        Reset the pagination's inner state\n        "

    @abstractmethod
    def next_page_token(self, response: requests.Response, last_records: List[Record]) -> Optional[Mapping[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Returns the next_page_token to use to fetch the next page of records.\n\n        :param response: the response to process\n        :param last_records: the records extracted from the response\n        :return: A mapping {"next_page_token": <token>} for the next page from the input response object. Returning None means there are no more pages to read in this response.\n        '
        pass

    @abstractmethod
    def path(self) -> Optional[str]:
        if False:
            return 10
        '\n        Returns the URL path to hit to fetch the next page of records\n\n        e.g: if you wanted to hit https://myapi.com/v1/some_entity then this will return "some_entity"\n\n        :return: path to hit to fetch the next request. Returning None means the path is not defined by the next_page_token\n        '
        pass