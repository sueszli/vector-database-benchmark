from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional
import requests
from airbyte_cdk.sources.declarative.types import Record, StreamSlice, StreamState

@dataclass
class HttpSelector:
    """
    Responsible for translating an HTTP response into a list of records by extracting records from the response and optionally filtering
    records based on a heuristic.
    """

    @abstractmethod
    def select_records(self, response: requests.Response, stream_state: StreamState, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> List[Record]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Selects records from the response\n        :param response: The response to select the records from\n        :param stream_state: The stream state\n        :param stream_slice: The stream slice\n        :param next_page_token: The paginator token\n        :return: List of Records selected from the response\n        '
        pass