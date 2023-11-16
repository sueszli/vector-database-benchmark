from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Union
from airbyte_cdk.sources.declarative.types import StreamSlice, StreamState

@dataclass
class RequestOptionsProvider:
    """
    Defines the request options to set on an outgoing HTTP request

    Options can be passed by
    - request parameter
    - request headers
    - body data
    - json content
    """

    @abstractmethod
    def get_request_params(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> MutableMapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifies the query parameters that should be set on an outgoing HTTP request given the inputs.\n\n        E.g: you might want to define query parameters for paging if next_page_token is not None.\n        '
        pass

    @abstractmethod
    def get_request_headers(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        'Return any non-auth headers. Authentication headers will overwrite any overlapping headers returned from this method.'

    @abstractmethod
    def get_request_body_data(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Optional[Union[Mapping, str]]:
        if False:
            i = 10
            return i + 15
        '\n        Specifies how to populate the body of the request with a non-JSON payload.\n\n        If returns a ready text that it will be sent as is.\n        If returns a dict that it will be converted to a urlencoded form.\n        E.g. {"key1": "value1", "key2": "value2"} => "key1=value1&key2=value2"\n\n        At the same time only one of the \'request_body_data\' and \'request_body_json\' functions can be overridden.\n        '

    @abstractmethod
    def get_request_body_json(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Optional[Mapping]:
        if False:
            i = 10
            return i + 15
        "\n        Specifies how to populate the body of the request with a JSON payload.\n\n        At the same time only one of the 'request_body_data' and 'request_body_json' functions can be overridden.\n        "