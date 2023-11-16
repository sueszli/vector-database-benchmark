from abc import abstractmethod
from enum import Enum
from typing import Any, Callable, Mapping, MutableMapping, Optional, Union
import requests
from airbyte_cdk.sources.declarative.auth.declarative_authenticator import DeclarativeAuthenticator
from airbyte_cdk.sources.declarative.requesters.error_handlers.response_status import ResponseStatus
from airbyte_cdk.sources.declarative.requesters.request_options.request_options_provider import RequestOptionsProvider
from airbyte_cdk.sources.declarative.types import StreamSlice, StreamState

class HttpMethod(Enum):
    """
    Http Method to use when submitting an outgoing HTTP request
    """
    GET = 'GET'
    POST = 'POST'

class Requester(RequestOptionsProvider):

    @abstractmethod
    def get_authenticator(self) -> DeclarativeAuthenticator:
        if False:
            return 10
        '\n        Specifies the authenticator to use when submitting requests\n        '
        pass

    @abstractmethod
    def get_url_base(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        :return: URL base for the  API endpoint e.g: if you wanted to hit https://myapi.com/v1/some_entity then this should return "https://myapi.com/v1/"\n        '

    @abstractmethod
    def get_path(self, *, stream_state: Optional[StreamState], stream_slice: Optional[StreamSlice], next_page_token: Optional[Mapping[str, Any]]) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL path for the API endpoint e.g: if you wanted to hit https://myapi.com/v1/some_entity then this should return "some_entity"\n        '

    @abstractmethod
    def get_method(self) -> HttpMethod:
        if False:
            i = 10
            return i + 15
        '\n        Specifies the HTTP method to use\n        '

    @abstractmethod
    def get_request_params(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> MutableMapping[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifies the query parameters that should be set on an outgoing HTTP request given the inputs.\n\n        E.g: you might want to define query parameters for paging if next_page_token is not None.\n        '

    @abstractmethod
    def interpret_response_status(self, response: requests.Response) -> ResponseStatus:
        if False:
            i = 10
            return i + 15
        '\n        Specifies conditions for backoff, error handling and reporting based on the response from the server.\n\n        By default, back off on the following HTTP response statuses:\n         - 429 (Too Many Requests) indicating rate limiting\n         - 500s to handle transient server errors\n\n        Unexpected but transient exceptions (connection timeout, DNS resolution failed, etc..) are retried by default.\n        '

    @abstractmethod
    def get_request_headers(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        '\n        Return any non-auth headers. Authentication headers will overwrite any overlapping headers returned from this method.\n        '

    @abstractmethod
    def get_request_body_data(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Optional[Mapping[str, Any]]:
        if False:
            return 10
        '\n        Specifies how to populate the body of the request with a non-JSON payload.\n\n        If returns a ready text that it will be sent as is.\n        If returns a dict that it will be converted to a urlencoded form.\n        E.g. {"key1": "value1", "key2": "value2"} => "key1=value1&key2=value2"\n\n        At the same time only one of the \'request_body_data\' and \'request_body_json\' functions can be overridden.\n        '

    @abstractmethod
    def get_request_body_json(self, *, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Optional[Mapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        "\n        Specifies how to populate the body of the request with a JSON payload.\n\n        At the same time only one of the 'request_body_data' and 'request_body_json' functions can be overridden.\n        "

    @abstractmethod
    def send_request(self, stream_state: Optional[StreamState]=None, stream_slice: Optional[StreamSlice]=None, next_page_token: Optional[Mapping[str, Any]]=None, path: Optional[str]=None, request_headers: Optional[Mapping[str, Any]]=None, request_params: Optional[Mapping[str, Any]]=None, request_body_data: Optional[Union[Mapping[str, Any], str]]=None, request_body_json: Optional[Mapping[str, Any]]=None, log_formatter: Optional[Callable[[requests.Response], Any]]=None) -> Optional[requests.Response]:
        if False:
            while True:
                i = 10
        "\n        Sends a request and returns the response. Might return no response if the error handler chooses to ignore the response or throw an exception in case of an error.\n        If path is set, the path configured on the requester itself is ignored.\n        If header, params and body are set, they are merged with the ones configured on the requester itself.\n\n        If a log formatter is provided, it's used to log the performed request and response. If it's not provided, no logging is performed.\n        "