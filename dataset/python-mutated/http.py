import logging
import os
import urllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union
from urllib.parse import urljoin
import requests
import requests_cache
from airbyte_cdk.models import SyncMode
from airbyte_cdk.sources.http_config import MAX_CONNECTION_POOL_SIZE
from airbyte_cdk.sources.streams.availability_strategy import AvailabilityStrategy
from airbyte_cdk.sources.streams.call_rate import APIBudget, CachedLimiterSession, LimiterSession
from airbyte_cdk.sources.streams.core import Stream, StreamData
from airbyte_cdk.sources.streams.http.availability_strategy import HttpAvailabilityStrategy
from airbyte_cdk.sources.utils.types import JsonType
from airbyte_cdk.utils.constants import ENV_REQUEST_CACHE_PATH
from requests.auth import AuthBase
from .auth.core import HttpAuthenticator, NoAuth
from .exceptions import DefaultBackoffException, RequestBodyException, UserDefinedBackoffException
from .rate_limiting import default_backoff_handler, user_defined_backoff_handler
BODY_REQUEST_METHODS = ('GET', 'POST', 'PUT', 'PATCH')

class HttpStream(Stream, ABC):
    """
    Base abstract class for an Airbyte Stream using the HTTP protocol. Basic building block for users building an Airbyte source for a HTTP API.
    """
    source_defined_cursor = True
    page_size: Optional[int] = None

    def __init__(self, authenticator: Optional[Union[AuthBase, HttpAuthenticator]]=None, api_budget: Optional[APIBudget]=None):
        if False:
            print('Hello World!')
        self._api_budget: APIBudget = api_budget or APIBudget(policies=[])
        self._session = self.request_session()
        self._session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=MAX_CONNECTION_POOL_SIZE, pool_maxsize=MAX_CONNECTION_POOL_SIZE))
        self._authenticator: HttpAuthenticator = NoAuth()
        if isinstance(authenticator, AuthBase):
            self._session.auth = authenticator
        elif authenticator:
            self._authenticator = authenticator

    @property
    def cache_filename(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Override if needed. Return the name of cache file\n        Note that if the environment variable REQUEST_CACHE_PATH is not set, the cache will be in-memory only.\n        '
        return f'{self.name}.sqlite'

    @property
    def use_cache(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Override if needed. If True, all records will be cached.\n        Note that if the environment variable REQUEST_CACHE_PATH is not set, the cache will be in-memory only.\n        '
        return False

    def request_session(self) -> requests.Session:
        if False:
            while True:
                i = 10
        '\n        Session factory based on use_cache property and call rate limits (api_budget parameter)\n        :return: instance of request-based session\n        '
        if self.use_cache:
            cache_dir = os.getenv(ENV_REQUEST_CACHE_PATH)
            if cache_dir:
                sqlite_path = str(Path(cache_dir) / self.cache_filename)
            else:
                sqlite_path = 'file::memory:?cache=shared'
            return CachedLimiterSession(sqlite_path, backend='sqlite', api_budget=self._api_budget)
        else:
            return LimiterSession(api_budget=self._api_budget)

    def clear_cache(self) -> None:
        if False:
            print('Hello World!')
        '\n        Clear cached requests for current session, can be called any time\n        '
        if isinstance(self._session, requests_cache.CachedSession):
            self._session.cache.clear()

    @property
    @abstractmethod
    def url_base(self) -> str:
        if False:
            print('Hello World!')
        '\n        :return: URL base for the  API endpoint e.g: if you wanted to hit https://myapi.com/v1/some_entity then this should return "https://myapi.com/v1/"\n        '

    @property
    def http_method(self) -> str:
        if False:
            return 10
        '\n        Override if needed. See get_request_data/get_request_json if using POST/PUT/PATCH.\n        '
        return 'GET'

    @property
    def raise_on_http_errors(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Override if needed. If set to False, allows opting-out of raising HTTP code exception.\n        '
        return True

    @property
    def max_retries(self) -> Union[int, None]:
        if False:
            while True:
                i = 10
        '\n        Override if needed. Specifies maximum amount of retries for backoff policy. Return None for no limit.\n        '
        return 5

    @property
    def max_time(self) -> Union[int, None]:
        if False:
            i = 10
            return i + 15
        '\n        Override if needed. Specifies maximum total waiting time (in seconds) for backoff policy. Return None for no limit.\n        '
        return 60 * 10

    @property
    def retry_factor(self) -> float:
        if False:
            return 10
        '\n        Override if needed. Specifies factor for backoff policy.\n        '
        return 5

    @property
    def authenticator(self) -> HttpAuthenticator:
        if False:
            for i in range(10):
                print('nop')
        return self._authenticator

    @property
    def availability_strategy(self) -> Optional[AvailabilityStrategy]:
        if False:
            i = 10
            return i + 15
        return HttpAvailabilityStrategy()

    @abstractmethod
    def next_page_token(self, response: requests.Response) -> Optional[Mapping[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Override this method to define a pagination strategy.\n\n        The value returned from this method is passed to most other methods in this class. Use it to form a request e.g: set headers or query params.\n\n        :return: The token for the next page from the input response object. Returning None means there are no more pages to read in this response.\n        '

    @abstractmethod
    def path(self, *, stream_state: Optional[Mapping[str, Any]]=None, stream_slice: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> str:
        if False:
            return 10
        '\n        Returns the URL path for the API endpoint e.g: if you wanted to hit https://myapi.com/v1/some_entity then this should return "some_entity"\n        '

    def request_params(self, stream_state: Optional[Mapping[str, Any]], stream_slice: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> MutableMapping[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Override this method to define the query parameters that should be set on an outgoing HTTP request given the inputs.\n\n        E.g: you might want to define query parameters for paging if next_page_token is not None.\n        '
        return {}

    def request_headers(self, stream_state: Optional[Mapping[str, Any]], stream_slice: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        '\n        Override to return any non-auth headers. Authentication headers will overwrite any overlapping headers returned from this method.\n        '
        return {}

    def request_body_data(self, stream_state: Optional[Mapping[str, Any]], stream_slice: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Optional[Union[Mapping[str, Any], str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Override when creating POST/PUT/PATCH requests to populate the body of the request with a non-JSON payload.\n\n        If returns a ready text that it will be sent as is.\n        If returns a dict that it will be converted to a urlencoded form.\n        E.g. {"key1": "value1", "key2": "value2"} => "key1=value1&key2=value2"\n\n        At the same time only one of the \'request_body_data\' and \'request_body_json\' functions can be overridden.\n        '
        return None

    def request_body_json(self, stream_state: Optional[Mapping[str, Any]], stream_slice: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Optional[Mapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        "\n        Override when creating POST/PUT/PATCH requests to populate the body of the request with a JSON payload.\n\n        At the same time only one of the 'request_body_data' and 'request_body_json' functions can be overridden.\n        "
        return None

    def request_kwargs(self, stream_state: Optional[Mapping[str, Any]], stream_slice: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Override to return a mapping of keyword arguments to be used when creating the HTTP request.\n        Any option listed in https://docs.python-requests.org/en/latest/api/#requests.adapters.BaseAdapter.send for can be returned from\n        this method. Note that these options do not conflict with request-level options such as headers, request params, etc..\n        '
        return {}

    @abstractmethod
    def parse_response(self, response: requests.Response, *, stream_state: Mapping[str, Any], stream_slice: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Iterable[Mapping[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the raw response object into a list of records.\n        By default, this returns an iterable containing the input. Override to parse differently.\n        :param response:\n        :param stream_state:\n        :param stream_slice:\n        :param next_page_token:\n        :return: An iterable containing the parsed response\n        '

    def should_retry(self, response: requests.Response) -> bool:
        if False:
            print('Hello World!')
        '\n        Override to set different conditions for backoff based on the response from the server.\n\n        By default, back off on the following HTTP response statuses:\n         - 429 (Too Many Requests) indicating rate limiting\n         - 500s to handle transient server errors\n\n        Unexpected but transient exceptions (connection timeout, DNS resolution failed, etc..) are retried by default.\n        '
        return response.status_code == 429 or 500 <= response.status_code < 600

    def backoff_time(self, response: requests.Response) -> Optional[float]:
        if False:
            while True:
                i = 10
        '\n        Override this method to dynamically determine backoff time e.g: by reading the X-Retry-After header.\n\n        This method is called only if should_backoff() returns True for the input request.\n\n        :param response:\n        :return how long to backoff in seconds. The return value may be a floating point number for subsecond precision. Returning None defers backoff\n        to the default backoff behavior (e.g using an exponential algorithm).\n        '
        return None

    def error_message(self, response: requests.Response) -> str:
        if False:
            while True:
                i = 10
        '\n        Override this method to specify a custom error message which can incorporate the HTTP response received\n\n        :param response: The incoming HTTP response from the partner API\n        :return:\n        '
        return ''

    def must_deduplicate_query_params(self) -> bool:
        if False:
            print('Hello World!')
        return False

    def deduplicate_query_params(self, url: str, params: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        '\n        Remove query parameters from params mapping if they are already encoded in the URL.\n        :param url: URL with\n        :param params:\n        :return:\n        '
        if params is None:
            params = {}
        query_string = urllib.parse.urlparse(url).query
        query_dict = {k: v[0] for (k, v) in urllib.parse.parse_qs(query_string).items()}
        duplicate_keys_with_same_value = {k for k in query_dict.keys() if str(params.get(k)) == str(query_dict[k])}
        return {k: v for (k, v) in params.items() if k not in duplicate_keys_with_same_value}

    def _create_prepared_request(self, path: str, headers: Optional[Mapping[str, str]]=None, params: Optional[Mapping[str, str]]=None, json: Optional[Mapping[str, Any]]=None, data: Optional[Union[str, Mapping[str, Any]]]=None) -> requests.PreparedRequest:
        if False:
            while True:
                i = 10
        url = self._join_url(self.url_base, path)
        if self.must_deduplicate_query_params():
            query_params = self.deduplicate_query_params(url, params)
        else:
            query_params = params or {}
        args = {'method': self.http_method, 'url': url, 'headers': headers, 'params': query_params}
        if self.http_method.upper() in BODY_REQUEST_METHODS:
            if json and data:
                raise RequestBodyException("At the same time only one of the 'request_body_data' and 'request_body_json' functions can return data")
            elif json:
                args['json'] = json
            elif data:
                args['data'] = data
        prepared_request: requests.PreparedRequest = self._session.prepare_request(requests.Request(**args))
        return prepared_request

    @classmethod
    def _join_url(cls, url_base: str, path: str) -> str:
        if False:
            print('Hello World!')
        return urljoin(url_base, path)

    def _send(self, request: requests.PreparedRequest, request_kwargs: Mapping[str, Any]) -> requests.Response:
        if False:
            return 10
        "\n        Wraps sending the request in rate limit and error handlers.\n        Please note that error handling for HTTP status codes will be ignored if raise_on_http_errors is set to False\n\n        This method handles two types of exceptions:\n            1. Expected transient exceptions e.g: 429 status code.\n            2. Unexpected transient exceptions e.g: timeout.\n\n        To trigger a backoff, we raise an exception that is handled by the backoff decorator. If an exception is not handled by the decorator will\n        fail the sync.\n\n        For expected transient exceptions, backoff time is determined by the type of exception raised:\n            1. CustomBackoffException uses the user-provided backoff value\n            2. DefaultBackoffException falls back on the decorator's default behavior e.g: exponential backoff\n\n        Unexpected transient exceptions use the default backoff parameters.\n        Unexpected persistent exceptions are not handled and will cause the sync to fail.\n        "
        self.logger.debug('Making outbound API request', extra={'headers': request.headers, 'url': request.url, 'request_body': request.body})
        response: requests.Response = self._session.send(request, **request_kwargs)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('Receiving response', extra={'headers': response.headers, 'status': response.status_code, 'body': response.text})
        if self.should_retry(response):
            custom_backoff_time = self.backoff_time(response)
            error_message = self.error_message(response)
            if custom_backoff_time:
                raise UserDefinedBackoffException(backoff=custom_backoff_time, request=request, response=response, error_message=error_message)
            else:
                raise DefaultBackoffException(request=request, response=response, error_message=error_message)
        elif self.raise_on_http_errors:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                self.logger.error(response.text)
                raise exc
        return response

    def _send_request(self, request: requests.PreparedRequest, request_kwargs: Mapping[str, Any]) -> requests.Response:
        if False:
            i = 10
            return i + 15
        '\n        Creates backoff wrappers which are responsible for retry logic\n        '
        '\n        Backoff package has max_tries parameter that means total number of\n        tries before giving up, so if this number is 0 no calls expected to be done.\n        But for this class we call it max_REtries assuming there would be at\n        least one attempt and some retry attempts, to comply this logic we add\n        1 to expected retries attempts.\n        '
        max_tries = self.max_retries
        "\n        According to backoff max_tries docstring:\n            max_tries: The maximum number of attempts to make before giving\n                up ...The default value of None means there is no limit to\n                the number of tries.\n        This implies that if max_tries is explicitly set to None there is no\n        limit to retry attempts, otherwise it is limited number of tries. But\n        this is not true for current version of backoff packages (1.8.0). Setting\n        max_tries to 0 or negative number would result in endless retry attempts.\n        Add this condition to avoid an endless loop if it hasn't been set\n        explicitly (i.e. max_retries is not None).\n        "
        max_time = self.max_time
        '\n        According to backoff max_time docstring:\n            max_time: The maximum total amount of time to try for before\n                giving up. Once expired, the exception will be allowed to\n                escape. If a callable is passed, it will be\n                evaluated at runtime and its return value used.\n        '
        if max_tries is not None:
            max_tries = max(0, max_tries) + 1
        user_backoff_handler = user_defined_backoff_handler(max_tries=max_tries, max_time=max_time)(self._send)
        backoff_handler = default_backoff_handler(max_tries=max_tries, max_time=max_time, factor=self.retry_factor)
        return backoff_handler(user_backoff_handler)(request, request_kwargs)

    @classmethod
    def parse_response_error_message(cls, response: requests.Response) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Parses the raw response object from a failed request into a user-friendly error message.\n        By default, this method tries to grab the error message from JSON responses by following common API patterns. Override to parse differently.\n\n        :param response:\n        :return: A user-friendly message that indicates the cause of the error\n        '

        def _try_get_error(value: Optional[JsonType]) -> Optional[str]:
            if False:
                print('Hello World!')
            if isinstance(value, str):
                return value
            elif isinstance(value, list):
                errors_in_value = [_try_get_error(v) for v in value]
                return ', '.join((v for v in errors_in_value if v is not None))
            elif isinstance(value, dict):
                new_value = value.get('message') or value.get('messages') or value.get('error') or value.get('errors') or value.get('failures') or value.get('failure') or value.get('detail')
                return _try_get_error(new_value)
            return None
        try:
            body = response.json()
            return _try_get_error(body)
        except requests.exceptions.JSONDecodeError:
            return None

    def get_error_display_message(self, exception: BaseException) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Retrieves the user-friendly display message that corresponds to an exception.\n        This will be called when encountering an exception while reading records from the stream, and used to build the AirbyteTraceMessage.\n\n        The default implementation of this method only handles HTTPErrors by passing the response to self.parse_response_error_message().\n        The method should be overriden as needed to handle any additional exception types.\n\n        :param exception: The exception that was raised\n        :return: A user-friendly message that indicates the cause of the error\n        '
        if isinstance(exception, requests.HTTPError) and exception.response is not None:
            return self.parse_response_error_message(exception.response)
        return None

    def read_records(self, sync_mode: SyncMode, cursor_field: Optional[List[str]]=None, stream_slice: Optional[Mapping[str, Any]]=None, stream_state: Optional[Mapping[str, Any]]=None) -> Iterable[StreamData]:
        if False:
            i = 10
            return i + 15
        yield from self._read_pages(lambda req, res, state, _slice: self.parse_response(res, stream_slice=_slice, stream_state=state), stream_slice, stream_state)

    def _read_pages(self, records_generator_fn: Callable[[requests.PreparedRequest, requests.Response, Mapping[str, Any], Optional[Mapping[str, Any]]], Iterable[StreamData]], stream_slice: Optional[Mapping[str, Any]]=None, stream_state: Optional[Mapping[str, Any]]=None) -> Iterable[StreamData]:
        if False:
            i = 10
            return i + 15
        stream_state = stream_state or {}
        pagination_complete = False
        next_page_token = None
        while not pagination_complete:
            (request, response) = self._fetch_next_page(stream_slice, stream_state, next_page_token)
            yield from records_generator_fn(request, response, stream_state, stream_slice)
            next_page_token = self.next_page_token(response)
            if not next_page_token:
                pagination_complete = True
        yield from []

    def _fetch_next_page(self, stream_slice: Optional[Mapping[str, Any]]=None, stream_state: Optional[Mapping[str, Any]]=None, next_page_token: Optional[Mapping[str, Any]]=None) -> Tuple[requests.PreparedRequest, requests.Response]:
        if False:
            print('Hello World!')
        request_headers = self.request_headers(stream_state=stream_state, stream_slice=stream_slice, next_page_token=next_page_token)
        request = self._create_prepared_request(path=self.path(stream_state=stream_state, stream_slice=stream_slice, next_page_token=next_page_token), headers=dict(request_headers, **self.authenticator.get_auth_header()), params=self.request_params(stream_state=stream_state, stream_slice=stream_slice, next_page_token=next_page_token), json=self.request_body_json(stream_state=stream_state, stream_slice=stream_slice, next_page_token=next_page_token), data=self.request_body_data(stream_state=stream_state, stream_slice=stream_slice, next_page_token=next_page_token))
        request_kwargs = self.request_kwargs(stream_state=stream_state, stream_slice=stream_slice, next_page_token=next_page_token)
        response = self._send_request(request, request_kwargs)
        return (request, response)

class HttpSubStream(HttpStream, ABC):

    def __init__(self, parent: HttpStream, **kwargs: Any):
        if False:
            while True:
                i = 10
        '\n        :param parent: should be the instance of HttpStream class\n        '
        super().__init__(**kwargs)
        self.parent = parent

    def stream_slices(self, sync_mode: SyncMode, cursor_field: Optional[List[str]]=None, stream_state: Optional[Mapping[str, Any]]=None) -> Iterable[Optional[Mapping[str, Any]]]:
        if False:
            print('Hello World!')
        parent_stream_slices = self.parent.stream_slices(sync_mode=SyncMode.full_refresh, cursor_field=cursor_field, stream_state=stream_state)
        for stream_slice in parent_stream_slices:
            parent_records = self.parent.read_records(sync_mode=SyncMode.full_refresh, cursor_field=cursor_field, stream_slice=stream_slice, stream_state=stream_state)
            for record in parent_records:
                yield {'parent': record}