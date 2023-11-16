import abc
import base64
import json
from enum import Enum
from typing import Optional, Any, Tuple, Callable, Dict, Sequence, Generic, TypeVar, cast, Union
from ..exceptions import HttpResponseError, DecodeError
from . import PollingMethod
from ..pipeline.policies._utils import get_retry_after
from ..pipeline._tools import is_rest
from .._enum_meta import CaseInsensitiveEnumMeta
from .. import PipelineClient
from ..pipeline import PipelineResponse
from ..pipeline.transport import HttpTransport, HttpRequest as LegacyHttpRequest, HttpResponse as LegacyHttpResponse, AsyncHttpResponse as LegacyAsyncHttpResponse
from ..rest import HttpRequest, HttpResponse, AsyncHttpResponse
HttpRequestType = Union[LegacyHttpRequest, HttpRequest]
HttpResponseType = Union[LegacyHttpResponse, HttpResponse]
AllHttpResponseType = Union[LegacyHttpResponse, HttpResponse, LegacyAsyncHttpResponse, AsyncHttpResponse]
LegacyPipelineResponseType = PipelineResponse[LegacyHttpRequest, LegacyHttpResponse]
NewPipelineResponseType = PipelineResponse[HttpRequest, HttpResponse]
PipelineResponseType = PipelineResponse[HttpRequestType, HttpResponseType]
HttpRequestTypeVar = TypeVar('HttpRequestTypeVar', bound=HttpRequestType)
HttpResponseTypeVar = TypeVar('HttpResponseTypeVar', bound=HttpResponseType)
AllHttpResponseTypeVar = TypeVar('AllHttpResponseTypeVar', bound=AllHttpResponseType)
ABC = abc.ABC
PollingReturnType_co = TypeVar('PollingReturnType_co', covariant=True)
PipelineClientType = TypeVar('PipelineClientType')
HTTPResponseType_co = TypeVar('HTTPResponseType_co', covariant=True)
HTTPRequestType_co = TypeVar('HTTPRequestType_co', covariant=True)
_FINISHED = frozenset(['succeeded', 'canceled', 'failed'])
_FAILED = frozenset(['canceled', 'failed'])
_SUCCEEDED = frozenset(['succeeded'])

def _get_content(response: AllHttpResponseType) -> bytes:
    if False:
        print('Hello World!')
    'Get the content of this response. This is designed specifically to avoid\n    a warning of mypy for body() access, as this method is deprecated.\n\n    :param response: The response object.\n    :type response: any\n    :return: The content of this response.\n    :rtype: bytes\n    '
    if isinstance(response, (LegacyHttpResponse, LegacyAsyncHttpResponse)):
        return response.body()
    return response.content

def _finished(status):
    if False:
        while True:
            i = 10
    if hasattr(status, 'value'):
        status = status.value
    return str(status).lower() in _FINISHED

def _failed(status):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(status, 'value'):
        status = status.value
    return str(status).lower() in _FAILED

def _succeeded(status):
    if False:
        print('Hello World!')
    if hasattr(status, 'value'):
        status = status.value
    return str(status).lower() in _SUCCEEDED

class BadStatus(Exception):
    pass

class BadResponse(Exception):
    pass

class OperationFailed(Exception):
    pass

def _as_json(response: AllHttpResponseType) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Assuming this is not empty, return the content as JSON.\n\n    Result/exceptions is not determined if you call this method without testing _is_empty.\n\n    :param response: The response object.\n    :type response: any\n    :return: The content of this response as dict.\n    :rtype: dict\n    :raises: DecodeError if response body contains invalid json data.\n    '
    try:
        return json.loads(response.text())
    except ValueError as err:
        raise DecodeError('Error occurred in deserializing the response body.') from err

def _raise_if_bad_http_status_and_method(response: AllHttpResponseType) -> None:
    if False:
        i = 10
        return i + 15
    'Check response status code is valid.\n\n    Must be 200, 201, 202, or 204.\n\n    :param response: The response object.\n    :type response: any\n    :raises: BadStatus if invalid status.\n    '
    code = response.status_code
    if code in {200, 201, 202, 204}:
        return
    raise BadStatus('Invalid return status {!r} for {!r} operation'.format(code, response.request.method))

def _is_empty(response: AllHttpResponseType) -> bool:
    if False:
        print('Hello World!')
    'Check if response body contains meaningful content.\n\n    :param response: The response object.\n    :type response: any\n    :return: True if response body is empty, False otherwise.\n    :rtype: bool\n    '
    return not bool(_get_content(response))

class LongRunningOperation(ABC, Generic[HTTPRequestType_co, HTTPResponseType_co]):
    """Protocol to implement for a long running operation algorithm."""

    @abc.abstractmethod
    def can_poll(self, pipeline_response: PipelineResponse[HTTPRequestType_co, HTTPResponseType_co]) -> bool:
        if False:
            while True:
                i = 10
        'Answer if this polling method could be used.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: True if this polling method could be used, False otherwise.\n        :rtype: bool\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_polling_url(self) -> str:
        if False:
            while True:
                i = 10
        'Return the polling URL.\n\n        :return: The polling URL.\n        :rtype: str\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def set_initial_status(self, pipeline_response: PipelineResponse[HTTPRequestType_co, HTTPResponseType_co]) -> str:
        if False:
            return 10
        'Process first response after initiating long running operation.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The initial status.\n        :rtype: str\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_status(self, pipeline_response: PipelineResponse[HTTPRequestType_co, HTTPResponseType_co]) -> str:
        if False:
            while True:
                i = 10
        'Return the status string extracted from this response.\n\n        :param pipeline_response: The response object.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The status string.\n        :rtype: str\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_final_get_url(self, pipeline_response: PipelineResponse[HTTPRequestType_co, HTTPResponseType_co]) -> Optional[str]:
        if False:
            while True:
                i = 10
        'If a final GET is needed, returns the URL.\n\n        :param pipeline_response: Success REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The URL to the final GET, or None if no final GET is needed.\n        :rtype: str or None\n        '
        raise NotImplementedError()

class _LroOption(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Known LRO options from Swagger."""
    FINAL_STATE_VIA = 'final-state-via'

class _FinalStateViaOption(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Possible final-state-via options."""
    AZURE_ASYNC_OPERATION_FINAL_STATE = 'azure-async-operation'
    LOCATION_FINAL_STATE = 'location'
    OPERATION_LOCATION_FINAL_STATE = 'operation-location'

class OperationResourcePolling(LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]):
    """Implements a operation resource polling, typically from Operation-Location.

    :param str operation_location_header: Name of the header to return operation format (default 'operation-location')
    :keyword dict[str, any] lro_options: Additional options for LRO. For more information, see
     https://aka.ms/azsdk/autorest/openapi/lro-options
    """
    _async_url: str
    'Url to resource monitor (AzureAsyncOperation or Operation-Location)'
    _location_url: Optional[str]
    'Location header if present'
    _request: Any
    'The initial request done'

    def __init__(self, operation_location_header: str='operation-location', *, lro_options: Optional[Dict[str, Any]]=None):
        if False:
            return 10
        self._operation_location_header = operation_location_header
        self._location_url = None
        self._lro_options = lro_options or {}

    def can_poll(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> bool:
        if False:
            print('Hello World!')
        'Check if status monitor header (e.g. Operation-Location) is present.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: True if this polling method could be used, False otherwise.\n        :rtype: bool\n        '
        response = pipeline_response.http_response
        return self._operation_location_header in response.headers

    def get_polling_url(self) -> str:
        if False:
            while True:
                i = 10
        'Return the polling URL.\n\n        Will extract it from the defined header to read (e.g. Operation-Location)\n\n        :return: The polling URL.\n        :rtype: str\n        '
        return self._async_url

    def get_final_get_url(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'If a final GET is needed, returns the URL.\n\n        :param pipeline_response: Success REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The URL to the final GET, or None if no final GET is needed.\n        :rtype: str or None\n        '
        if self._lro_options.get(_LroOption.FINAL_STATE_VIA) == _FinalStateViaOption.LOCATION_FINAL_STATE and self._location_url:
            return self._location_url
        if self._lro_options.get(_LroOption.FINAL_STATE_VIA) in [_FinalStateViaOption.AZURE_ASYNC_OPERATION_FINAL_STATE, _FinalStateViaOption.OPERATION_LOCATION_FINAL_STATE] and self._request.method == 'POST':
            return None
        response = pipeline_response.http_response
        if not _is_empty(response):
            body = _as_json(response)
            resource_location = body.get('resourceLocation')
            if resource_location:
                return resource_location
        if self._request.method in {'PUT', 'PATCH'}:
            return self._request.url
        if self._request.method == 'POST' and self._location_url:
            return self._location_url
        return None

    def set_initial_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            while True:
                i = 10
        'Process first response after initiating long running operation.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The initial status.\n        :rtype: str\n        '
        self._request = pipeline_response.http_response.request
        response = pipeline_response.http_response
        self._set_async_url_if_present(response)
        if response.status_code in {200, 201, 202, 204} and self._async_url:
            try:
                return self.get_status(pipeline_response)
            except Exception:
                pass
            return 'InProgress'
        raise OperationFailed('Operation failed or canceled')

    def _set_async_url_if_present(self, response: AllHttpResponseTypeVar) -> None:
        if False:
            print('Hello World!')
        self._async_url = response.headers[self._operation_location_header]
        location_url = response.headers.get('location')
        if location_url:
            self._location_url = location_url

    def get_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            print('Hello World!')
        'Process the latest status update retrieved from an "Operation-Location" header.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The status string.\n        :rtype: str\n        :raises: BadResponse if response has no body, or body does not contain status.\n        '
        response = pipeline_response.http_response
        if _is_empty(response):
            raise BadResponse('The response from long running operation does not contain a body.')
        body = _as_json(response)
        status = body.get('status')
        if not status:
            raise BadResponse('No status found in body')
        return status

class LocationPolling(LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]):
    """Implements a Location polling."""
    _location_url: str
    'Location header'

    def can_poll(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> bool:
        if False:
            return 10
        'True if contains a Location header\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: True if this polling method could be used, False otherwise.\n        :rtype: bool\n        '
        response = pipeline_response.http_response
        return 'location' in response.headers

    def get_polling_url(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the Location header value.\n\n        :return: The polling URL.\n        :rtype: str\n        '
        return self._location_url

    def get_final_get_url(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'If a final GET is needed, returns the URL.\n\n        Always return None for a Location polling.\n\n        :param pipeline_response: Success REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: Always None for this implementation.\n        :rtype: None\n        '
        return None

    def set_initial_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            return 10
        'Process first response after initiating long running operation.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The initial status.\n        :rtype: str\n        '
        response = pipeline_response.http_response
        self._location_url = response.headers['location']
        if response.status_code in {200, 201, 202, 204} and self._location_url:
            return 'InProgress'
        raise OperationFailed('Operation failed or canceled')

    def get_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the status string extracted from this response.\n\n        For Location polling, it means the status monitor returns 202.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The status string.\n        :rtype: str\n        '
        response = pipeline_response.http_response
        if 'location' in response.headers:
            self._location_url = response.headers['location']
        return 'InProgress' if response.status_code == 202 else 'Succeeded'

class StatusCheckPolling(LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]):
    """Should be the fallback polling, that don't poll but exit successfully
    if not other polling are detected and status code is 2xx.
    """

    def can_poll(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> bool:
        if False:
            while True:
                i = 10
        'Answer if this polling method could be used.\n\n        For this implementation, always True.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: True if this polling method could be used, False otherwise.\n        :rtype: bool\n        '
        return True

    def get_polling_url(self) -> str:
        if False:
            print('Hello World!')
        "Return the polling URL.\n\n        This is not implemented for this polling, since we're never supposed to loop.\n\n        :return: The polling URL.\n        :rtype: str\n        "
        raise ValueError("This polling doesn't support polling url")

    def set_initial_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Process first response after initiating long running operation.\n\n        Will succeed immediately.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The initial status.\n        :rtype: str\n        '
        return 'Succeeded'

    def get_status(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> str:
        if False:
            i = 10
            return i + 15
        'Return the status string extracted from this response.\n\n        Only possible status is success.\n\n        :param pipeline_response: Initial REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The status string.\n        :rtype: str\n        '
        return 'Succeeded'

    def get_final_get_url(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> Optional[str]:
        if False:
            while True:
                i = 10
        'If a final GET is needed, returns the URL.\n\n        :param pipeline_response: Success REST call response.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :rtype: str\n        :return: Always None for this implementation.\n        '
        return None

class _SansIOLROBasePolling(Generic[PollingReturnType_co, PipelineClientType, HttpRequestTypeVar, AllHttpResponseTypeVar]):
    """A base class that has no opinion on IO, to help mypy be accurate.

    :param float timeout: Default polling internal in absence of Retry-After header, in seconds.
    :param list[LongRunningOperation] lro_algorithms: Ordered list of LRO algorithms to use.
    :param lro_options: Additional options for LRO. For more information, see the algorithm's docstring.
    :type lro_options: dict[str, any]
    :param path_format_arguments: A dictionary of the format arguments to be used to format the URL.
    :type path_format_arguments: dict[str, str]
    """
    _deserialization_callback: Callable[[Any], PollingReturnType_co]
    'The deserialization callback that returns the final instance.'
    _operation: LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]
    "The algorithm this poller has decided to use. Will loop through 'can_poll' of the input algorithms to decide."
    _status: str
    'Hold the current status of this poller'
    _client: PipelineClientType
    'The Azure Core Pipeline client used to make request.'

    def __init__(self, timeout: float=30, lro_algorithms: Optional[Sequence[LongRunningOperation[HttpRequestTypeVar, AllHttpResponseTypeVar]]]=None, lro_options: Optional[Dict[str, Any]]=None, path_format_arguments: Optional[Dict[str, str]]=None, **operation_config: Any):
        if False:
            i = 10
            return i + 15
        self._lro_algorithms = lro_algorithms or [OperationResourcePolling(lro_options=lro_options), LocationPolling(), StatusCheckPolling()]
        self._timeout = timeout
        self._operation_config = operation_config
        self._lro_options = lro_options
        self._path_format_arguments = path_format_arguments

    def initialize(self, client: PipelineClientType, initial_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar], deserialization_callback: Callable[[PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]], PollingReturnType_co]) -> None:
        if False:
            while True:
                i = 10
        'Set the initial status of this LRO.\n\n        :param client: The Azure Core Pipeline client used to make request.\n        :type client: ~azure.core.pipeline.PipelineClient\n        :param initial_response: The initial response for the call.\n        :type initial_response: ~azure.core.pipeline.PipelineResponse\n        :param deserialization_callback: A callback function to deserialize the final response.\n        :type deserialization_callback: callable\n        :raises: HttpResponseError if initial status is incorrect LRO state\n        '
        self._client = client
        self._pipeline_response = self._initial_response = initial_response
        self._deserialization_callback = deserialization_callback
        for operation in self._lro_algorithms:
            if operation.can_poll(initial_response):
                self._operation = operation
                break
        else:
            raise BadResponse('Unable to find status link for polling.')
        try:
            _raise_if_bad_http_status_and_method(self._initial_response.http_response)
            self._status = self._operation.set_initial_status(initial_response)
        except BadStatus as err:
            self._status = 'Failed'
            raise HttpResponseError(response=initial_response.http_response, error=err) from err
        except BadResponse as err:
            self._status = 'Failed'
            raise HttpResponseError(response=initial_response.http_response, message=str(err), error=err) from err
        except OperationFailed as err:
            raise HttpResponseError(response=initial_response.http_response, error=err) from err

    def get_continuation_token(self) -> str:
        if False:
            i = 10
            return i + 15
        import pickle
        return base64.b64encode(pickle.dumps(self._initial_response)).decode('ascii')

    @classmethod
    def from_continuation_token(cls, continuation_token: str, **kwargs: Any) -> Tuple[Any, Any, Callable[[Any], PollingReturnType_co]]:
        if False:
            i = 10
            return i + 15
        try:
            client = kwargs['client']
        except KeyError:
            raise ValueError("Need kwarg 'client' to be recreated from continuation_token") from None
        try:
            deserialization_callback = kwargs['deserialization_callback']
        except KeyError:
            raise ValueError("Need kwarg 'deserialization_callback' to be recreated from continuation_token") from None
        import pickle
        initial_response = pickle.loads(base64.b64decode(continuation_token))
        initial_response.context.transport = client._pipeline._transport
        return (client, initial_response, deserialization_callback)

    def status(self) -> str:
        if False:
            print('Hello World!')
        'Return the current status as a string.\n\n        :rtype: str\n        :return: The current status.\n        '
        if not self._operation:
            raise ValueError('set_initial_status was never called. Did you give this instance to a poller?')
        return self._status

    def finished(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is this polling finished?\n\n        :rtype: bool\n        :return: True if finished, False otherwise.\n        '
        return _finished(self.status())

    def resource(self) -> PollingReturnType_co:
        if False:
            print('Hello World!')
        'Return the built resource.\n\n        :rtype: any\n        :return: The built resource.\n        '
        return self._parse_resource(self._pipeline_response)

    def _parse_resource(self, pipeline_response: PipelineResponse[HttpRequestTypeVar, AllHttpResponseTypeVar]) -> PollingReturnType_co:
        if False:
            i = 10
            return i + 15
        'Assuming this response is a resource, use the deserialization callback to parse it.\n        If body is empty, assuming no resource to return.\n\n        :param pipeline_response: The response object.\n        :type pipeline_response: ~azure.core.pipeline.PipelineResponse\n        :return: The parsed resource.\n        :rtype: any\n        '
        response = pipeline_response.http_response
        if not _is_empty(response):
            return self._deserialization_callback(pipeline_response)
        return None

    def _get_request_id(self) -> str:
        if False:
            while True:
                i = 10
        return self._pipeline_response.http_response.request.headers['x-ms-client-request-id']

    def _extract_delay(self) -> float:
        if False:
            while True:
                i = 10
        delay = get_retry_after(self._pipeline_response)
        if delay:
            return delay
        return self._timeout

class LROBasePolling(_SansIOLROBasePolling[PollingReturnType_co, PipelineClient[HttpRequestTypeVar, HttpResponseTypeVar], HttpRequestTypeVar, HttpResponseTypeVar], PollingMethod[PollingReturnType_co]):
    """A base LRO poller.

    This assumes a basic flow:
    - I analyze the response to decide the polling approach
    - I poll
    - I ask the final resource depending of the polling approach

    If your polling need are more specific, you could implement a PollingMethod directly
    """
    _initial_response: PipelineResponse[HttpRequestTypeVar, HttpResponseTypeVar]
    'Store the initial response.'
    _pipeline_response: PipelineResponse[HttpRequestTypeVar, HttpResponseTypeVar]
    'Store the latest received HTTP response, initialized by the first answer.'

    @property
    def _transport(self) -> HttpTransport[HttpRequestTypeVar, HttpResponseTypeVar]:
        if False:
            for i in range(10):
                print('nop')
        return self._client._pipeline._transport

    def __getattribute__(self, name: str) -> Any:
        if False:
            return 10
        'Find the right method for the job.\n\n        This contains a workaround for azure-mgmt-core 1.0.0 to 1.4.0, where the MRO\n        is changing when azure-core was refactored in 1.27.0. The MRO change was causing\n        AsyncARMPolling to look-up the wrong methods and find the non-async ones.\n\n        :param str name: The name of the attribute to retrieve.\n        :rtype: Any\n        :return: The attribute value.\n        '
        cls = object.__getattribute__(self, '__class__')
        if cls.__name__ == 'AsyncARMPolling' and name in ['run', 'update_status', 'request_status', '_sleep', '_delay', '_poll']:
            return getattr(super(LROBasePolling, self), name)
        return super().__getattribute__(name)

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            self._poll()
        except BadStatus as err:
            self._status = 'Failed'
            raise HttpResponseError(response=self._pipeline_response.http_response, error=err) from err
        except BadResponse as err:
            self._status = 'Failed'
            raise HttpResponseError(response=self._pipeline_response.http_response, message=str(err), error=err) from err
        except OperationFailed as err:
            raise HttpResponseError(response=self._pipeline_response.http_response, error=err) from err

    def _poll(self) -> None:
        if False:
            print('Hello World!')
        "Poll status of operation so long as operation is incomplete and\n        we have an endpoint to query.\n\n        :raises: OperationFailed if operation status 'Failed' or 'Canceled'.\n        :raises: BadStatus if response status invalid.\n        :raises: BadResponse if response invalid.\n        "
        if not self.finished():
            self.update_status()
        while not self.finished():
            self._delay()
            self.update_status()
        if _failed(self.status()):
            raise OperationFailed('Operation failed or canceled')
        final_get_url = self._operation.get_final_get_url(self._pipeline_response)
        if final_get_url:
            self._pipeline_response = self.request_status(final_get_url)
            _raise_if_bad_http_status_and_method(self._pipeline_response.http_response)

    def _sleep(self, delay: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._transport.sleep(delay)

    def _delay(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Check for a 'retry-after' header to set timeout,\n        otherwise use configured timeout.\n        "
        delay = self._extract_delay()
        self._sleep(delay)

    def update_status(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the current status of the LRO.'
        self._pipeline_response = self.request_status(self._operation.get_polling_url())
        _raise_if_bad_http_status_and_method(self._pipeline_response.http_response)
        self._status = self._operation.get_status(self._pipeline_response)

    def request_status(self, status_link: str) -> PipelineResponse[HttpRequestTypeVar, HttpResponseTypeVar]:
        if False:
            for i in range(10):
                print('nop')
        "Do a simple GET to this status link.\n\n        This method re-inject 'x-ms-client-request-id'.\n\n        :param str status_link: The URL to poll.\n        :rtype: azure.core.pipeline.PipelineResponse\n        :return: The response of the status request.\n        "
        if self._path_format_arguments:
            status_link = self._client.format_url(status_link, **self._path_format_arguments)
        if 'request_id' not in self._operation_config:
            self._operation_config['request_id'] = self._get_request_id()
        if is_rest(self._initial_response.http_response):
            rest_request = cast(HttpRequestTypeVar, HttpRequest('GET', status_link))
            return cast(PipelineResponse[HttpRequestTypeVar, HttpResponseTypeVar], self._client.send_request(rest_request, _return_pipeline_response=True, **self._operation_config))
        request = cast(HttpRequestTypeVar, self._client.get(status_link))
        return cast(PipelineResponse[HttpRequestTypeVar, HttpResponseTypeVar], self._client._pipeline.run(request, stream=False, **self._operation_config))
__all__ = ['BadResponse', 'BadStatus', 'OperationFailed', 'LongRunningOperation', 'OperationResourcePolling', 'LocationPolling', 'StatusCheckPolling', 'LROBasePolling']