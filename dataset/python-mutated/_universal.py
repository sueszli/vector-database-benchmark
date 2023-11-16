"""
This module is the requests implementation of Pipeline ABC
"""
import json
import inspect
import logging
import os
import platform
import xml.etree.ElementTree as ET
import types
import re
import uuid
from typing import IO, cast, Union, Optional, AnyStr, Dict, Any, Set, Mapping
import urllib.parse
from azure.core import __version__ as azcore_version
from azure.core.exceptions import DecodeError
from azure.core.pipeline import PipelineRequest, PipelineResponse
from ._base import SansIOHTTPPolicy
from ..transport import HttpRequest as LegacyHttpRequest
from ..transport._base import _HttpResponseBase as LegacySansIOHttpResponse
from ...rest import HttpRequest
from ...rest._rest_py3 import _HttpResponseBase as SansIOHttpResponse
_LOGGER = logging.getLogger(__name__)
HTTPRequestType = Union[LegacyHttpRequest, HttpRequest]
HTTPResponseType = Union[LegacySansIOHttpResponse, SansIOHttpResponse]
PipelineResponseType = PipelineResponse[HTTPRequestType, HTTPResponseType]

class HeadersPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A simple policy that sends the given headers with the request.

    This will overwrite any headers already defined in the request. Headers can be
    configured up front, where any custom headers will be applied to all outgoing
    operations, and additional headers can also be added dynamically per operation.

    :param dict base_headers: Headers to send with the request.

    .. admonition:: Example:

        .. literalinclude:: ../samples/test_example_sansio.py
            :start-after: [START headers_policy]
            :end-before: [END headers_policy]
            :language: python
            :dedent: 4
            :caption: Configuring a headers policy.
    """

    def __init__(self, base_headers: Optional[Dict[str, str]]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self._headers: Dict[str, str] = base_headers or {}
        self._headers.update(kwargs.pop('headers', {}))

    @property
    def headers(self) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        'The current headers collection.\n\n        :rtype: dict[str, str]\n        :return: The current headers collection.\n        '
        return self._headers

    def add_header(self, key: str, value: str) -> None:
        if False:
            i = 10
            return i + 15
        "Add a header to the configuration to be applied to all requests.\n\n        :param str key: The header.\n        :param str value: The header's value.\n        "
        self._headers[key] = value

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            while True:
                i = 10
        'Updates with the given headers before sending the request to the next policy.\n\n        :param request: The PipelineRequest object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        request.http_request.headers.update(self.headers)
        additional_headers = request.context.options.pop('headers', {})
        if additional_headers:
            request.http_request.headers.update(additional_headers)

class _Unset:
    pass

class RequestIdPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A simple policy that sets the given request id in the header.

    This will overwrite request id that is already defined in the request. Request id can be
    configured up front, where the request id will be applied to all outgoing
    operations, and additional request id can also be set dynamically per operation.

    :keyword str request_id: The request id to be added into header.
    :keyword bool auto_request_id: Auto generates a unique request ID per call if true which is by default.
    :keyword str request_id_header_name: Header name to use. Default is "x-ms-client-request-id".

    .. admonition:: Example:

        .. literalinclude:: ../samples/test_example_sansio.py
            :start-after: [START request_id_policy]
            :end-before: [END request_id_policy]
            :language: python
            :dedent: 4
            :caption: Configuring a request id policy.
    """

    def __init__(self, *, request_id: Union[str, Any]=_Unset, auto_request_id: bool=True, request_id_header_name: str='x-ms-client-request-id', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super()
        self._request_id = request_id
        self._auto_request_id = auto_request_id
        self._request_id_header_name = request_id_header_name

    def set_request_id(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        'Add the request id to the configuration to be applied to all requests.\n\n        :param str value: The request id value.\n        '
        self._request_id = value

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            return 10
        'Updates with the given request id before sending the request to the next policy.\n\n        :param request: The PipelineRequest object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        request_id = unset = object()
        if 'request_id' in request.context.options:
            request_id = request.context.options.pop('request_id')
            if request_id is None:
                return
        elif self._request_id is None:
            return
        elif self._request_id is not _Unset:
            if self._request_id_header_name in request.http_request.headers:
                return
            request_id = self._request_id
        elif self._auto_request_id:
            if self._request_id_header_name in request.http_request.headers:
                return
            request_id = str(uuid.uuid1())
        if request_id is not unset:
            header = {self._request_id_header_name: cast(str, request_id)}
            request.http_request.headers.update(header)

class UserAgentPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """User-Agent Policy. Allows custom values to be added to the User-Agent header.

    :param str base_user_agent: Sets the base user agent value.

    :keyword bool user_agent_overwrite: Overwrites User-Agent when True. Defaults to False.
    :keyword bool user_agent_use_env: Gets user-agent from environment. Defaults to True.
    :keyword str user_agent: If specified, this will be added in front of the user agent string.
    :keyword str sdk_moniker: If specified, the user agent string will be
        azsdk-python-[sdk_moniker] Python/[python_version] ([platform_version])

    .. admonition:: Example:

        .. literalinclude:: ../samples/test_example_sansio.py
            :start-after: [START user_agent_policy]
            :end-before: [END user_agent_policy]
            :language: python
            :dedent: 4
            :caption: Configuring a user agent policy.
    """
    _USERAGENT = 'User-Agent'
    _ENV_ADDITIONAL_USER_AGENT = 'AZURE_HTTP_USER_AGENT'

    def __init__(self, base_user_agent: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            return 10
        self.overwrite: bool = kwargs.pop('user_agent_overwrite', False)
        self.use_env: bool = kwargs.pop('user_agent_use_env', True)
        application_id: Optional[str] = kwargs.pop('user_agent', None)
        sdk_moniker: str = kwargs.pop('sdk_moniker', 'core/{}'.format(azcore_version))
        if base_user_agent:
            self._user_agent = base_user_agent
        else:
            self._user_agent = 'azsdk-python-{} Python/{} ({})'.format(sdk_moniker, platform.python_version(), platform.platform())
        if application_id:
            self._user_agent = '{} {}'.format(application_id, self._user_agent)

    @property
    def user_agent(self) -> str:
        if False:
            i = 10
            return i + 15
        'The current user agent value.\n\n        :return: The current user agent value.\n        :rtype: str\n        '
        if self.use_env:
            add_user_agent_header = os.environ.get(self._ENV_ADDITIONAL_USER_AGENT, None)
            if add_user_agent_header is not None:
                return '{} {}'.format(self._user_agent, add_user_agent_header)
        return self._user_agent

    def add_user_agent(self, value: str) -> None:
        if False:
            print('Hello World!')
        'Add value to current user agent with a space.\n        :param str value: value to add to user agent.\n        '
        self._user_agent = '{} {}'.format(self._user_agent, value)

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            print('Hello World!')
        'Modifies the User-Agent header before the request is sent.\n\n        :param request: The PipelineRequest object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        http_request = request.http_request
        options_dict = request.context.options
        if 'user_agent' in options_dict:
            user_agent = options_dict.pop('user_agent')
            if options_dict.pop('user_agent_overwrite', self.overwrite):
                http_request.headers[self._USERAGENT] = user_agent
            else:
                user_agent = '{} {}'.format(user_agent, self.user_agent)
                http_request.headers[self._USERAGENT] = user_agent
        elif self.overwrite or self._USERAGENT not in http_request.headers:
            http_request.headers[self._USERAGENT] = self.user_agent

class NetworkTraceLoggingPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """The logging policy in the pipeline is used to output HTTP network trace to the configured logger.

    This accepts both global configuration, and per-request level with "enable_http_logger"

    :param bool logging_enable: Use to enable per operation. Defaults to False.

    .. admonition:: Example:

        .. literalinclude:: ../samples/test_example_sansio.py
            :start-after: [START network_trace_logging_policy]
            :end-before: [END network_trace_logging_policy]
            :language: python
            :dedent: 4
            :caption: Configuring a network trace logging policy.
    """

    def __init__(self, logging_enable: bool=False, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        self.enable_http_logger = logging_enable

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            i = 10
            return i + 15
        'Logs HTTP request to the DEBUG logger.\n\n        :param request: The PipelineRequest object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        http_request = request.http_request
        options = request.context.options
        logging_enable = options.pop('logging_enable', self.enable_http_logger)
        request.context['logging_enable'] = logging_enable
        if logging_enable:
            if not _LOGGER.isEnabledFor(logging.DEBUG):
                return
            try:
                log_string = "Request URL: '{}'".format(http_request.url)
                log_string += "\nRequest method: '{}'".format(http_request.method)
                log_string += '\nRequest headers:'
                for (header, value) in http_request.headers.items():
                    log_string += "\n    '{}': '{}'".format(header, value)
                log_string += '\nRequest body:'
                if isinstance(http_request.body, types.GeneratorType):
                    log_string += '\nFile upload'
                    _LOGGER.debug(log_string)
                    return
                try:
                    if isinstance(http_request.body, types.AsyncGeneratorType):
                        log_string += '\nFile upload'
                        _LOGGER.debug(log_string)
                        return
                except AttributeError:
                    pass
                if http_request.body:
                    log_string += '\n{}'.format(str(http_request.body))
                    _LOGGER.debug(log_string)
                    return
                log_string += '\nThis request has no body'
                _LOGGER.debug(log_string)
            except Exception as err:
                _LOGGER.debug('Failed to log request: %r', err)

    def on_response(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> None:
        if False:
            while True:
                i = 10
        'Logs HTTP response to the DEBUG logger.\n\n        :param request: The PipelineRequest object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :param response: The PipelineResponse object.\n        :type response: ~azure.core.pipeline.PipelineResponse\n        '
        http_response = response.http_response
        try:
            logging_enable = response.context['logging_enable']
            if logging_enable:
                if not _LOGGER.isEnabledFor(logging.DEBUG):
                    return
                log_string = "Response status: '{}'".format(http_response.status_code)
                log_string += '\nResponse headers:'
                for (res_header, value) in http_response.headers.items():
                    log_string += "\n    '{}': '{}'".format(res_header, value)
                log_string += '\nResponse content:'
                pattern = re.compile('attachment; ?filename=["\\w.]+', re.IGNORECASE)
                header = http_response.headers.get('content-disposition')
                if header and pattern.match(header):
                    filename = header.partition('=')[2]
                    log_string += '\nFile attachments: {}'.format(filename)
                elif http_response.headers.get('content-type', '').endswith('octet-stream'):
                    log_string += '\nBody contains binary data.'
                elif http_response.headers.get('content-type', '').startswith('image'):
                    log_string += '\nBody contains image data.'
                elif response.context.options.get('stream', False):
                    log_string += '\nBody is streamable.'
                else:
                    log_string += '\n{}'.format(http_response.text())
                _LOGGER.debug(log_string)
        except Exception as err:
            _LOGGER.debug('Failed to log response: %s', repr(err))

class _HiddenClassProperties(type):

    @property
    def DEFAULT_HEADERS_WHITELIST(cls) -> Set[str]:
        if False:
            i = 10
            return i + 15
        return cls.DEFAULT_HEADERS_ALLOWLIST

    @DEFAULT_HEADERS_WHITELIST.setter
    def DEFAULT_HEADERS_WHITELIST(cls, value: Set[str]) -> None:
        if False:
            i = 10
            return i + 15
        cls.DEFAULT_HEADERS_ALLOWLIST = value

class HttpLoggingPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType], metaclass=_HiddenClassProperties):
    """The Pipeline policy that handles logging of HTTP requests and responses.

    :param logger: The logger to use for logging. Default to azure.core.pipeline.policies.http_logging_policy.
    :type logger: logging.Logger
    """
    DEFAULT_HEADERS_ALLOWLIST: Set[str] = set(['x-ms-request-id', 'x-ms-client-request-id', 'x-ms-return-client-request-id', 'x-ms-error-code', 'traceparent', 'Accept', 'Cache-Control', 'Connection', 'Content-Length', 'Content-Type', 'Date', 'ETag', 'Expires', 'If-Match', 'If-Modified-Since', 'If-None-Match', 'If-Unmodified-Since', 'Last-Modified', 'Pragma', 'Request-Id', 'Retry-After', 'Server', 'Transfer-Encoding', 'User-Agent', 'WWW-Authenticate'])
    REDACTED_PLACEHOLDER: str = 'REDACTED'
    MULTI_RECORD_LOG: str = 'AZURE_SDK_LOGGING_MULTIRECORD'

    def __init__(self, logger: Optional[logging.Logger]=None, **kwargs: Any):
        if False:
            print('Hello World!')
        self.logger: logging.Logger = logger or logging.getLogger('azure.core.pipeline.policies.http_logging_policy')
        self.allowed_query_params: Set[str] = set()
        self.allowed_header_names: Set[str] = set(self.__class__.DEFAULT_HEADERS_ALLOWLIST)

    def _redact_query_param(self, key: str, value: str) -> str:
        if False:
            return 10
        lower_case_allowed_query_params = [param.lower() for param in self.allowed_query_params]
        return value if key.lower() in lower_case_allowed_query_params else HttpLoggingPolicy.REDACTED_PLACEHOLDER

    def _redact_header(self, key: str, value: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        lower_case_allowed_header_names = [header.lower() for header in self.allowed_header_names]
        return value if key.lower() in lower_case_allowed_header_names else HttpLoggingPolicy.REDACTED_PLACEHOLDER

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            i = 10
            return i + 15
        'Logs HTTP method, url and headers.\n        :param request: The PipelineRequest object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        http_request = request.http_request
        options = request.context.options
        logger = request.context.setdefault('logger', options.pop('logger', self.logger))
        if not logger.isEnabledFor(logging.INFO):
            return
        try:
            parsed_url = list(urllib.parse.urlparse(http_request.url))
            parsed_qp = urllib.parse.parse_qsl(parsed_url[4], keep_blank_values=True)
            filtered_qp = [(key, self._redact_query_param(key, value)) for (key, value) in parsed_qp]
            parsed_url[4] = '&'.join(['='.join(part) for part in filtered_qp])
            redacted_url = urllib.parse.urlunparse(parsed_url)
            multi_record = os.environ.get(HttpLoggingPolicy.MULTI_RECORD_LOG, False)
            if multi_record:
                logger.info('Request URL: %r', redacted_url)
                logger.info('Request method: %r', http_request.method)
                logger.info('Request headers:')
                for (header, value) in http_request.headers.items():
                    value = self._redact_header(header, value)
                    logger.info('    %r: %r', header, value)
                if isinstance(http_request.body, types.GeneratorType):
                    logger.info('File upload')
                    return
                try:
                    if isinstance(http_request.body, types.AsyncGeneratorType):
                        logger.info('File upload')
                        return
                except AttributeError:
                    pass
                if http_request.body:
                    logger.info('A body is sent with the request')
                    return
                logger.info('No body was attached to the request')
                return
            log_string = "Request URL: '{}'".format(redacted_url)
            log_string += "\nRequest method: '{}'".format(http_request.method)
            log_string += '\nRequest headers:'
            for (header, value) in http_request.headers.items():
                value = self._redact_header(header, value)
                log_string += "\n    '{}': '{}'".format(header, value)
            if isinstance(http_request.body, types.GeneratorType):
                log_string += '\nFile upload'
                logger.info(log_string)
                return
            try:
                if isinstance(http_request.body, types.AsyncGeneratorType):
                    log_string += '\nFile upload'
                    logger.info(log_string)
                    return
            except AttributeError:
                pass
            if http_request.body:
                log_string += '\nA body is sent with the request'
                logger.info(log_string)
                return
            log_string += '\nNo body was attached to the request'
            logger.info(log_string)
        except Exception as err:
            logger.warning('Failed to log request: %s', repr(err))

    def on_response(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> None:
        if False:
            while True:
                i = 10
        http_response = response.http_response
        options = request.context.options
        logger = request.context.setdefault('logger', options.pop('logger', self.logger))
        try:
            if not logger.isEnabledFor(logging.INFO):
                return
            multi_record = os.environ.get(HttpLoggingPolicy.MULTI_RECORD_LOG, False)
            if multi_record:
                logger.info('Response status: %r', http_response.status_code)
                logger.info('Response headers:')
                for (res_header, value) in http_response.headers.items():
                    value = self._redact_header(res_header, value)
                    logger.info('    %r: %r', res_header, value)
                return
            log_string = 'Response status: {}'.format(http_response.status_code)
            log_string += '\nResponse headers:'
            for (res_header, value) in http_response.headers.items():
                value = self._redact_header(res_header, value)
                log_string += "\n    '{}': '{}'".format(res_header, value)
            logger.info(log_string)
        except Exception as err:
            logger.warning('Failed to log response: %s', repr(err))

class ContentDecodePolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """Policy for decoding unstreamed response content.

    :param response_encoding: The encoding to use if known for this service (will disable auto-detection)
    :type response_encoding: str
    """
    JSON_REGEXP = re.compile('^(application|text)/([0-9a-z+.-]+\\+)?json$')
    CONTEXT_NAME = 'deserialized_data'

    def __init__(self, response_encoding: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self._response_encoding = response_encoding

    @classmethod
    def deserialize_from_text(cls, data: Optional[Union[AnyStr, IO[AnyStr]]], mime_type: Optional[str]=None, response: Optional[HTTPResponseType]=None) -> Any:
        if False:
            while True:
                i = 10
        'Decode response data according to content-type.\n\n        Accept a stream of data as well, but will be load at once in memory for now.\n        If no content-type, will return the string version (not bytes, not stream)\n\n        :param data: The data to deserialize.\n        :type data: str or bytes or file-like object\n        :param response: The HTTP response.\n        :type response: ~azure.core.pipeline.transport.HttpResponse\n        :param str mime_type: The mime type. As mime type, charset is not expected.\n        :param response: If passed, exception will be annotated with that response\n        :type response: any\n        :raises ~azure.core.exceptions.DecodeError: If deserialization fails\n        :returns: A dict (JSON), XML tree or str, depending of the mime_type\n        :rtype: dict[str, Any] or xml.etree.ElementTree.Element or str\n        '
        if not data:
            return None
        if hasattr(data, 'read'):
            data = cast(IO, data).read()
        if isinstance(data, bytes):
            data_as_str = data.decode(encoding='utf-8-sig')
        else:
            data_as_str = cast(str, data)
        if mime_type is None:
            return data_as_str
        if cls.JSON_REGEXP.match(mime_type):
            try:
                return json.loads(data_as_str)
            except ValueError as err:
                raise DecodeError(message='JSON is invalid: {}'.format(err), response=response, error=err) from err
        elif 'xml' in (mime_type or []):
            try:
                return ET.fromstring(data_as_str)
            except ET.ParseError as err:

                def _json_attemp(data):
                    if False:
                        return 10
                    try:
                        return (True, json.loads(data))
                    except ValueError:
                        return (False, None)
                (success, json_result) = _json_attemp(data)
                if success:
                    return json_result
                _LOGGER.critical("Wasn't XML not JSON, failing")
                raise DecodeError('XML is invalid', response=response) from err
        elif mime_type.startswith('text/'):
            return data_as_str
        raise DecodeError('Cannot deserialize content-type: {}'.format(mime_type))

    @classmethod
    def deserialize_from_http_generics(cls, response: HTTPResponseType, encoding: Optional[str]=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        'Deserialize from HTTP response.\n\n        Headers will tested for "content-type"\n\n        :param response: The HTTP response\n        :type response: any\n        :param str encoding: The encoding to use if known for this service (will disable auto-detection)\n        :raises ~azure.core.exceptions.DecodeError: If deserialization fails\n        :returns: A dict (JSON), XML tree or str, depending of the mime_type\n        :rtype: dict[str, Any] or xml.etree.ElementTree.Element or str\n        '
        if response.content_type:
            mime_type = response.content_type.split(';')[0].strip().lower()
        else:
            mime_type = 'application/json'
        if hasattr(response, 'read'):
            if not inspect.iscoroutinefunction(response.read):
                response.read()
        return cls.deserialize_from_text(response.text(encoding), mime_type, response=response)

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            while True:
                i = 10
        options = request.context.options
        response_encoding = options.pop('response_encoding', self._response_encoding)
        if response_encoding:
            request.context['response_encoding'] = response_encoding

    def on_response(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> None:
        if False:
            print('Hello World!')
        'Extract data from the body of a REST response object.\n        This will load the entire payload in memory.\n        Will follow Content-Type to parse.\n        We assume everything is UTF8 (BOM acceptable).\n\n        :param request: The PipelineRequest object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :param response: The PipelineResponse object.\n        :type response: ~azure.core.pipeline.PipelineResponse\n        :raises JSONDecodeError: If JSON is requested and parsing is impossible.\n        :raises UnicodeDecodeError: If bytes is not UTF8\n        :raises xml.etree.ElementTree.ParseError: If bytes is not valid XML\n        :raises ~azure.core.exceptions.DecodeError: If deserialization fails\n        '
        if response.context.options.get('stream', True):
            return
        response_encoding = request.context.get('response_encoding')
        response.context[self.CONTEXT_NAME] = self.deserialize_from_http_generics(response.http_response, response_encoding)

class ProxyPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A proxy policy.

    Dictionary mapping protocol or protocol and host to the URL of the proxy
    to be used on each Request.

    :param dict proxies: Maps protocol or protocol and hostname to the URL
     of the proxy.

    .. admonition:: Example:

        .. literalinclude:: ../samples/test_example_sansio.py
            :start-after: [START proxy_policy]
            :end-before: [END proxy_policy]
            :language: python
            :dedent: 4
            :caption: Configuring a proxy policy.
    """

    def __init__(self, proxies: Optional[Mapping[str, str]]=None, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        self.proxies = proxies

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            return 10
        ctxt = request.context.options
        if self.proxies and 'proxies' not in ctxt:
            ctxt['proxies'] = self.proxies