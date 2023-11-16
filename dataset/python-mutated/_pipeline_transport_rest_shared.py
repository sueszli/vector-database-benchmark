from __future__ import absolute_import
from io import BytesIO
from email.message import Message
from email.policy import HTTP
from email import message_from_bytes as message_parser
import os
from typing import TYPE_CHECKING, cast, IO, Union, Tuple, Optional, Callable, Type, Iterator, List, Sequence
from http.client import HTTPConnection
from urllib.parse import urlparse
from ..pipeline import PipelineRequest, PipelineResponse, PipelineContext
from ..pipeline._tools import await_result as _await_result
if TYPE_CHECKING:
    from azure.core.rest._rest_py3 import HttpRequest as RestHttpRequestPy3
    from azure.core.pipeline.transport import HttpRequest as PipelineTransportHttpRequest
    HTTPRequestType = Union[RestHttpRequestPy3, PipelineTransportHttpRequest]
    from ..pipeline.policies import SansIOHTTPPolicy
    from azure.core.pipeline.transport import HttpResponse as PipelineTransportHttpResponse, AioHttpTransportResponse as PipelineTransportAioHttpTransportResponse
    from azure.core.pipeline.transport._base import _HttpResponseBase as PipelineTransportHttpResponseBase
binary_type = str

class BytesIOSocket:
    """Mocking the "makefile" of socket for HTTPResponse.
    This can be used to create a http.client.HTTPResponse object
    based on bytes and not a real socket.

    :param bytes bytes_data: The bytes to use to mock the socket.
    """

    def __init__(self, bytes_data):
        if False:
            return 10
        self.bytes_data = bytes_data

    def makefile(self, *_):
        if False:
            while True:
                i = 10
        return BytesIO(self.bytes_data)

def _format_parameters_helper(http_request, params):
    if False:
        while True:
            i = 10
    "Helper for format_parameters.\n\n    Format parameters into a valid query string.\n    It's assumed all parameters have already been quoted as\n    valid URL strings.\n\n    :param http_request: The http request whose parameters\n     we are trying to format\n    :type http_request: any\n    :param dict params: A dictionary of parameters.\n    "
    query = urlparse(http_request.url).query
    if query:
        http_request.url = http_request.url.partition('?')[0]
        existing_params = {p[0]: p[-1] for p in [p.partition('=') for p in query.split('&')]}
        params.update(existing_params)
    query_params = []
    for (k, v) in params.items():
        if isinstance(v, list):
            for w in v:
                if w is None:
                    raise ValueError('Query parameter {} cannot be None'.format(k))
                query_params.append('{}={}'.format(k, w))
        else:
            if v is None:
                raise ValueError('Query parameter {} cannot be None'.format(k))
            query_params.append('{}={}'.format(k, v))
    query = '?' + '&'.join(query_params)
    http_request.url = http_request.url + query

def _pad_attr_name(attr: str, backcompat_attrs: Sequence[str]) -> str:
    if False:
        print('Hello World!')
    "Pad hidden attributes so users can access them.\n\n    Currently, for our backcompat attributes, we define them\n    as private, so they're hidden from intellisense and sphinx,\n    but still allow users to access them as public attributes\n    for backcompat purposes. This function is called so if\n    users access publicly call a private backcompat attribute,\n    we can return them the private variable in getattr\n\n    :param str attr: The attribute name\n    :param list[str] backcompat_attrs: The list of backcompat attributes\n    :rtype: str\n    :return: The padded attribute name\n    "
    return '_{}'.format(attr) if attr in backcompat_attrs else attr

def _prepare_multipart_body_helper(http_request: 'HTTPRequestType', content_index: int=0) -> int:
    if False:
        return 10
    'Helper for prepare_multipart_body.\n\n    Will prepare the body of this request according to the multipart information.\n\n    This call assumes the on_request policies have been applied already in their\n    correct context (sync/async)\n\n    Does nothing if "set_multipart_mixed" was never called.\n    :param http_request: The http request whose multipart body we are trying\n     to prepare\n    :type http_request: any\n    :param int content_index: The current index of parts within the batch message.\n    :returns: The updated index after all parts in this request have been added.\n    :rtype: int\n    '
    if not http_request.multipart_mixed_info:
        return 0
    requests: Sequence['HTTPRequestType'] = http_request.multipart_mixed_info[0]
    boundary: Optional[str] = http_request.multipart_mixed_info[2]
    main_message = Message()
    main_message.add_header('Content-Type', 'multipart/mixed')
    if boundary:
        main_message.set_boundary(boundary)
    for req in requests:
        part_message = Message()
        if req.multipart_mixed_info:
            content_index = req.prepare_multipart_body(content_index=content_index)
            part_message.add_header('Content-Type', req.headers['Content-Type'])
            payload = req.serialize()
            payload = payload[payload.index(b'--'):]
        else:
            part_message.add_header('Content-Type', 'application/http')
            part_message.add_header('Content-Transfer-Encoding', 'binary')
            part_message.add_header('Content-ID', str(content_index))
            payload = req.serialize()
            content_index += 1
        part_message.set_payload(payload)
        main_message.attach(part_message)
    full_message = main_message.as_bytes(policy=HTTP)
    final_boundary: str = cast(str, main_message.get_boundary())
    eol = b'\r\n'
    (_, _, body) = full_message.split(eol, 2)
    http_request.set_bytes_body(body)
    http_request.headers['Content-Type'] = 'multipart/mixed; boundary=' + final_boundary
    return content_index

class _HTTPSerializer(HTTPConnection):
    """Hacking the stdlib HTTPConnection to serialize HTTP request as strings."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.buffer = b''
        kwargs.setdefault('host', 'fakehost')
        super(_HTTPSerializer, self).__init__(*args, **kwargs)

    def putheader(self, header, *values):
        if False:
            for i in range(10):
                print('nop')
        if header in ['Host', 'Accept-Encoding']:
            return
        super(_HTTPSerializer, self).putheader(header, *values)

    def send(self, data):
        if False:
            return 10
        self.buffer += data

def _serialize_request(http_request: 'HTTPRequestType') -> bytes:
    if False:
        return 10
    'Helper for serialize.\n\n    Serialize a request using the application/http spec/\n\n    :param http_request: The http request which we are trying\n     to serialize.\n    :type http_request: any\n    :rtype: bytes\n    :return: The serialized request\n    '
    if isinstance(http_request.body, dict):
        raise TypeError('Cannot serialize an HTTPRequest with dict body.')
    serializer = _HTTPSerializer()
    serializer.request(method=http_request.method, url=http_request.url, body=http_request.body, headers=http_request.headers)
    return serializer.buffer

def _decode_parts_helper(response: 'PipelineTransportHttpResponseBase', message: Message, http_response_type: Type['PipelineTransportHttpResponseBase'], requests: Sequence['PipelineTransportHttpRequest'], deserialize_response: Callable) -> List['PipelineTransportHttpResponse']:
    if False:
        return 10
    'Helper for _decode_parts.\n\n    Rebuild an HTTP response from pure string.\n\n    :param response: The response to decode\n    :type response: ~azure.core.pipeline.transport.HttpResponse\n    :param message: The message to decode\n    :type message: ~email.message.Message\n    :param http_response_type: The type of response to return\n    :type http_response_type: ~azure.core.pipeline.transport.HttpResponse\n    :param requests: The requests that were batched together\n    :type requests: list[~azure.core.pipeline.transport.HttpRequest]\n    :param deserialize_response: The function to deserialize the response\n    :type deserialize_response: callable\n    :rtype: list[~azure.core.pipeline.transport.HttpResponse]\n    :return: The list of responses\n    '
    responses = []
    for (index, raw_response) in enumerate(message.get_payload()):
        content_type = raw_response.get_content_type()
        if content_type == 'application/http':
            try:
                matching_request = requests[index]
            except IndexError:
                matching_request = response.request
            responses.append(deserialize_response(raw_response.get_payload(decode=True), matching_request, http_response_type=http_response_type))
        elif content_type == 'multipart/mixed' and requests[index].multipart_mixed_info:
            changeset_requests = requests[index].multipart_mixed_info[0]
            changeset_responses = response._decode_parts(raw_response, http_response_type, changeset_requests)
            responses.extend(changeset_responses)
        else:
            raise ValueError("Multipart doesn't support part other than application/http for now")
    return responses

def _get_raw_parts_helper(response, http_response_type: Type):
    if False:
        print('Hello World!')
    'Helper for _get_raw_parts\n\n    Assuming this body is multipart, return the iterator or parts.\n\n    If parts are application/http use http_response_type or HttpClientTransportResponse\n    as envelope.\n\n    :param response: The response to decode\n    :type response: ~azure.core.pipeline.transport.HttpResponse\n    :param http_response_type: The type of response to return\n    :type http_response_type: any\n    :rtype: iterator[~azure.core.pipeline.transport.HttpResponse]\n    :return: The parts of the response\n    '
    body_as_bytes = response.body()
    http_body = b'Content-Type: ' + response.content_type.encode('ascii') + b'\r\n\r\n' + body_as_bytes
    message: Message = message_parser(http_body)
    requests = response.request.multipart_mixed_info[0]
    return response._decode_parts(message, http_response_type, requests)

def _parts_helper(response: 'PipelineTransportHttpResponse') -> Iterator['PipelineTransportHttpResponse']:
    if False:
        while True:
            i = 10
    'Assuming the content-type is multipart/mixed, will return the parts as an iterator.\n\n    :param response: The response to decode\n    :type response: ~azure.core.pipeline.transport.HttpResponse\n    :rtype: iterator[HttpResponse]\n    :return: The parts of the response\n    :raises ValueError: If the content is not multipart/mixed\n    '
    if not response.content_type or not response.content_type.startswith('multipart/mixed'):
        raise ValueError("You can't get parts if the response is not multipart/mixed")
    responses = response._get_raw_parts()
    if response.request.multipart_mixed_info:
        policies: Sequence['SansIOHTTPPolicy'] = response.request.multipart_mixed_info[1]
        import concurrent.futures

        def parse_responses(response):
            if False:
                return 10
            http_request = response.request
            context = PipelineContext(None)
            pipeline_request = PipelineRequest(http_request, context)
            pipeline_response = PipelineResponse(http_request, response, context=context)
            for policy in policies:
                _await_result(policy.on_response, pipeline_request, pipeline_response)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            [_ for _ in executor.map(parse_responses, responses)]
    return responses

def _format_data_helper(data: Union[str, IO]) -> Union[Tuple[None, str], Tuple[Optional[str], IO, str]]:
    if False:
        print('Hello World!')
    'Helper for _format_data.\n\n    Format field data according to whether it is a stream or\n    a string for a form-data request.\n\n    :param data: The request field data.\n    :type data: str or file-like object.\n    :rtype: tuple[str, IO, str] or tuple[None, str]\n    :return: A tuple of (data name, data IO, "application/octet-stream") or (None, data str)\n    '
    if hasattr(data, 'read'):
        data = cast(IO, data)
        data_name = None
        try:
            if data.name[0] != '<' and data.name[-1] != '>':
                data_name = os.path.basename(data.name)
        except (AttributeError, TypeError):
            pass
        return (data_name, data, 'application/octet-stream')
    return (None, cast(str, data))

def _aiohttp_body_helper(response: 'PipelineTransportAioHttpTransportResponse') -> bytes:
    if False:
        i = 10
        return i + 15
    "Helper for body method of Aiohttp responses.\n\n    Since aiohttp body methods need decompression work synchronously,\n    need to share this code across old and new aiohttp transport responses\n    for backcompat.\n\n    :param response: The response to decode\n    :type response: ~azure.core.pipeline.transport.AioHttpTransportResponse\n    :rtype: bytes\n    :return: The response's bytes\n    "
    if response._content is None:
        raise ValueError('Body is not available. Call async method load_body, or do your call with stream=False.')
    if not response._decompress:
        return response._content
    if response._decompressed_content:
        return response._content
    enc = response.headers.get('Content-Encoding')
    if not enc:
        return response._content
    enc = enc.lower()
    if enc in ('gzip', 'deflate'):
        import zlib
        zlib_mode = 16 + zlib.MAX_WBITS if enc == 'gzip' else -zlib.MAX_WBITS
        decompressor = zlib.decompressobj(wbits=zlib_mode)
        response._content = decompressor.decompress(response._content)
        response._decompressed_content = True
        return response._content
    return response._content