"""A Python module for interacting with Slack's Web API."""
import asyncio
import copy
import hashlib
import hmac
import io
import json
import logging
import mimetypes
import urllib
import uuid
import warnings
from http.client import HTTPResponse
from ssl import SSLContext
from typing import BinaryIO, Dict, List, Any
from typing import Optional, Union
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen, OpenerDirector, ProxyHandler, HTTPSHandler
import aiohttp
from aiohttp import FormData, BasicAuth
import slack_sdk.errors as err
from slack_sdk.errors import SlackRequestError
from .async_internal_utils import _files_to_data, _get_event_loop, _request_with_session
from .deprecation import show_deprecation_warning_if_any
from .internal_utils import convert_bool_to_0_or_1, get_user_agent, _get_url, _build_req_args, _build_unexpected_body_error_message
from .legacy_slack_response import LegacySlackResponse as SlackResponse
from ..proxy_env_variable_loader import load_http_proxy_from_env

class LegacyBaseClient:
    BASE_URL = 'https://www.slack.com/api/'

    def __init__(self, token: Optional[str]=None, base_url: str=BASE_URL, timeout: int=30, loop: Optional[asyncio.AbstractEventLoop]=None, ssl: Optional[SSLContext]=None, proxy: Optional[str]=None, run_async: bool=False, use_sync_aiohttp: bool=False, session: Optional[aiohttp.ClientSession]=None, headers: Optional[dict]=None, user_agent_prefix: Optional[str]=None, user_agent_suffix: Optional[str]=None, team_id: Optional[str]=None, logger: Optional[logging.Logger]=None):
        if False:
            return 10
        self.token = None if token is None else token.strip()
        'A string specifying an `xoxp-*` or `xoxb-*` token.'
        self.base_url = base_url
        "A string representing the Slack API base URL.\n        Default is `'https://www.slack.com/api/'`."
        self.timeout = timeout
        'The maximum number of seconds the client will wait\n        to connect and receive a response from Slack.\n        Default is 30 seconds.'
        self.ssl = ssl
        'An [`ssl.SSLContext`](https://docs.python.org/3/library/ssl.html#ssl.SSLContext)\n        instance, helpful for specifying your own custom\n        certificate chain.'
        self.proxy = proxy
        'String representing a fully-qualified URL to a proxy through which\n        to route all requests to the Slack API. Even if this parameter\n        is not specified, if any of the following environment variables are\n        present, they will be loaded into this parameter: `HTTPS_PROXY`,\n        `https_proxy`, `HTTP_PROXY` or `http_proxy`.'
        self.run_async = run_async
        self.use_sync_aiohttp = use_sync_aiohttp
        self.session = session
        self.headers = headers or {}
        '`dict` representing additional request headers to attach to all requests.'
        self.headers['User-Agent'] = get_user_agent(user_agent_prefix, user_agent_suffix)
        self.default_params = {}
        if team_id is not None:
            self.default_params['team_id'] = team_id
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        if self.proxy is None or len(self.proxy.strip()) == 0:
            env_variable = load_http_proxy_from_env(self._logger)
            if env_variable is not None:
                self.proxy = env_variable
        self._event_loop = loop

    def api_call(self, api_method: str, *, http_verb: str='POST', files: Optional[dict]=None, data: Union[dict, FormData]=None, params: Optional[dict]=None, json: Optional[dict]=None, headers: Optional[dict]=None, auth: Optional[dict]=None) -> Union[asyncio.Future, SlackResponse]:
        if False:
            i = 10
            return i + 15
        "Create a request and execute the API call to Slack.\n        Args:\n            api_method (str): The target Slack API method.\n                e.g. 'chat.postMessage'\n            http_verb (str): HTTP Verb. e.g. 'POST'\n            files (dict): Files to multipart upload.\n                e.g. {image OR file: file_object OR file_path}\n            data: The body to attach to the request. If a dictionary is\n                provided, form-encoding will take place.\n                e.g. {'key1': 'value1', 'key2': 'value2'}\n            params (dict): The URL parameters to append to the URL.\n                e.g. {'key1': 'value1', 'key2': 'value2'}\n            json (dict): JSON for the body to attach to the request\n                (if files or data is not specified).\n                e.g. {'key1': 'value1', 'key2': 'value2'}\n            headers (dict): Additional request headers\n            auth (dict): A dictionary that consists of client_id and client_secret\n        Returns:\n            (SlackResponse)\n                The server's response to an HTTP request. Data\n                from the response can be accessed like a dict.\n                If the response included 'next_cursor' it can\n                be iterated on to execute subsequent requests.\n        Raises:\n            SlackApiError: The following Slack API call failed:\n                'chat.postMessage'.\n            SlackRequestError: Json data can only be submitted as\n                POST requests.\n        "
        api_url = _get_url(self.base_url, api_method)
        headers = headers or {}
        headers.update(self.headers)
        if auth is not None:
            if isinstance(auth, dict):
                auth = BasicAuth(auth['client_id'], auth['client_secret'])
            elif isinstance(auth, BasicAuth):
                headers['Authorization'] = auth.encode()
        req_args = _build_req_args(token=self.token, http_verb=http_verb, files=files, data=data, default_params=self.default_params, params=params, json=json, headers=headers, auth=auth, ssl=self.ssl, proxy=self.proxy)
        show_deprecation_warning_if_any(api_method)
        if self.run_async or self.use_sync_aiohttp:
            if self._event_loop is None:
                self._event_loop = _get_event_loop()
            future = asyncio.ensure_future(self._send(http_verb=http_verb, api_url=api_url, req_args=req_args), loop=self._event_loop)
            if self.run_async:
                return future
            if self.use_sync_aiohttp:
                return self._event_loop.run_until_complete(future)
        return self._sync_send(api_url=api_url, req_args=req_args)

    async def _send(self, http_verb: str, api_url: str, req_args: dict) -> SlackResponse:
        """Sends the request out for transmission.
        Args:
            http_verb (str): The HTTP verb. e.g. 'GET' or 'POST'.
            api_url (str): The Slack API url. e.g. 'https://slack.com/api/chat.postMessage'
            req_args (dict): The request arguments to be attached to the request.
            e.g.
            {
                json: {
                    'attachments': [{"pretext": "pre-hello", "text": "text-world"}],
                    'channel': '#random'
                }
            }
        Returns:
            The response parsed into a SlackResponse object.
        """
        open_files = _files_to_data(req_args)
        try:
            if 'params' in req_args:
                req_args['params'] = convert_bool_to_0_or_1(req_args['params'])
            res = await self._request(http_verb=http_verb, api_url=api_url, req_args=req_args)
        finally:
            for f in open_files:
                f.close()
        data = {'client': self, 'http_verb': http_verb, 'api_url': api_url, 'req_args': req_args, 'use_sync_aiohttp': self.use_sync_aiohttp}
        return SlackResponse(**{**data, **res}).validate()

    async def _request(self, *, http_verb, api_url, req_args) -> Dict[str, Any]:
        """Submit the HTTP request with the running session or a new session.
        Returns:
            A dictionary of the response data.
        """
        return await _request_with_session(current_session=self.session, timeout=self.timeout, logger=self._logger, http_verb=http_verb, api_url=api_url, req_args=req_args)

    def _sync_send(self, api_url, req_args) -> SlackResponse:
        if False:
            print('Hello World!')
        params = req_args['params'] if 'params' in req_args else None
        data = req_args['data'] if 'data' in req_args else None
        files = req_args['files'] if 'files' in req_args else None
        _json = req_args['json'] if 'json' in req_args else None
        headers = req_args['headers'] if 'headers' in req_args else None
        token = params.get('token') if params and 'token' in params else None
        auth = req_args['auth'] if 'auth' in req_args else None
        if auth is not None:
            headers = {}
            if isinstance(auth, BasicAuth):
                headers['Authorization'] = auth.encode()
            elif isinstance(auth, str):
                headers['Authorization'] = auth
            else:
                self._logger.warning(f'As the auth: {auth}: {type(auth)} is unsupported, skipped')
        body_params = {}
        if params:
            body_params.update(params)
        if data:
            body_params.update(data)
        return self._urllib_api_call(token=token, url=api_url, query_params={}, body_params=body_params, files=files, json_body=_json, additional_headers=headers)

    def _request_for_pagination(self, api_url: str, req_args: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "This method is supposed to be used only for SlackResponse pagination\n        You can paginate using Python's for iterator as below:\n          for response in client.conversations_list(limit=100):\n              # do something with each response here\n        "
        response = self._perform_urllib_http_request(url=api_url, args=req_args)
        return {'status_code': int(response['status']), 'headers': dict(response['headers']), 'data': json.loads(response['body'])}

    def _urllib_api_call(self, *, token: Optional[str]=None, url: str, query_params: Dict[str, str], json_body: Dict, body_params: Dict[str, str], files: Dict[str, io.BytesIO], additional_headers: Dict[str, str]) -> SlackResponse:
        if False:
            return 10
        "Performs a Slack API request and returns the result.\n\n        Args:\n            token: Slack API Token (either bot token or user token)\n            url: Complete URL (e.g., https://www.slack.com/api/chat.postMessage)\n            query_params: Query string\n            json_body: JSON data structure (it's still a dict at this point),\n                if you give this argument, body_params and files will be skipped\n            body_params: Form body params\n            files: Files to upload\n            additional_headers: Request headers to append\n        Returns:\n            API response\n        "
        files_to_close: List[BinaryIO] = []
        try:
            query_params = convert_bool_to_0_or_1(query_params)
            body_params = convert_bool_to_0_or_1(body_params)
            if self._logger.level <= logging.DEBUG:

                def convert_params(values: dict) -> dict:
                    if False:
                        return 10
                    if not values or not isinstance(values, dict):
                        return {}
                    return {k: '(bytes)' if isinstance(v, bytes) else v for (k, v) in values.items()}
                headers = {k: '(redacted)' if k.lower() == 'authorization' else v for (k, v) in additional_headers.items()}
                self._logger.debug(f'Sending a request - url: {url}, query_params: {convert_params(query_params)}, body_params: {convert_params(body_params)}, files: {convert_params(files)}, json_body: {json_body}, headers: {headers}')
            request_data = {}
            if files is not None and isinstance(files, dict) and (len(files) > 0):
                if body_params:
                    for (k, v) in body_params.items():
                        request_data.update({k: v})
                for (k, v) in files.items():
                    if isinstance(v, str):
                        f: BinaryIO = open(v.encode('utf-8', 'ignore'), 'rb')
                        files_to_close.append(f)
                        request_data.update({k: f})
                    elif isinstance(v, (bytearray, bytes)):
                        request_data.update({k: io.BytesIO(v)})
                    else:
                        request_data.update({k: v})
            request_headers = self._build_urllib_request_headers(token=token or self.token, has_json=json is not None, has_files=files is not None, additional_headers=additional_headers)
            request_args = {'headers': request_headers, 'data': request_data, 'params': body_params, 'files': files, 'json': json_body}
            if query_params:
                q = urlencode(query_params)
                url = f'{url}&{q}' if '?' in url else f'{url}?{q}'
            response = self._perform_urllib_http_request(url=url, args=request_args)
            body = response.get('body', None)
            response_body_data: Optional[Union[dict, bytes]] = body
            if body is not None and (not isinstance(body, bytes)):
                try:
                    response_body_data = json.loads(response['body'])
                except json.decoder.JSONDecodeError:
                    message = _build_unexpected_body_error_message(response.get('body', ''))
                    raise err.SlackApiError(message, response)
            all_params: Dict[str, Any] = copy.copy(body_params) if body_params is not None else {}
            if query_params:
                all_params.update(query_params)
            request_args['params'] = all_params
            return SlackResponse(client=self, http_verb='POST', api_url=url, req_args=request_args, data=response_body_data, headers=dict(response['headers']), status_code=response['status'], use_sync_aiohttp=False).validate()
        finally:
            for f in files_to_close:
                if not f.closed:
                    f.close()

    def _perform_urllib_http_request(self, *, url: str, args: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Performs an HTTP request and parses the response.\n\n        Args:\n            url: Complete URL (e.g., https://www.slack.com/api/chat.postMessage)\n            args: args has "headers", "data", "params", and "json"\n                "headers": Dict[str, str]\n                "data": Dict[str, Any]\n                "params": Dict[str, str],\n                "json": Dict[str, Any],\n\n        Returns:\n            dict {status: int, headers: Headers, body: str}\n        '
        headers = args['headers']
        if args['json']:
            body = json.dumps(args['json'])
            headers['Content-Type'] = 'application/json;charset=utf-8'
        elif args['data']:
            boundary = f'--------------{uuid.uuid4()}'
            sep_boundary = b'\r\n--' + boundary.encode('ascii')
            end_boundary = sep_boundary + b'--\r\n'
            body = io.BytesIO()
            data = args['data']
            for (key, value) in data.items():
                readable = getattr(value, 'readable', None)
                if readable and value.readable():
                    filename = 'Uploaded file'
                    name_attr = getattr(value, 'name', None)
                    if name_attr:
                        filename = name_attr.decode('utf-8') if isinstance(name_attr, bytes) else name_attr
                    if 'filename' in data:
                        filename = data['filename']
                    mimetype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
                    title = f'\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\n' + f'Content-Type: {mimetype}\r\n'
                    value = value.read()
                else:
                    title = f'\r\nContent-Disposition: form-data; name="{key}"\r\n'
                    value = str(value).encode('utf-8')
                body.write(sep_boundary)
                body.write(title.encode('utf-8'))
                body.write(b'\r\n')
                body.write(value)
            body.write(end_boundary)
            body = body.getvalue()
            headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
            headers['Content-Length'] = len(body)
        elif args['params']:
            body = urlencode(args['params'])
            headers['Content-Type'] = 'application/x-www-form-urlencoded'
        else:
            body = None
        if isinstance(body, str):
            body = body.encode('utf-8')
        try:
            if url.lower().startswith('http'):
                req = Request(method='POST', url=url, data=body, headers=headers)
                opener: Optional[OpenerDirector] = None
                if self.proxy is not None:
                    if isinstance(self.proxy, str):
                        opener = urllib.request.build_opener(ProxyHandler({'http': self.proxy, 'https': self.proxy}), HTTPSHandler(context=self.ssl))
                    else:
                        raise SlackRequestError(f'Invalid proxy detected: {self.proxy} must be a str value')
                resp: Optional[HTTPResponse] = None
                if opener:
                    resp = opener.open(req, timeout=self.timeout)
                else:
                    resp = urlopen(req, context=self.ssl, timeout=self.timeout)
                if resp.headers.get_content_type() == 'application/gzip':
                    body: bytes = resp.read()
                    return {'status': resp.code, 'headers': resp.headers, 'body': body}
                charset = resp.headers.get_content_charset() or 'utf-8'
                body: str = resp.read().decode(charset)
                return {'status': resp.code, 'headers': resp.headers, 'body': body}
            raise SlackRequestError(f'Invalid URL detected: {url}')
        except HTTPError as e:
            response_headers = dict(e.headers.items())
            resp = {'status': e.code, 'headers': response_headers}
            if e.code == 429:
                if 'retry-after' not in response_headers and 'Retry-After' in response_headers:
                    response_headers['retry-after'] = response_headers['Retry-After']
                if 'Retry-After' not in response_headers and 'retry-after' in response_headers:
                    response_headers['Retry-After'] = response_headers['retry-after']
            charset = e.headers.get_content_charset() or 'utf-8'
            body: str = e.read().decode(charset)
            resp['body'] = body
            return resp
        except Exception as err:
            self._logger.error(f'Failed to send a request to Slack API server: {err}')
            raise err

    def _build_urllib_request_headers(self, token: str, has_json: bool, has_files: bool, additional_headers: dict) -> Dict[str, str]:
        if False:
            print('Hello World!')
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        headers.update(self.headers)
        if token:
            headers.update({'Authorization': 'Bearer {}'.format(token)})
        if additional_headers:
            headers.update(additional_headers)
        if has_json:
            headers.update({'Content-Type': 'application/json;charset=utf-8'})
        if has_files:
            headers.pop('Content-Type', None)
        return headers

    @staticmethod
    def validate_slack_signature(*, signing_secret: str, data: str, timestamp: str, signature: str) -> bool:
        if False:
            while True:
                i = 10
        "\n        Slack creates a unique string for your app and shares it with you. Verify\n        requests from Slack with confidence by verifying signatures using your\n        signing secret.\n        On each HTTP request that Slack sends, we add an X-Slack-Signature HTTP\n        header. The signature is created by combining the signing secret with the\n        body of the request we're sending using a standard HMAC-SHA256 keyed hash.\n        https://api.slack.com/docs/verifying-requests-from-slack#how_to_make_a_request_signature_in_4_easy_steps__an_overview\n        Args:\n            signing_secret: Your application's signing secret, available in the\n                Slack API dashboard\n            data: The raw body of the incoming request - no headers, just the body.\n            timestamp: from the 'X-Slack-Request-Timestamp' header\n            signature: from the 'X-Slack-Signature' header - the calculated signature\n                should match this.\n        Returns:\n            True if signatures matches\n        "
        warnings.warn('As this method is deprecated since slackclient 2.6.0, use `from slack.signature import SignatureVerifier` instead', DeprecationWarning)
        format_req = str.encode(f'v0:{timestamp}:{data}')
        encoded_secret = str.encode(signing_secret)
        request_hash = hmac.new(encoded_secret, format_req, hashlib.sha256).hexdigest()
        calculated_signature = f'v0={request_hash}'
        return hmac.compare_digest(calculated_signature, signature)