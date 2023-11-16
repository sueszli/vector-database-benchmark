from http.server import BaseHTTPRequestHandler
from threading import Lock
from typing import Optional, Callable, Tuple, Dict, Any, List, TYPE_CHECKING
from urllib.parse import parse_qs, urlparse
from UM.Logger import Logger
from cura.OAuth2.Models import AuthenticationResponse, ResponseData, HTTP_STATUS
from UM.i18n import i18nCatalog
if TYPE_CHECKING:
    from cura.OAuth2.Models import ResponseStatus
    from cura.OAuth2.AuthorizationHelpers import AuthorizationHelpers
catalog = i18nCatalog('cura')

class AuthorizationRequestHandler(BaseHTTPRequestHandler):
    """This handler handles all HTTP requests on the local web server.

    It also requests the access token for the 2nd stage of the OAuth flow.
    """

    def __init__(self, request, client_address, server) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(request, client_address, server)
        self.authorization_helpers: Optional[AuthorizationHelpers] = None
        self.authorization_callback: Optional[Callable[[AuthenticationResponse], None]] = None
        self.verification_code: Optional[str] = None
        self.state: Optional[str] = None

    def do_HEAD(self) -> None:
        if False:
            return 10
        self.do_GET()

    def do_GET(self) -> None:
        if False:
            print('Hello World!')
        parsed_url = urlparse(self.path)
        query = parse_qs(parsed_url.query)
        if parsed_url.path == '/callback':
            (server_response, token_response) = self._handleCallback(query)
        else:
            server_response = self._handleNotFound()
            token_response = None
        self._sendHeaders(server_response.status, server_response.content_type, server_response.redirect_uri)
        if server_response.data_stream:
            self._sendData(server_response.data_stream)
        if token_response and self.authorization_callback is not None:
            self.authorization_callback(token_response)

    def _handleCallback(self, query: Dict[Any, List]) -> Tuple[ResponseData, Optional[AuthenticationResponse]]:
        if False:
            for i in range(10):
                print('nop')
        'Handler for the callback URL redirect.\n\n        :param query: Dict containing the HTTP query parameters.\n        :return: HTTP ResponseData containing a success page to show to the user.\n        '
        code = self._queryGet(query, 'code')
        state = self._queryGet(query, 'state')
        if state != self.state:
            Logger.log('w', f'The provided state was not correct. Got {state} and expected {self.state}')
            token_response = AuthenticationResponse(success=False, err_message=catalog.i18nc('@message', 'The provided state is not correct.'))
        elif code and self.authorization_helpers is not None and (self.verification_code is not None):
            Logger.log('d', 'Timeout when authenticating with the account server.')
            token_response = AuthenticationResponse(success=False, err_message=catalog.i18nc('@message', 'Timeout when authenticating with the account server.'))
            lock = Lock()
            lock.acquire()

            def callback(response: AuthenticationResponse) -> None:
                if False:
                    i = 10
                    return i + 15
                nonlocal token_response
                token_response = response
                lock.release()
            self.authorization_helpers.getAccessTokenUsingAuthorizationCode(code, self.verification_code, callback)
            lock.acquire(timeout=60)
        elif self._queryGet(query, 'error_code') == 'user_denied':
            Logger.log('d', 'User did not give the required permission when authorizing this application')
            token_response = AuthenticationResponse(success=False, err_message=catalog.i18nc('@message', 'Please give the required permissions when authorizing this application.'))
        else:
            Logger.log('w', f"Unexpected error when logging in. Error_code: {self._queryGet(query, 'error_code')}, State: {state}")
            token_response = AuthenticationResponse(success=False, error_message=catalog.i18nc('@message', 'Something unexpected happened when trying to log in, please try again.'))
        if self.authorization_helpers is None:
            return (ResponseData(), token_response)
        return (ResponseData(status=HTTP_STATUS['REDIRECT'], data_stream=b'Redirecting...', redirect_uri=self.authorization_helpers.settings.AUTH_SUCCESS_REDIRECT if token_response.success else self.authorization_helpers.settings.AUTH_FAILED_REDIRECT), token_response)

    @staticmethod
    def _handleNotFound() -> ResponseData:
        if False:
            return 10
        'Handle all other non-existing server calls.'
        return ResponseData(status=HTTP_STATUS['NOT_FOUND'], content_type='text/html', data_stream=b'Not found.')

    def _sendHeaders(self, status: 'ResponseStatus', content_type: str, redirect_uri: str=None) -> None:
        if False:
            while True:
                i = 10
        self.send_response(status.code, status.message)
        self.send_header('Content-type', content_type)
        if redirect_uri:
            self.send_header('Location', redirect_uri)
        self.end_headers()

    def _sendData(self, data: bytes) -> None:
        if False:
            while True:
                i = 10
        self.wfile.write(data)

    @staticmethod
    def _queryGet(query_data: Dict[Any, List], key: str, default: Optional[str]=None) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Convenience helper for getting values from a pre-parsed query string'
        return query_data.get(key, [default])[0]