from http.server import HTTPServer
from socketserver import ThreadingMixIn
from typing import Callable, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from cura.OAuth2.Models import AuthenticationResponse
    from cura.OAuth2.AuthorizationHelpers import AuthorizationHelpers

class AuthorizationRequestServer(ThreadingMixIn, HTTPServer):
    """The authorization request callback handler server.

    This subclass is needed to be able to pass some data to the request handler. This cannot be done on the request
    handler directly as the HTTPServer creates an instance of the handler after init.
    """

    def setAuthorizationHelpers(self, authorization_helpers: 'AuthorizationHelpers') -> None:
        if False:
            i = 10
            return i + 15
        'Set the authorization helpers instance on the request handler.'
        self.RequestHandlerClass.authorization_helpers = authorization_helpers

    def setAuthorizationCallback(self, authorization_callback: Callable[['AuthenticationResponse'], Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the authorization callback on the request handler.'
        self.RequestHandlerClass.authorization_callback = authorization_callback

    def setVerificationCode(self, verification_code: str) -> None:
        if False:
            i = 10
            return i + 15
        'Set the verification code on the request handler.'
        self.RequestHandlerClass.verification_code = verification_code

    def setState(self, state: str) -> None:
        if False:
            return 10
        self.RequestHandlerClass.state = state