import sys
import threading
from typing import Any, Callable, Optional, TYPE_CHECKING
from UM.Logger import Logger
got_server_type = False
try:
    from cura.OAuth2.AuthorizationRequestServer import AuthorizationRequestServer
    from cura.OAuth2.AuthorizationRequestHandler import AuthorizationRequestHandler
    got_server_type = True
except PermissionError:
    Logger.error("Can't start a server due to a PermissionError when starting the http.server.")
if TYPE_CHECKING:
    from cura.OAuth2.Models import AuthenticationResponse
    from cura.OAuth2.AuthorizationHelpers import AuthorizationHelpers

class LocalAuthorizationServer:

    def __init__(self, auth_helpers: 'AuthorizationHelpers', auth_state_changed_callback: Callable[['AuthenticationResponse'], Any], daemon: bool) -> None:
        if False:
            print('Hello World!')
        'The local LocalAuthorizationServer takes care of the oauth2 callbacks.\n\n        Once the flow is completed, this server should be closed down again by calling\n        :py:meth:`cura.OAuth2.LocalAuthorizationServer.LocalAuthorizationServer.stop()`\n\n        :param auth_helpers: An instance of the authorization helpers class.\n        :param auth_state_changed_callback: A callback function to be called when the authorization state changes.\n        :param daemon: Whether the server thread should be run in daemon mode.\n\n        .. note::\n\n            Daemon threads are abruptly stopped at shutdown. Their resources (e.g. open files) may never be released.\n        '
        self._web_server = None
        self._web_server_thread = None
        self._web_server_port = auth_helpers.settings.CALLBACK_PORT
        self._auth_helpers = auth_helpers
        self._auth_state_changed_callback = auth_state_changed_callback
        self._daemon = daemon

    def start(self, verification_code: str, state: str) -> None:
        if False:
            i = 10
            return i + 15
        'Starts the local web server to handle the authorization callback.\n\n        :param verification_code: The verification code part of the OAuth2 client identification.\n        :param state: The unique state code (to ensure that the request we get back is really from the server.\n        '
        if self._web_server:
            Logger.log('d', 'Auth web server was already running. Updating the verification code')
            self._web_server.setVerificationCode(verification_code)
            return
        if self._web_server_port is None:
            raise Exception('Unable to start server without specifying the port.')
        Logger.log('d', 'Starting local web server to handle authorization callback on port %s', self._web_server_port)
        if got_server_type:
            self._web_server = AuthorizationRequestServer(('0.0.0.0', self._web_server_port), AuthorizationRequestHandler)
            self._web_server.setAuthorizationHelpers(self._auth_helpers)
            self._web_server.setAuthorizationCallback(self._auth_state_changed_callback)
            self._web_server.setVerificationCode(verification_code)
            self._web_server.setState(state)
            self._web_server_thread = threading.Thread(None, self._serve_forever, daemon=self._daemon)
            self._web_server_thread.start()

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        'Stops the web server if it was running. It also does some cleanup.'
        Logger.log('d', 'Stopping local oauth2 web server...')
        if self._web_server:
            try:
                self._web_server.shutdown()
            except OSError:
                pass
        Logger.log('d', 'Local oauth2 web server was shut down')
        self._web_server = None
        self._web_server_thread = None

    def _serve_forever(self) -> None:
        if False:
            print('Hello World!')
        '\n        If the platform is windows, this function calls the serve_forever function of the _web_server, catching any\n        OSErrors that may occur in the thread, thus making the reported message more log-friendly.\n        If it is any other platform, it just calls the serve_forever function immediately.\n\n        :return: None\n        '
        Logger.log('d', 'Local web server for authorization has started')
        if self._web_server:
            if sys.platform == 'win32':
                try:
                    self._web_server.serve_forever()
                except OSError:
                    Logger.logException('w', 'An exception happened while serving the auth server')
            else:
                self._web_server.serve_forever()