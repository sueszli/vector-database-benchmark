import base64
import binascii
import json
from typing import Any, Awaitable, Dict, List, Optional, Union
import tornado.concurrent
import tornado.locks
import tornado.netutil
import tornado.web
import tornado.websocket
from tornado.websocket import WebSocketHandler
from typing_extensions import Final
from streamlit import config
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import Runtime, SessionClient, SessionClientDisconnectedError
from streamlit.runtime.runtime_util import serialize_forward_msg
from streamlit.web.server.server_util import is_url_from_allowed_origins
_LOGGER: Final = get_logger(__name__)

class BrowserWebSocketHandler(WebSocketHandler, SessionClient):
    """Handles a WebSocket connection from the browser"""

    def initialize(self, runtime: Runtime) -> None:
        if False:
            while True:
                i = 10
        self._runtime = runtime
        self._session_id: Optional[str] = None
        if config.get_option('server.enableXsrfProtection'):
            _ = self.xsrf_token

    def check_origin(self, origin: str) -> bool:
        if False:
            print('Hello World!')
        'Set up CORS.'
        return super().check_origin(origin) or is_url_from_allowed_origins(origin)

    def write_forward_msg(self, msg: ForwardMsg) -> None:
        if False:
            i = 10
            return i + 15
        'Send a ForwardMsg to the browser.'
        try:
            self.write_message(serialize_forward_msg(msg), binary=True)
        except tornado.websocket.WebSocketClosedError as e:
            raise SessionClientDisconnectedError from e

    def select_subprotocol(self, subprotocols: List[str]) -> Optional[str]:
        if False:
            return 10
        'Return the first subprotocol in the given list.\n\n        This method is used by Tornado to select a protocol when the\n        Sec-WebSocket-Protocol header is set in an HTTP Upgrade request.\n\n        NOTE: We repurpose the Sec-WebSocket-Protocol header here in a slightly\n        unfortunate (but necessary) way. The browser WebSocket API doesn\'t allow us to\n        set arbitrary HTTP headers, and this header is the only one where we have the\n        ability to set it to arbitrary values, so we use it to pass tokens (in this\n        case, the previous session ID to allow us to reconnect to it) from client to\n        server as the *third* value in the list.\n\n        The reason why the auth token is set as the third value is that:\n          * when Sec-WebSocket-Protocol is set, many clients expect the server to\n            respond with a selected subprotocol to use. We don\'t want that reply to be\n            the session token, so we by convention have the client always set the first\n            protocol to "streamlit" and select that.\n          * the second protocol in the list is reserved in some deployment environments\n            for an auth token that we currently don\'t use\n        '
        if subprotocols:
            return subprotocols[0]
        return None

    def open(self, *args, **kwargs) -> Optional[Awaitable[None]]:
        if False:
            print('Hello World!')
        is_public_cloud_app = False
        try:
            header_content = self.request.headers['X-Streamlit-User']
            payload = base64.b64decode(header_content)
            user_obj = json.loads(payload)
            email = user_obj['email']
            is_public_cloud_app = user_obj['isPublicCloudApp']
        except (KeyError, binascii.Error, json.decoder.JSONDecodeError):
            email = 'test@example.com'
        user_info: Dict[str, Optional[str]] = dict()
        if is_public_cloud_app:
            user_info['email'] = None
        else:
            user_info['email'] = email
        existing_session_id = None
        try:
            ws_protocols = [p.strip() for p in self.request.headers['Sec-Websocket-Protocol'].split(',')]
            if len(ws_protocols) >= 3:
                existing_session_id = ws_protocols[2]
        except KeyError:
            pass
        self._session_id = self._runtime.connect_session(client=self, user_info=user_info, existing_session_id=existing_session_id)
        return None

    def on_close(self) -> None:
        if False:
            print('Hello World!')
        if not self._session_id:
            return
        self._runtime.disconnect_session(self._session_id)
        self._session_id = None

    def get_compression_options(self) -> Optional[Dict[Any, Any]]:
        if False:
            return 10
        'Enable WebSocket compression.\n\n        Returning an empty dict enables websocket compression. Returning\n        None disables it.\n\n        (See the docstring in the parent class.)\n        '
        if config.get_option('server.enableWebsocketCompression'):
            return {}
        return None

    def on_message(self, payload: Union[str, bytes]) -> None:
        if False:
            i = 10
            return i + 15
        if not self._session_id:
            return
        try:
            if isinstance(payload, str):
                raise RuntimeError('WebSocket received an unexpected `str` message. (We expect `bytes` only.)')
            msg = BackMsg()
            msg.ParseFromString(payload)
            _LOGGER.debug('Received the following back message:\n%s', msg)
        except Exception as ex:
            _LOGGER.error(ex)
            self._runtime.handle_backmsg_deserialization_exception(self._session_id, ex)
            return
        if msg.WhichOneof('type') == 'debug_disconnect_websocket':
            if config.get_option('global.developmentMode'):
                self.close()
            else:
                _LOGGER.warning('Client tried to disconnect websocket when not in development mode.')
        elif msg.WhichOneof('type') == 'debug_shutdown_runtime':
            if config.get_option('global.developmentMode'):
                self._runtime.stop()
            else:
                _LOGGER.warning('Client tried to shut down runtime when not in development mode.')
        else:
            self._runtime.handle_backmsg(self._session_id, msg)