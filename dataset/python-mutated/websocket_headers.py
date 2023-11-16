from typing import Dict, Optional
from streamlit import runtime
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web.server.browser_websocket_handler import BrowserWebSocketHandler

@gather_metrics('_get_websocket_headers')
def _get_websocket_headers() -> Optional[Dict[str, str]]:
    if False:
        print('Hello World!')
    "Return a copy of the HTTP request headers for the current session's\n    WebSocket connection. If there's no active session, return None instead.\n\n    Raise an error if the server is not running.\n\n    Note to the intrepid: this is an UNSUPPORTED, INTERNAL API. (We don't have plans\n    to remove it without a replacement, but we don't consider this a production-ready\n    function, and its signature may change without a deprecation warning.)\n    "
    ctx = get_script_run_ctx()
    if ctx is None:
        return None
    session_client = runtime.get_instance().get_client(ctx.session_id)
    if session_client is None:
        return None
    if not isinstance(session_client, BrowserWebSocketHandler):
        raise RuntimeError(f'SessionClient is not a BrowserWebSocketHandler! ({session_client})')
    return dict(session_client.request.headers)