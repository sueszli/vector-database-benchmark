from typing import Callable, Optional
from urllib.parse import urlparse
from websocket import WebSocketApp
from lightning.app.utilities.auth import _AuthTokenGetter

class _LightningLogsSocketAPI(_AuthTokenGetter):

    @staticmethod
    def _app_logs_socket_url(host: str, project_id: str, app_id: str, token: str, component: str) -> str:
        if False:
            while True:
                i = 10
        return f'wss://{host}/v1/projects/{project_id}/appinstances/{app_id}/logs?token={token}&component={component}&follow=true'

    def create_lightning_logs_socket(self, project_id: str, app_id: str, component: str, on_message_callback: Callable[[WebSocketApp, str], None], on_error_callback: Optional[Callable[[Exception, str], None]]=None) -> WebSocketApp:
        if False:
            for i in range(10):
                print('nop')
        'Creates and returns WebSocketApp to listen to lightning app logs.\n\n            .. code-block:: python\n                # Synchronous reading, run_forever() is blocking\n\n\n                def print_log_msg(ws_app, msg):\n                    print(msg)\n\n\n                flow_logs_socket = client.create_lightning_logs_socket("project_id", "app_id", "flow", print_log_msg)\n                flow_socket.run_forever()\n\n            .. code-block:: python\n                # Asynchronous reading (with Threads)\n\n\n                def print_log_msg(ws_app, msg):\n                    print(msg)\n\n\n                flow_logs_socket = client.create_lightning_logs_socket("project_id", "app_id", "flow", print_log_msg)\n                work_logs_socket = client.create_lightning_logs_socket("project_id", "app_id", "work_1", print_log_msg)\n\n                flow_logs_thread = Thread(target=flow_logs_socket.run_forever)\n                work_logs_thread = Thread(target=work_logs_socket.run_forever)\n\n                flow_logs_thread.start()\n                work_logs_thread.start()\n                # .......\n\n                flow_logs_socket.close()\n                work_logs_thread.close()\n\n        Arguments:\n            project_id: Project ID.\n            app_id: Application ID.\n            component: Component name eg flow.\n            on_message_callback: Callback object which is called when received data.\n            on_error_callback: Callback object which is called when we get error.\n\n        Returns:\n            WebSocketApp of the wanted socket\n\n        '
        _token = self._get_api_token()
        clean_ws_host = urlparse(self.api_client.configuration.host).netloc
        socket_url = self._app_logs_socket_url(host=clean_ws_host, project_id=project_id, app_id=app_id, token=_token, component=component)
        return WebSocketApp(socket_url, on_message=on_message_callback, on_error=on_error_callback)