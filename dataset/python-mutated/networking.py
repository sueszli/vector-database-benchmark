"""
Defines helper methods useful for setting up ports, launching servers, and
creating tunnels.
"""
from __future__ import annotations
import os
import socket
import threading
import time
import warnings
from functools import partial
from typing import TYPE_CHECKING
import requests
import uvicorn
from uvicorn.config import Config
from gradio.exceptions import ServerFailedToStartError
from gradio.routes import App
from gradio.tunneling import Tunnel
from gradio.utils import SourceFileReloader, watchfn
if TYPE_CHECKING:
    from gradio.blocks import Blocks
INITIAL_PORT_VALUE = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
TRY_NUM_PORTS = int(os.getenv('GRADIO_NUM_PORTS', '100'))
LOCALHOST_NAME = os.getenv('GRADIO_SERVER_NAME', '127.0.0.1')
GRADIO_API_SERVER = 'https://api.gradio.app/v2/tunnel-request'
should_watch = bool(os.getenv('GRADIO_WATCH_DIRS', False))
GRADIO_WATCH_DIRS = os.getenv('GRADIO_WATCH_DIRS', '').split(',') if should_watch else []
GRADIO_WATCH_FILE = os.getenv('GRADIO_WATCH_FILE', 'app')
GRADIO_WATCH_DEMO_NAME = os.getenv('GRADIO_WATCH_DEMO_NAME', 'demo')

class Server(uvicorn.Server):

    def __init__(self, config: Config, reloader: SourceFileReloader | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.running_app = config.app
        super().__init__(config)
        self.reloader = reloader
        if self.reloader:
            self.event = threading.Event()
            self.watch = partial(watchfn, self.reloader)

    def install_signal_handlers(self):
        if False:
            print('Hello World!')
        pass

    def run_in_thread(self):
        if False:
            while True:
                i = 10
        self.thread = threading.Thread(target=self.run, daemon=True)
        if self.reloader:
            self.watch_thread = threading.Thread(target=self.watch, daemon=True)
            self.watch_thread.start()
        self.thread.start()
        start = time.time()
        while not self.started:
            time.sleep(0.001)
            if time.time() - start > 5:
                raise ServerFailedToStartError('Server failed to start. Please check that the port is available.')

    def close(self):
        if False:
            return 10
        self.should_exit = True
        if self.reloader:
            self.reloader.stop()
            self.watch_thread.join()
        self.thread.join()

def get_first_available_port(initial: int, final: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Gets the first open port in a specified range of port numbers\n    Parameters:\n    initial: the initial value in the range of port numbers\n    final: final (exclusive) value in the range of port numbers, should be greater than `initial`\n    Returns:\n    port: the first open port in the range\n    '
    for port in range(initial, final):
        try:
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((LOCALHOST_NAME, port))
            s.close()
            return port
        except OSError:
            pass
    raise OSError(f'All ports from {initial} to {final - 1} are in use. Please close a port.')

def configure_app(app: App, blocks: Blocks) -> App:
    if False:
        print('Hello World!')
    auth = blocks.auth
    if auth is not None:
        if not callable(auth):
            app.auth = {account[0]: account[1] for account in auth}
        else:
            app.auth = auth
    else:
        app.auth = None
    app.blocks = blocks
    app.cwd = os.getcwd()
    app.favicon_path = blocks.favicon_path
    app.tokens = {}
    return app

def start_server(blocks: Blocks, server_name: str | None=None, server_port: int | None=None, ssl_keyfile: str | None=None, ssl_certfile: str | None=None, ssl_keyfile_password: str | None=None, app_kwargs: dict | None=None) -> tuple[str, int, str, App, Server]:
    if False:
        return 10
    'Launches a local server running the provided Interface\n    Parameters:\n        blocks: The Blocks object to run on the server\n        server_name: to make app accessible on local network, set this to "0.0.0.0". Can be set by environment variable GRADIO_SERVER_NAME.\n        server_port: will start gradio app on this port (if available). Can be set by environment variable GRADIO_SERVER_PORT.\n        auth: If provided, username and password (or list of username-password tuples) required to access the Blocks. Can also provide function that takes username and password and returns True if valid login.\n        ssl_keyfile: If a path to a file is provided, will use this as the private key file to create a local server running on https.\n        ssl_certfile: If a path to a file is provided, will use this as the signed certificate for https. Needs to be provided if ssl_keyfile is provided.\n        ssl_keyfile_password: If a password is provided, will use this with the ssl certificate for https.\n        app_kwargs: Additional keyword arguments to pass to the gradio.routes.App constructor.\n\n    Returns:\n        port: the port number the server is running on\n        path_to_local_server: the complete address that the local server can be accessed at\n        app: the FastAPI app object\n        server: the server object that is a subclass of uvicorn.Server (used to close the server)\n    '
    if ssl_keyfile is not None and ssl_certfile is None:
        raise ValueError('ssl_certfile must be provided if ssl_keyfile is provided.')
    server_name = server_name or LOCALHOST_NAME
    url_host_name = 'localhost' if server_name == '0.0.0.0' else server_name
    if server_name.startswith('[') and server_name.endswith(']'):
        host = server_name[1:-1]
    else:
        host = server_name
    app = App.create_app(blocks, app_kwargs=app_kwargs)
    server_ports = [server_port] if server_port is not None else range(INITIAL_PORT_VALUE, INITIAL_PORT_VALUE + TRY_NUM_PORTS)
    for port in server_ports:
        try:
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((LOCALHOST_NAME, port))
            s.close()
            config = uvicorn.Config(app=app, port=port, host=host, log_level='warning', ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile, ssl_keyfile_password=ssl_keyfile_password)
            reloader = None
            if GRADIO_WATCH_DIRS:
                change_event = threading.Event()
                app.change_event = change_event
                reloader = SourceFileReloader(app=app, watch_dirs=GRADIO_WATCH_DIRS, watch_file=GRADIO_WATCH_FILE, demo_name=GRADIO_WATCH_DEMO_NAME, stop_event=threading.Event(), change_event=change_event)
            server = Server(config=config, reloader=reloader)
            server.run_in_thread()
            break
        except (OSError, ServerFailedToStartError):
            pass
    else:
        raise OSError(f'Cannot find empty port in range: {min(server_ports)}-{max(server_ports)}. You can specify a different port by setting the GRADIO_SERVER_PORT environment variable or passing the `server_port` parameter to `launch()`.')
    if ssl_keyfile is not None:
        path_to_local_server = f'https://{url_host_name}:{port}/'
    else:
        path_to_local_server = f'http://{url_host_name}:{port}/'
    return (server_name, port, path_to_local_server, app, server)

def setup_tunnel(local_host: str, local_port: int, share_token: str, share_server_address: str | None) -> str:
    if False:
        i = 10
        return i + 15
    if share_server_address is None:
        response = requests.get(GRADIO_API_SERVER)
        if not (response and response.status_code == 200):
            raise RuntimeError('Could not get share link from Gradio API Server.')
        payload = response.json()[0]
        (remote_host, remote_port) = (payload['host'], int(payload['port']))
    else:
        (remote_host, remote_port) = share_server_address.split(':')
        remote_port = int(remote_port)
    try:
        tunnel = Tunnel(remote_host, remote_port, local_host, local_port, share_token)
        address = tunnel.start_tunnel()
        return address
    except Exception as e:
        raise RuntimeError(str(e)) from e

def url_ok(url: str) -> bool:
    if False:
        return 10
    try:
        for _ in range(5):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                r = requests.head(url, timeout=3, verify=False)
            if r.status_code in (200, 401, 302):
                return True
            time.sleep(0.5)
    except (ConnectionError, requests.exceptions.ConnectionError):
        return False
    return False