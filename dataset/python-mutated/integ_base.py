import os
import shutil
import tempfile
import logging
import subprocess
import time
import re
import pytest
from flask import Flask, request, Response
from threading import Thread
from collections import deque
from unittest import TestCase
from pathlib import Path
from werkzeug.serving import make_server
from samcli.cli.global_config import GlobalConfig
from samcli.cli.main import TELEMETRY_PROMPT
from tests.testing_utils import get_sam_command
LOG = logging.getLogger(__name__)
TELEMETRY_ENDPOINT_PORT = '18298'
TELEMETRY_ENDPOINT_HOST = 'localhost'
TELEMETRY_ENDPOINT_URL = 'http://{}:{}'.format(TELEMETRY_ENDPOINT_HOST, TELEMETRY_ENDPOINT_PORT)
EXPECTED_TELEMETRY_PROMPT = re.sub('\\n', os.linesep, TELEMETRY_PROMPT)

@pytest.mark.xdist_group(name='sam_telemetry')
class IntegBase(TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.cmd = get_sam_command()

    def setUp(self):
        if False:
            print('Hello World!')
        self.maxDiff = None
        self.config_dir = tempfile.mkdtemp()
        self._gc = GlobalConfig()
        self._gc.config_dir = Path(self.config_dir)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.config_dir and shutil.rmtree(self.config_dir)

    def run_cmd(self, cmd_list=None, stdin_data='', optout_envvar_value=None):
        if False:
            print('Hello World!')
        cmd_list = cmd_list or [self.cmd, 'local', 'generate-event', 's3', 'put']
        env = os.environ.copy()
        env.pop('SAM_CLI_TELEMETRY', None)
        if optout_envvar_value:
            env['SAM_CLI_TELEMETRY'] = optout_envvar_value
        env['__SAM_CLI_APP_DIR'] = self.config_dir
        env['__SAM_CLI_TELEMETRY_ENDPOINT_URL'] = '{}/metrics'.format(TELEMETRY_ENDPOINT_URL)
        process = subprocess.Popen(cmd_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        return process

    def unset_config(self):
        if False:
            for i in range(10):
                print('nop')
        config_file = Path(self.config_dir, 'metadata.json')
        if config_file.exists():
            config_file.unlink()

    def set_config(self, telemetry_enabled=None):
        if False:
            return 10
        self._gc.telemetry_enabled = telemetry_enabled

    def get_global_config(self):
        if False:
            for i in range(10):
                print('nop')
        return self._gc

class TelemetryServer:
    """
    HTTP Server that can receive and store Telemetry requests. Caller can later retrieve the responses for
    assertion

    Examples
    --------
    >>> with TelemetryServer() as server:
    >>>     # Server is running now
    >>>     # Set the Telemetry backend endpoint to the server's URL
    >>>     env = os.environ.copy().setdefault("__SAM_CLI_TELEMETRY_ENDPOINT_URL", server.url)
    >>>     # Run SAM CLI command
    >>>     p = subprocess.Popen(["samdev", "local", "generate-event", "s3", "put"], env=env)
    >>>     p.wait()  # Wait for process to complete
    >>>     # Get the first metrics request that was sent
    >>>     r = server.get_request(0)
    >>>     assert r.method == 'POST'
    >>>     assert r.body == "{...}"
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.flask_app = Flask(__name__)
        self.flask_app.add_url_rule('/metrics', endpoint='/metrics', view_func=self._request_handler, methods=['POST'], provide_automatic_options=False)
        self._requests = deque()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.server = make_server(TELEMETRY_ENDPOINT_HOST, TELEMETRY_ENDPOINT_PORT, self.flask_app)
        self.thread = Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            return 10
        time.sleep(2)
        self.server.shutdown()
        self.thread.join()

    def get_request(self, index):
        if False:
            while True:
                i = 10
        return self._requests[index]

    def get_all_requests(self):
        if False:
            print('Hello World!')
        return list(self._requests)

    def _request_handler(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Handles Flask requests\n        '
        request_data = {'endpoint': request.endpoint, 'method': request.method, 'data': request.get_json(), 'headers': dict(request.headers)}
        self._requests.append(request_data)
        return Response(response={}, status=200)