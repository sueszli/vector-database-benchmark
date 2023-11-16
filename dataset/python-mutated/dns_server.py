"""Module to setup an RFC2136-capable DNS server"""
import os
import os.path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from types import TracebackType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from certbot_integration_tests.utils import constants
if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources
BIND_DOCKER_IMAGE = 'internetsystemsconsortium/bind9:9.16'
BIND_BIND_ADDRESS = ('127.0.0.1', 45953)
BIND_TEST_QUERY = bytearray.fromhex('0011cb37000000010000000000000000010003')

class DNSServer:
    """
    DNSServer configures and handles the lifetime of an RFC2136-capable server.
    DNServer provides access to the dns_xdist parameter, listing the address and port
    to use for each pytest node.

    At this time, DNSServer should only be used with a single node, but may be expanded in
    future to support parallelization (https://github.com/certbot/certbot/issues/8455).
    """

    def __init__(self, unused_nodes: List[str], show_output: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create an DNSServer instance.\n        :param list nodes: list of node names that will be setup by pytest xdist\n        :param bool show_output: if True, print the output of the DNS server\n        '
        self.bind_root = tempfile.mkdtemp()
        self.process: Optional[subprocess.Popen] = None
        self.dns_xdist = {'address': BIND_BIND_ADDRESS[0], 'port': BIND_BIND_ADDRESS[1]}
        self._output = sys.stderr if show_output else open(os.devnull, 'w')

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start the DNS server'
        try:
            self._configure_bind()
            self._start_bind()
        except:
            self.stop()
            raise

    def stop(self) -> None:
        if False:
            print('Hello World!')
        'Stop the DNS server, and clean its resources'
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(constants.MAX_SUBPROCESS_WAIT)
            except BaseException as e:
                print('BIND9 did not stop cleanly: {}'.format(e), file=sys.stderr)
        shutil.rmtree(self.bind_root, ignore_errors=True)
        if self._output != sys.stderr:
            self._output.close()

    def _configure_bind(self) -> None:
        if False:
            while True:
                i = 10
        'Configure the BIND9 server based on the prebaked configuration'
        ref = importlib_resources.files('certbot_integration_tests') / 'assets' / 'bind-config'
        with importlib_resources.as_file(ref) as path:
            for directory in ('conf', 'zones'):
                shutil.copytree(os.path.join(path, directory), os.path.join(self.bind_root, directory))

    def _start_bind(self) -> None:
        if False:
            return 10
        'Launch the BIND9 server as a Docker container'
        addr_str = '{}:{}'.format(BIND_BIND_ADDRESS[0], BIND_BIND_ADDRESS[1])
        self.process = subprocess.Popen(['docker', 'run', '--rm', '-p', '{}:53/udp'.format(addr_str), '-p', '{}:53/tcp'.format(addr_str), '-v', '{}/conf:/etc/bind'.format(self.bind_root), '-v', '{}/zones:/var/lib/bind'.format(self.bind_root), BIND_DOCKER_IMAGE], stdout=self._output, stderr=self._output)
        if self.process.poll():
            raise ValueError('BIND9 server stopped unexpectedly')
        try:
            self._wait_until_ready()
        except:
            self.stop()
            raise

    def _wait_until_ready(self, attempts: int=30) -> None:
        if False:
            print('Hello World!')
        '\n        Polls the DNS server over TCP until it gets a response, or until\n        it runs out of attempts and raises a ValueError.\n        The DNS response message must match the txn_id of the DNS query message,\n        but otherwise the contents are ignored.\n        :param int attempts: The number of attempts to make.\n        '
        if not self.process:
            raise ValueError('DNS server has not been started. Please run start() first.')
        for _ in range(attempts):
            if self.process.poll():
                raise ValueError('BIND9 server stopped unexpectedly')
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            try:
                sock.connect(BIND_BIND_ADDRESS)
                sock.sendall(BIND_TEST_QUERY)
                buf = sock.recv(1024)
                if buf and len(buf) > 4 and (buf[2:4] == BIND_TEST_QUERY[2:4]):
                    return
                time.sleep(1)
            except:
                time.sleep(1)
            finally:
                sock.close()
        raise ValueError('Gave up waiting for DNS server {} to respond'.format(BIND_BIND_ADDRESS))

    def __start__(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        self.start()
        return self.dns_xdist

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            i = 10
            return i + 15
        self.stop()