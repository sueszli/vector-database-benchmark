"""Module to setup an ACME CA server environment able to run multiple tests in parallel"""
import argparse
import errno
import json
import os
from os.path import join
import shutil
import subprocess
import sys
import tempfile
import time
from types import TracebackType
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Type
import requests
from certbot_integration_tests.utils import misc
from certbot_integration_tests.utils import pebble_artifacts
from certbot_integration_tests.utils import proxy
from certbot_integration_tests.utils.constants import *

class ACMEServer:
    """
    ACMEServer configures and handles the lifecycle of an ACME CA server and an HTTP reverse proxy
    instance, to allow parallel execution of integration tests against the unique http-01 port
    expected by the ACME CA server.
    Typically all pytest integration tests will be executed in this context.
    ACMEServer gives access the acme_xdist parameter, listing the ports and directory url to use
    for each pytest node. It exposes also start and stop methods in order to start the stack, and
    stop it with proper resources cleanup.
    ACMEServer is also a context manager, and so can be used to ensure ACME server is
    started/stopped upon context enter/exit.
    """

    def __init__(self, acme_server: str, nodes: List[str], http_proxy: bool=True, stdout: bool=False, dns_server: Optional[str]=None, http_01_port: Optional[int]=None) -> None:
        if False:
            return 10
        '\n        Create an ACMEServer instance.\n        :param str acme_server: the type of acme server used (boulder-v2 or pebble)\n        :param list nodes: list of node names that will be setup by pytest xdist\n        :param bool http_proxy: if False do not start the HTTP proxy\n        :param bool stdout: if True stream all subprocesses stdout to standard stdout\n        :param str dns_server: if set, Pebble/Boulder will use it to resolve domains\n        :param int http_01_port: port to use for http-01 validation; currently\n            only supported for pebble without an HTTP proxy\n        '
        self._construct_acme_xdist(acme_server, nodes)
        self._acme_type = 'pebble' if acme_server == 'pebble' else 'boulder'
        self._proxy = http_proxy
        self._workspace = tempfile.mkdtemp()
        self._processes: List[subprocess.Popen] = []
        self._stdout = sys.stdout if stdout else open(os.devnull, 'w')
        self._dns_server = dns_server
        self._preterminate_cmds_args: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self._http_01_port = BOULDER_HTTP_01_PORT if self._acme_type == 'boulder' else DEFAULT_HTTP_01_PORT
        if http_01_port:
            if self._acme_type == 'pebble' and self._proxy or self._acme_type == 'boulder':
                raise ValueError('Setting http_01_port is not currently supported when using Boulder or the HTTP proxy')
            self._http_01_port = http_01_port

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start the test stack'
        try:
            if self._proxy:
                self._prepare_http_proxy()
            if self._acme_type == 'pebble':
                self._prepare_pebble_server()
            if self._acme_type == 'boulder':
                self._prepare_boulder_server()
        except BaseException as e:
            self.stop()
            raise e

    def stop(self) -> None:
        if False:
            return 10
        'Stop the test stack, and clean its resources'
        print('=> Tear down the test infrastructure...')
        try:
            self._run_preterminate_cmds()
            for process in self._processes:
                try:
                    process.terminate()
                except OSError as e:
                    if e.errno != errno.ESRCH:
                        raise
            for process in self._processes:
                process.wait(MAX_SUBPROCESS_WAIT)
        finally:
            if os.path.exists(self._workspace):
                shutil.rmtree(self._workspace)
        if self._stdout != sys.stdout:
            self._stdout.close()
        print('=> Test infrastructure stopped and cleaned up.')

    def __enter__(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        self.start()
        return self.acme_xdist

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if False:
            i = 10
            return i + 15
        self.stop()

    def _construct_acme_xdist(self, acme_server: str, nodes: List[str]) -> None:
        if False:
            while True:
                i = 10
        'Generate and return the acme_xdist dict'
        acme_xdist: Dict[str, Any] = {'acme_server': acme_server}
        if acme_server == 'pebble':
            acme_xdist['directory_url'] = PEBBLE_DIRECTORY_URL
            acme_xdist['challtestsrv_url'] = PEBBLE_CHALLTESTSRV_URL
        else:
            acme_xdist['directory_url'] = BOULDER_V2_DIRECTORY_URL
            acme_xdist['challtestsrv_url'] = BOULDER_V2_CHALLTESTSRV_URL
        acme_xdist['http_port'] = dict(zip(nodes, range(5200, 5200 + len(nodes))))
        acme_xdist['https_port'] = dict(zip(nodes, range(5100, 5100 + len(nodes))))
        acme_xdist['other_port'] = dict(zip(nodes, range(5300, 5300 + len(nodes))))
        self.acme_xdist = acme_xdist

    def _prepare_pebble_server(self) -> None:
        if False:
            i = 10
            return i + 15
        'Configure and launch the Pebble server'
        print('=> Starting pebble instance deployment...')
        pebble_artifacts_rv = pebble_artifacts.fetch(self._workspace, self._http_01_port)
        (pebble_path, challtestsrv_path, pebble_config_path) = pebble_artifacts_rv
        environ = os.environ.copy()
        environ['PEBBLE_VA_NOSLEEP'] = '1'
        environ['PEBBLE_WFE_NONCEREJECT'] = '0'
        environ['PEBBLE_AUTHZREUSE'] = '100'
        environ['PEBBLE_ALTERNATE_ROOTS'] = str(PEBBLE_ALTERNATE_ROOTS)
        if self._dns_server:
            dns_server = self._dns_server
        else:
            dns_server = '127.0.0.1:8053'
            self._launch_process([challtestsrv_path, '-management', ':{0}'.format(CHALLTESTSRV_PORT), '-defaultIPv6', '""', '-defaultIPv4', '127.0.0.1', '-http01', '""', '-tlsalpn01', '""', '-https01', '""'])
        self._launch_process([pebble_path, '-config', pebble_config_path, '-dnsserver', dns_server, '-strict'], env=environ)
        from certbot_integration_tests.utils import pebble_ocsp_server
        self._launch_process([sys.executable, pebble_ocsp_server.__file__])
        print('=> Waiting for pebble instance to respond...')
        misc.check_until_timeout(self.acme_xdist['directory_url'])
        print('=> Finished pebble instance deployment.')

    def _prepare_boulder_server(self) -> None:
        if False:
            while True:
                i = 10
        'Configure and launch the Boulder server'
        print('=> Starting boulder instance deployment...')
        instance_path = join(self._workspace, 'boulder')
        process = self._launch_process(['git', 'clone', 'https://github.com/letsencrypt/boulder', '--single-branch', '--depth=1', instance_path])
        process.wait(MAX_SUBPROCESS_WAIT)
        os.rename(join(instance_path, 'test/rate-limit-policies-b.yml'), join(instance_path, 'test/rate-limit-policies.yml'))
        if self._dns_server:
            for suffix in ['', '-remote-a', '-remote-b']:
                with open(join(instance_path, 'test/config/va{}.json'.format(suffix)), 'r') as f:
                    config = json.loads(f.read())
                config['va']['dnsResolvers'] = [self._dns_server]
                with open(join(instance_path, 'test/config/va{}.json'.format(suffix)), 'w') as f:
                    f.write(json.dumps(config, indent=2, separators=(',', ': ')))
        self._register_preterminate_cmd(['docker-compose', 'down'], cwd=instance_path)
        self._register_preterminate_cmd(['docker', 'run', '--rm', '-v', '{0}:/workspace'.format(self._workspace), 'alpine', 'rm', '-rf', '/workspace/boulder'])
        try:
            self._launch_process(['docker-compose', 'up', '--force-recreate'], cwd=instance_path)
            print('=> Waiting for boulder instance to respond...')
            misc.check_until_timeout(self.acme_xdist['directory_url'], attempts=300)
            if not self._dns_server:
                response = requests.post(f'{BOULDER_V2_CHALLTESTSRV_URL}/set-default-ipv4', json={'ip': '10.77.77.1'}, timeout=10)
                response.raise_for_status()
        except BaseException:
            print('=> Boulder setup failed. Boulder logs are:')
            process = self._launch_process(['docker-compose', 'logs'], cwd=instance_path, force_stderr=True)
            process.wait(MAX_SUBPROCESS_WAIT)
            raise
        print('=> Finished boulder instance deployment.')

    def _prepare_http_proxy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Configure and launch an HTTP proxy'
        print(f'=> Configuring the HTTP proxy on port {self._http_01_port}...')
        http_port_map = cast(Dict[str, int], self.acme_xdist['http_port'])
        mapping = {'.+\\.{0}\\.wtf'.format(node): 'http://127.0.0.1:{0}'.format(port) for (node, port) in http_port_map.items()}
        command = [sys.executable, proxy.__file__, str(self._http_01_port), json.dumps(mapping)]
        self._launch_process(command)
        print('=> Finished configuring the HTTP proxy.')

    def _launch_process(self, command: List[str], cwd: str=os.getcwd(), env: Optional[Mapping[str, str]]=None, force_stderr: bool=False) -> subprocess.Popen:
        if False:
            print('Hello World!')
        'Launch silently a subprocess OS command'
        if not env:
            env = os.environ
        stdout = sys.stderr if force_stderr else self._stdout
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        self._processes.append(process)
        return process

    def _register_preterminate_cmd(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        self._preterminate_cmds_args.append((args, kwargs))

    def _run_preterminate_cmds(self) -> None:
        if False:
            print('Hello World!')
        for (args, kwargs) in self._preterminate_cmds_args:
            process = self._launch_process(*args, **kwargs)
            process.wait(MAX_SUBPROCESS_WAIT)
        self._preterminate_cmds_args.clear()

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='CLI tool to start a local instance of Pebble or Boulder CA server.')
    parser.add_argument('--server-type', '-s', choices=['pebble', 'boulder-v2'], default='pebble', help='type of CA server to start: can be Pebble or Boulder. Pebble is used if not set.')
    parser.add_argument('--dns-server', '-d', help='specify the DNS server as `IP:PORT` to use by Pebble; if not specified, a local mock DNS server will be used to resolve domains to localhost.')
    parser.add_argument('--http-01-port', type=int, default=DEFAULT_HTTP_01_PORT, help='specify the port to use for http-01 validation; this is currently only supported for Pebble.')
    args = parser.parse_args()
    acme_server = ACMEServer(args.server_type, [], http_proxy=False, stdout=True, dns_server=args.dns_server, http_01_port=args.http_01_port)
    try:
        with acme_server as acme_xdist:
            print('--> Instance of {0} is running, directory URL is {0}'.format(acme_xdist['directory_url']))
            print('--> Press CTRL+C to stop the ACME server.')
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
if __name__ == '__main__':
    main()