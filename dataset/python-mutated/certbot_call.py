"""Module to call certbot in test mode"""
import os
import subprocess
import sys
from typing import Dict
from typing import List
from typing import Tuple
import certbot_integration_tests
from certbot_integration_tests.utils.constants import *

def certbot_test(certbot_args: List[str], directory_url: str, http_01_port: int, tls_alpn_01_port: int, config_dir: str, workspace: str, force_renew: bool=True) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Invoke the certbot executable available in PATH in a test context for the given args.\n    The test context consists in running certbot in debug mode, with various flags suitable\n    for tests (eg. no ssl check, customizable ACME challenge ports and config directory ...).\n    This command captures both stdout and stderr and returns it to the caller.\n    :param list certbot_args: the arguments to pass to the certbot executable\n    :param str directory_url: URL of the ACME directory server to use\n    :param int http_01_port: port for the HTTP-01 challenges\n    :param int tls_alpn_01_port: port for the TLS-ALPN-01 challenges\n    :param str config_dir: certbot configuration directory to use\n    :param str workspace: certbot current directory to use\n    :param bool force_renew: set False to not force renew existing certificates (default: True)\n    :return: stdout and stderr as strings\n    :rtype: `tuple` of `str`\n    '
    (command, env) = _prepare_args_env(certbot_args, directory_url, http_01_port, tls_alpn_01_port, config_dir, workspace, force_renew)
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, universal_newlines=True, cwd=workspace, env=env)
    print('--> Certbot log output was:')
    print(proc.stderr)
    proc.check_returncode()
    return (proc.stdout, proc.stderr)

def _prepare_environ(workspace: str) -> Dict[str, str]:
    if False:
        while True:
            i = 10
    new_environ = os.environ.copy()
    new_environ['TMPDIR'] = workspace
    if new_environ.get('PYTHONPATH'):
        certbot_root = os.path.dirname(os.path.dirname(os.path.dirname(certbot_integration_tests.__file__)))
        python_paths = [path for path in new_environ['PYTHONPATH'].split(':') if path != certbot_root]
        new_environ['PYTHONPATH'] = ':'.join(python_paths)
    return new_environ

def _prepare_args_env(certbot_args: List[str], directory_url: str, http_01_port: int, tls_alpn_01_port: int, config_dir: str, workspace: str, force_renew: bool) -> Tuple[List[str], Dict[str, str]]:
    if False:
        for i in range(10):
            print('nop')
    new_environ = _prepare_environ(workspace)
    additional_args = ['--no-random-sleep-on-renew']
    if force_renew:
        additional_args.append('--renew-by-default')
    command = ['certbot', '--server', directory_url, '--no-verify-ssl', '--http-01-port', str(http_01_port), '--https-port', str(tls_alpn_01_port), '--manual-public-ip-logging-ok', '--config-dir', config_dir, '--work-dir', os.path.join(workspace, 'work'), '--logs-dir', os.path.join(workspace, 'logs'), '--non-interactive', '--no-redirect', '--agree-tos', '--register-unsafely-without-email', '--debug', '-vv']
    command.extend(certbot_args)
    command.extend(additional_args)
    print('--> Invoke command:\n=====\n{0}\n====='.format(subprocess.list2cmdline(command)))
    return (command, new_environ)

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    args = sys.argv[1:]
    directory_url = os.environ.get('SERVER', PEBBLE_DIRECTORY_URL)
    http_01_port = int(os.environ.get('HTTP_01_PORT', DEFAULT_HTTP_01_PORT))
    tls_alpn_01_port = int(os.environ.get('TLS_ALPN_01_PORT', TLS_ALPN_01_PORT))
    workspace = os.environ.get('WORKSPACE', os.path.join(os.getcwd(), '.certbot_test_workspace'))
    if not os.path.exists(workspace):
        print('--> Creating a workspace for certbot_test: {0}'.format(workspace))
        os.mkdir(workspace)
    else:
        print('--> Using an existing workspace for certbot_test: {0}'.format(workspace))
    config_dir = os.path.join(workspace, 'conf')
    (command, env) = _prepare_args_env(args, directory_url, http_01_port, tls_alpn_01_port, config_dir, workspace, True)
    subprocess.check_call(command, universal_newlines=True, cwd=workspace, env=env)
if __name__ == '__main__':
    main()