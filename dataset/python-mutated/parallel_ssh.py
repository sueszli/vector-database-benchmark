from __future__ import absolute_import
import re
import os
import traceback
from paramiko.ssh_exception import SSHException
from st2common.constants.secrets import MASKED_ATTRIBUTE_VALUE
from st2common.runners.paramiko_ssh import ParamikoSSHClient
from st2common.runners.paramiko_ssh import SSHCommandTimeoutError
from st2common import log as logging
from st2common.exceptions.ssh import NoHostsConnectedToException
from st2common.util import ip_utils
from st2common.util import concurrency as concurrency_lib
from st2common.util.jsonify import json_encode
from st2common.util.jsonify import json_loads
LOG = logging.getLogger(__name__)

class ParallelSSHClient(object):
    KEYS_TO_TRANSFORM = ['stdout', 'stderr']
    CONNECT_ERROR = 'Cannot connect to host.'

    def __init__(self, hosts, user=None, password=None, pkey_file=None, pkey_material=None, port=22, bastion_host=None, concurrency=10, raise_on_any_error=False, connect=True, passphrase=None, handle_stdout_line_func=None, handle_stderr_line_func=None, sudo_password=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param handle_stdout_line_func: Callback function which is called dynamically each time a\n                                        new stdout line is received.\n        :type handle_stdout_line_func: ``func``\n\n        :param handle_stderr_line_func: Callback function which is called dynamically each time a\n                                        new stderr line is received.\n        :type handle_stderr_line_func: ``func``\n        '
        self._ssh_user = user
        self._ssh_key_file = pkey_file
        self._ssh_key_material = pkey_material
        self._ssh_password = password
        self._hosts = hosts
        self._successful_connects = 0
        self._ssh_port = port
        self._bastion_host = bastion_host
        self._passphrase = passphrase
        self._handle_stdout_line_func = handle_stdout_line_func
        self._handle_stderr_line_func = handle_stderr_line_func
        self._sudo_password = sudo_password
        if not hosts:
            raise Exception('Need an non-empty list of hosts to talk to.')
        self._pool = concurrency_lib.get_green_pool_class()(concurrency)
        self._hosts_client = {}
        self._bad_hosts = {}
        self._scan_interval = 0.1
        if connect:
            connect_results = self.connect(raise_on_any_error=raise_on_any_error)
            extra = {'_connect_results': connect_results}
            LOG.debug('Connect to hosts complete.', extra=extra)

    def connect(self, raise_on_any_error=False):
        if False:
            while True:
                i = 10
        '\n        Connect to hosts in hosts list. Returns status of connect as a dict.\n\n        :param raise_on_any_error: Optional Raise an exception even if connecting to one\n                                   of the hosts fails.\n        :type raise_on_any_error: ``boolean``\n\n        :rtype: ``dict`` of ``str`` to ``dict``\n        '
        results = {}
        for host in self._hosts:
            while not concurrency_lib.is_green_pool_free(self._pool):
                concurrency_lib.sleep(self._scan_interval)
            self._pool.spawn(self._connect, host=host, results=results, raise_on_any_error=raise_on_any_error)
        concurrency_lib.green_pool_wait_all(self._pool)
        if self._successful_connects < 1:
            LOG.error('Unable to connect to any of the hosts.', extra={'connect_results': results})
            msg = 'Unable to connect to any one of the hosts: %s.\n\n connect_errors=%s' % (self._hosts, json_encode(results, indent=2))
            raise NoHostsConnectedToException(msg)
        return results

    def run(self, cmd, timeout=None):
        if False:
            print('Hello World!')
        '\n        Run a command on remote hosts. Returns a dict containing results\n        of execution from all hosts.\n\n        :param cmd: Command to run. Must be shlex quoted.\n        :type cmd: ``str``\n\n        :param timeout: Optional Timeout for the command.\n        :type timeout: ``int``\n\n        :param cwd: Optional Current working directory. Must be shlex quoted.\n        :type cwd: ``str``\n\n        :rtype: ``dict`` of ``str`` to ``dict``\n        '
        options = {'cmd': cmd, 'timeout': timeout}
        results = self._execute_in_pool(self._run_command, **options)
        return results

    def put(self, local_path, remote_path, mode=None, mirror_local_mode=False):
        if False:
            print('Hello World!')
        '\n        Copy a file or folder to remote host.\n\n        :param local_path: Path to local file or dir. Must be shlex quoted.\n        :type local_path: ``str``\n\n        :param remote_path: Path to remote file or dir. Must be shlex quoted.\n        :type remote_path: ``str``\n\n        :param mode: Optional mode to use for the file or dir.\n        :type mode: ``int``\n\n        :param mirror_local_mode: Optional Flag to mirror the mode\n                                           on local file/dir on remote host.\n        :type mirror_local_mode: ``boolean``\n\n        :rtype: ``dict`` of ``str`` to ``dict``\n        '
        if not os.path.exists(local_path):
            raise Exception('Local path %s does not exist.' % local_path)
        options = {'local_path': local_path, 'remote_path': remote_path, 'mode': mode, 'mirror_local_mode': mirror_local_mode}
        return self._execute_in_pool(self._put_files, **options)

    def mkdir(self, path):
        if False:
            print('Hello World!')
        '\n        Create a directory on remote hosts.\n\n        :param path: Path to remote dir that must be created. Must be shlex quoted.\n        :type path: ``str``\n\n        :rtype path: ``dict`` of ``str`` to ``dict``\n        '
        options = {'path': path}
        return self._execute_in_pool(self._mkdir, **options)

    def delete_file(self, path):
        if False:
            i = 10
            return i + 15
        '\n        Delete a file on remote hosts.\n\n        :param path: Path to remote file that must be deleted. Must be shlex quoted.\n        :type path: ``str``\n\n        :rtype path: ``dict`` of ``str`` to ``dict``\n        '
        options = {'path': path}
        return self._execute_in_pool(self._delete_file, **options)

    def delete_dir(self, path, force=False, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete a dir on remote hosts.\n\n        :param path: Path to remote dir that must be deleted. Must be shlex quoted.\n        :type path: ``str``\n\n        :rtype path: ``dict`` of ``str`` to ``dict``\n        '
        options = {'path': path, 'force': force}
        return self._execute_in_pool(self._delete_dir, **options)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close all open SSH connections to hosts.\n        '
        for host in self._hosts_client.keys():
            try:
                self._hosts_client[host].close()
            except:
                LOG.exception('Failed shutting down SSH connection to host: %s', host)

    def _execute_in_pool(self, execute_method, **kwargs):
        if False:
            while True:
                i = 10
        results = {}
        for host in self._bad_hosts.keys():
            results[host] = self._bad_hosts[host]
        for host in self._hosts_client.keys():
            while not self._pool.free():
                concurrency_lib.sleep(self._scan_interval)
            self._pool.spawn(execute_method, host=host, results=results, **kwargs)
        concurrency_lib.green_pool_wait_all(self._pool)
        return results

    def _connect(self, host, results, raise_on_any_error=False):
        if False:
            return 10
        (hostname, port) = self._get_host_port_info(host)
        extra = {'host': host, 'port': port, 'user': self._ssh_user}
        if self._ssh_password:
            extra['password'] = '<redacted>'
        elif self._ssh_key_file:
            extra['key_file_path'] = self._ssh_key_file
        else:
            extra['private_key'] = '<redacted>'
        LOG.debug('Connecting to host.', extra=extra)
        client = ParamikoSSHClient(hostname=hostname, port=port, username=self._ssh_user, password=self._ssh_password, bastion_host=self._bastion_host, key_files=self._ssh_key_file, key_material=self._ssh_key_material, passphrase=self._passphrase, handle_stdout_line_func=self._handle_stdout_line_func, handle_stderr_line_func=self._handle_stderr_line_func)
        try:
            client.connect()
        except SSHException as ex:
            LOG.exception(ex)
            if raise_on_any_error:
                raise
            error_dict = self._generate_error_result(exc=ex, message='Connection error.')
            self._bad_hosts[hostname] = error_dict
            results[hostname] = error_dict
        except Exception as ex:
            error = 'Failed connecting to host %s.' % hostname
            LOG.exception(error)
            if raise_on_any_error:
                raise
            error_dict = self._generate_error_result(exc=ex, message=error)
            self._bad_hosts[hostname] = error_dict
            results[hostname] = error_dict
        else:
            self._successful_connects += 1
            self._hosts_client[hostname] = client
            results[hostname] = {'message': 'Connected to host.'}

    def _run_command(self, host, cmd, results, timeout=None):
        if False:
            return 10
        try:
            LOG.debug('Running command: %s on host: %s.', cmd, host)
            client = self._hosts_client[host]
            (stdout, stderr, exit_code) = client.run(cmd, timeout=timeout, call_line_handler_func=True)
            result = self._handle_command_result(stdout=stdout, stderr=stderr, exit_code=exit_code)
            results[host] = result
        except Exception as ex:
            cmd = self._sanitize_command_string(cmd=cmd)
            error = 'Failed executing command "%s" on host "%s"' % (cmd, host)
            LOG.exception(error)
            results[host] = self._generate_error_result(exc=ex, message=error)

    def _put_files(self, local_path, remote_path, host, results, mode=None, mirror_local_mode=False):
        if False:
            i = 10
            return i + 15
        try:
            LOG.debug('Copying file to host: %s' % host)
            if os.path.isdir(local_path):
                result = self._hosts_client[host].put_dir(local_path, remote_path)
            else:
                result = self._hosts_client[host].put(local_path, remote_path, mirror_local_mode=mirror_local_mode, mode=mode)
            LOG.debug('Result of copy: %s' % result)
            results[host] = result
        except Exception as ex:
            error = 'Failed sending file(s) in path %s to host %s' % (local_path, host)
            LOG.exception(error)
            results[host] = self._generate_error_result(exc=ex, message=error)

    def _mkdir(self, host, path, results):
        if False:
            for i in range(10):
                print('nop')
        try:
            result = self._hosts_client[host].mkdir(path)
            results[host] = result
        except Exception as ex:
            error = 'Failed "mkdir %s" on host %s.' % (path, host)
            LOG.exception(error)
            results[host] = self._generate_error_result(exc=ex, message=error)

    def _delete_file(self, host, path, results):
        if False:
            for i in range(10):
                print('nop')
        try:
            result = self._hosts_client[host].delete_file(path)
            results[host] = result
        except Exception as ex:
            error = 'Failed deleting file %s on host %s.' % (path, host)
            LOG.exception(error)
            results[host] = self._generate_error_result(exc=ex, message=error)

    def _delete_dir(self, host, path, results, force=False, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            result = self._hosts_client[host].delete_dir(path, force=force, timeout=timeout)
            results[host] = result
        except Exception as ex:
            error = 'Failed deleting dir %s on host %s.' % (path, host)
            LOG.exception(error)
            results[host] = self._generate_error_result(exc=ex, message=error)

    def _get_host_port_info(self, host_str):
        if False:
            i = 10
            return i + 15
        (hostname, port) = ip_utils.split_host_port(host_str)
        if not port:
            port = self._ssh_port
        return (hostname, port)

    def _handle_command_result(self, stdout, stderr, exit_code):
        if False:
            return 10
        if self._sudo_password:
            if re.search('sudo: \\d+ incorrect password attempts', stderr):
                match = re.search('\\[sudo\\] password for (.+?)\\:', stderr)
                if match:
                    username = match.groups()[0]
                else:
                    username = 'unknown'
                error = 'Invalid sudo password provided or sudo is not configured for this user (%s)' % username
                raise ValueError(error)
        is_succeeded = exit_code == 0
        result_dict = {'stdout': stdout, 'stderr': stderr, 'return_code': exit_code, 'succeeded': is_succeeded, 'failed': not is_succeeded}
        result = json_loads(result_dict, ParallelSSHClient.KEYS_TO_TRANSFORM)
        return result

    @staticmethod
    def _sanitize_command_string(cmd):
        if False:
            i = 10
            return i + 15
        '\n        Remove any potentially sensitive information from the command string.\n\n        For now we only mask the values of the sensitive environment variables.\n        '
        if not cmd:
            return cmd
        result = re.sub('ST2_ACTION_AUTH_TOKEN=(.+?)\\s+?', 'ST2_ACTION_AUTH_TOKEN=%s ' % MASKED_ATTRIBUTE_VALUE, cmd)
        return result

    @staticmethod
    def _generate_error_result(exc, message):
        if False:
            print('Hello World!')
        '\n        :param exc: Raised exception.\n        :type exc: Exception.\n\n        :param message: Error message which will be prefixed to the exception exception message.\n        :type message: ``str``\n        '
        exc_message = getattr(exc, 'message', str(exc))
        error_message = '%s %s' % (message, exc_message)
        traceback_message = traceback.format_exc()
        if isinstance(exc, SSHCommandTimeoutError):
            return_code = -9
            timeout = True
        else:
            timeout = False
            return_code = 255
        stdout = getattr(exc, 'stdout', None) or ''
        stderr = getattr(exc, 'stderr', None) or ''
        error_dict = {'failed': True, 'succeeded': False, 'timeout': timeout, 'return_code': return_code, 'stdout': stdout, 'stderr': stderr, 'error': error_message, 'traceback': traceback_message}
        return error_dict

    def __repr__(self):
        if False:
            return 10
        return '<ParallelSSHClient hosts=%s,user=%s,id=%s>' % (repr(self._hosts), self._ssh_user, id(self))