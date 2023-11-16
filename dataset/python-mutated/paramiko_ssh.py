from __future__ import annotations
DOCUMENTATION = '\n    author: Ansible Core Team\n    name: paramiko\n    short_description: Run tasks via Python SSH (paramiko)\n    description:\n        - Use the Python SSH implementation (Paramiko) to connect to targets\n        - The paramiko transport is provided because many distributions, in particular EL6 and before do not support ControlPersist\n          in their SSH implementations.\n        - This is needed on the Ansible control machine to be reasonably efficient with connections.\n          Thus paramiko is faster for most users on these platforms.\n          Users with ControlPersist capability can consider using -c ssh or configuring the transport in the configuration file.\n        - This plugin also borrows a lot of settings from the ssh plugin as they both cover the same protocol.\n    version_added: "0.1"\n    options:\n      remote_addr:\n        description:\n            - Address of the remote target\n        default: inventory_hostname\n        type: string\n        vars:\n            - name: inventory_hostname\n            - name: ansible_host\n            - name: ansible_ssh_host\n            - name: ansible_paramiko_host\n      port:\n          description: Remote port to connect to.\n          type: int\n          default: 22\n          ini:\n            - section: defaults\n              key: remote_port\n            - section: paramiko_connection\n              key: remote_port\n              version_added: \'2.15\'\n          env:\n            - name: ANSIBLE_REMOTE_PORT\n            - name: ANSIBLE_REMOTE_PARAMIKO_PORT\n              version_added: \'2.15\'\n          vars:\n            - name: ansible_port\n            - name: ansible_ssh_port\n            - name: ansible_paramiko_port\n              version_added: \'2.15\'\n          keyword:\n            - name: port\n      remote_user:\n        description:\n            - User to login/authenticate as\n            - Can be set from the CLI via the C(--user) or C(-u) options.\n        type: string\n        vars:\n            - name: ansible_user\n            - name: ansible_ssh_user\n            - name: ansible_paramiko_user\n        env:\n            - name: ANSIBLE_REMOTE_USER\n            - name: ANSIBLE_PARAMIKO_REMOTE_USER\n              version_added: \'2.5\'\n        ini:\n            - section: defaults\n              key: remote_user\n            - section: paramiko_connection\n              key: remote_user\n              version_added: \'2.5\'\n        keyword:\n            - name: remote_user\n      password:\n        description:\n          - Secret used to either login the ssh server or as a passphrase for ssh keys that require it\n          - Can be set from the CLI via the C(--ask-pass) option.\n        type: string\n        vars:\n            - name: ansible_password\n            - name: ansible_ssh_pass\n            - name: ansible_ssh_password\n            - name: ansible_paramiko_pass\n            - name: ansible_paramiko_password\n              version_added: \'2.5\'\n      use_rsa_sha2_algorithms:\n        description:\n            - Whether or not to enable RSA SHA2 algorithms for pubkeys and hostkeys\n            - On paramiko versions older than 2.9, this only affects hostkeys\n            - For behavior matching paramiko<2.9 set this to V(False)\n        vars:\n            - name: ansible_paramiko_use_rsa_sha2_algorithms\n        ini:\n            - {key: use_rsa_sha2_algorithms, section: paramiko_connection}\n        env:\n            - {name: ANSIBLE_PARAMIKO_USE_RSA_SHA2_ALGORITHMS}\n        default: True\n        type: boolean\n        version_added: \'2.14\'\n      host_key_auto_add:\n        description: \'Automatically add host keys\'\n        env: [{name: ANSIBLE_PARAMIKO_HOST_KEY_AUTO_ADD}]\n        ini:\n          - {key: host_key_auto_add, section: paramiko_connection}\n        type: boolean\n      look_for_keys:\n        default: True\n        description: \'False to disable searching for private key files in ~/.ssh/\'\n        env: [{name: ANSIBLE_PARAMIKO_LOOK_FOR_KEYS}]\n        ini:\n        - {key: look_for_keys, section: paramiko_connection}\n        type: boolean\n      proxy_command:\n        default: \'\'\n        description:\n            - Proxy information for running the connection via a jumphost\n            - Also this plugin will scan \'ssh_args\', \'ssh_extra_args\' and \'ssh_common_args\' from the \'ssh\' plugin settings for proxy information if set.\n        type: string\n        env: [{name: ANSIBLE_PARAMIKO_PROXY_COMMAND}]\n        ini:\n          - {key: proxy_command, section: paramiko_connection}\n        vars:\n          - name: ansible_paramiko_proxy_command\n            version_added: \'2.15\'\n      ssh_args:\n          description: Only used in parsing ProxyCommand for use in this plugin.\n          default: \'\'\n          type: string\n          ini:\n              - section: \'ssh_connection\'\n                key: \'ssh_args\'\n          env:\n              - name: ANSIBLE_SSH_ARGS\n          vars:\n              - name: ansible_ssh_args\n                version_added: \'2.7\'\n          deprecated:\n              why: In favor of the "proxy_command" option.\n              version: "2.18"\n              alternatives: proxy_command\n      ssh_common_args:\n          description: Only used in parsing ProxyCommand for use in this plugin.\n          type: string\n          ini:\n              - section: \'ssh_connection\'\n                key: \'ssh_common_args\'\n                version_added: \'2.7\'\n          env:\n              - name: ANSIBLE_SSH_COMMON_ARGS\n                version_added: \'2.7\'\n          vars:\n              - name: ansible_ssh_common_args\n          cli:\n              - name: ssh_common_args\n          default: \'\'\n          deprecated:\n              why: In favor of the "proxy_command" option.\n              version: "2.18"\n              alternatives: proxy_command\n      ssh_extra_args:\n          description: Only used in parsing ProxyCommand for use in this plugin.\n          type: string\n          vars:\n              - name: ansible_ssh_extra_args\n          env:\n            - name: ANSIBLE_SSH_EXTRA_ARGS\n              version_added: \'2.7\'\n          ini:\n            - key: ssh_extra_args\n              section: ssh_connection\n              version_added: \'2.7\'\n          cli:\n            - name: ssh_extra_args\n          default: \'\'\n          deprecated:\n              why: In favor of the "proxy_command" option.\n              version: "2.18"\n              alternatives: proxy_command\n      pty:\n        default: True\n        description: \'SUDO usually requires a PTY, True to give a PTY and False to not give a PTY.\'\n        env:\n          - name: ANSIBLE_PARAMIKO_PTY\n        ini:\n          - section: paramiko_connection\n            key: pty\n        type: boolean\n      record_host_keys:\n        default: True\n        description: \'Save the host keys to a file\'\n        env: [{name: ANSIBLE_PARAMIKO_RECORD_HOST_KEYS}]\n        ini:\n          - section: paramiko_connection\n            key: record_host_keys\n        type: boolean\n      host_key_checking:\n        description: \'Set this to "False" if you want to avoid host key checking by the underlying tools Ansible uses to connect to the host\'\n        type: boolean\n        default: True\n        env:\n          - name: ANSIBLE_HOST_KEY_CHECKING\n          - name: ANSIBLE_SSH_HOST_KEY_CHECKING\n            version_added: \'2.5\'\n          - name: ANSIBLE_PARAMIKO_HOST_KEY_CHECKING\n            version_added: \'2.5\'\n        ini:\n          - section: defaults\n            key: host_key_checking\n          - section: paramiko_connection\n            key: host_key_checking\n            version_added: \'2.5\'\n        vars:\n          - name: ansible_host_key_checking\n            version_added: \'2.5\'\n          - name: ansible_ssh_host_key_checking\n            version_added: \'2.5\'\n          - name: ansible_paramiko_host_key_checking\n            version_added: \'2.5\'\n      use_persistent_connections:\n        description: \'Toggles the use of persistence for connections\'\n        type: boolean\n        default: False\n        env:\n          - name: ANSIBLE_USE_PERSISTENT_CONNECTIONS\n        ini:\n          - section: defaults\n            key: use_persistent_connections\n      banner_timeout:\n        type: float\n        default: 30\n        version_added: \'2.14\'\n        description:\n          - Configures, in seconds, the amount of time to wait for the SSH\n            banner to be presented. This option is supported by paramiko\n            version 1.15.0 or newer.\n        ini:\n          - section: paramiko_connection\n            key: banner_timeout\n        env:\n          - name: ANSIBLE_PARAMIKO_BANNER_TIMEOUT\n      timeout:\n        type: int\n        default: 10\n        description: Number of seconds until the plugin gives up on failing to establish a TCP connection.\n        ini:\n          - section: defaults\n            key: timeout\n          - section: ssh_connection\n            key: timeout\n            version_added: \'2.11\'\n          - section: paramiko_connection\n            key: timeout\n            version_added: \'2.15\'\n        env:\n          - name: ANSIBLE_TIMEOUT\n          - name: ANSIBLE_SSH_TIMEOUT\n            version_added: \'2.11\'\n          - name: ANSIBLE_PARAMIKO_TIMEOUT\n            version_added: \'2.15\'\n        vars:\n          - name: ansible_ssh_timeout\n            version_added: \'2.11\'\n          - name: ansible_paramiko_timeout\n            version_added: \'2.15\'\n        cli:\n          - name: timeout\n      private_key_file:\n          description:\n              - Path to private key file to use for authentication.\n          type: string\n          ini:\n            - section: defaults\n              key: private_key_file\n            - section: paramiko_connection\n              key: private_key_file\n              version_added: \'2.15\'\n          env:\n            - name: ANSIBLE_PRIVATE_KEY_FILE\n            - name: ANSIBLE_PARAMIKO_PRIVATE_KEY_FILE\n              version_added: \'2.15\'\n          vars:\n            - name: ansible_private_key_file\n            - name: ansible_ssh_private_key_file\n            - name: ansible_paramiko_private_key_file\n              version_added: \'2.15\'\n          cli:\n            - name: private_key_file\n              option: \'--private-key\'\n'
import os
import socket
import tempfile
import traceback
import fcntl
import re
import typing as t
from ansible.module_utils.compat.version import LooseVersion
from binascii import hexlify
from ansible.errors import AnsibleAuthenticationFailure, AnsibleConnectionFailure, AnsibleError, AnsibleFileNotFound
from ansible.module_utils.compat.paramiko import PARAMIKO_IMPORT_ERR, paramiko
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible.utils.path import makedirs_safe
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
display = Display()
AUTHENTICITY_MSG = "\nparamiko: The authenticity of host '%s' can't be established.\nThe %s key fingerprint is %s.\nAre you sure you want to continue connecting (yes/no)?\n"
SETTINGS_REGEX = re.compile('(\\w+)(?:\\s*=\\s*|\\s+)(.+)')
MissingHostKeyPolicy: type = object
if paramiko:
    MissingHostKeyPolicy = paramiko.MissingHostKeyPolicy

class MyAddPolicy(MissingHostKeyPolicy):
    """
    Based on AutoAddPolicy in paramiko so we can determine when keys are added

    and also prompt for input.

    Policy for automatically adding the hostname and new host key to the
    local L{HostKeys} object, and saving it.  This is used by L{SSHClient}.
    """

    def __init__(self, connection: Connection) -> None:
        if False:
            return 10
        self.connection = connection
        self._options = connection._options

    def missing_host_key(self, client, hostname, key) -> None:
        if False:
            i = 10
            return i + 15
        if all((self.connection.get_option('host_key_checking'), not self.connection.get_option('host_key_auto_add'))):
            fingerprint = hexlify(key.get_fingerprint())
            ktype = key.get_name()
            if self.connection.get_option('use_persistent_connections') or self.connection.force_persistence:
                raise AnsibleError(AUTHENTICITY_MSG[1:92] % (hostname, ktype, fingerprint))
            inp = to_text(display.prompt_until(AUTHENTICITY_MSG % (hostname, ktype, fingerprint), private=False), errors='surrogate_or_strict')
            if inp not in ['yes', 'y', '']:
                raise AnsibleError('host connection rejected by user')
        key._added_by_ansible_this_time = True
        client._host_keys.add(hostname, key.get_name(), key)
SSH_CONNECTION_CACHE: dict[str, paramiko.client.SSHClient] = {}
SFTP_CONNECTION_CACHE: dict[str, paramiko.sftp_client.SFTPClient] = {}

class Connection(ConnectionBase):
    """ SSH based connections with Paramiko """
    transport = 'paramiko'
    _log_channel: str | None = None

    def _cache_key(self) -> str:
        if False:
            return 10
        return '%s__%s__' % (self.get_option('remote_addr'), self.get_option('remote_user'))

    def _connect(self) -> Connection:
        if False:
            for i in range(10):
                print('nop')
        cache_key = self._cache_key()
        if cache_key in SSH_CONNECTION_CACHE:
            self.ssh = SSH_CONNECTION_CACHE[cache_key]
        else:
            self.ssh = SSH_CONNECTION_CACHE[cache_key] = self._connect_uncached()
        self._connected = True
        return self

    def _set_log_channel(self, name: str) -> None:
        if False:
            return 10
        'Mimic paramiko.SSHClient.set_log_channel'
        self._log_channel = name

    def _parse_proxy_command(self, port: int=22) -> dict[str, t.Any]:
        if False:
            while True:
                i = 10
        proxy_command = None
        ssh_args = [self.get_option('ssh_extra_args'), self.get_option('ssh_common_args'), self.get_option('ssh_args', '')]
        args = self._split_ssh_args(' '.join(ssh_args))
        for (i, arg) in enumerate(args):
            if arg.lower() == 'proxycommand':
                proxy_command = args[i + 1]
            else:
                match = SETTINGS_REGEX.match(arg)
                if match:
                    if match.group(1).lower() == 'proxycommand':
                        proxy_command = match.group(2)
            if proxy_command:
                break
        proxy_command = self.get_option('proxy_command') or proxy_command
        sock_kwarg = {}
        if proxy_command:
            replacers = {'%h': self.get_option('remote_addr'), '%p': port, '%r': self.get_option('remote_user')}
            for (find, replace) in replacers.items():
                proxy_command = proxy_command.replace(find, str(replace))
            try:
                sock_kwarg = {'sock': paramiko.ProxyCommand(proxy_command)}
                display.vvv('CONFIGURE PROXY COMMAND FOR CONNECTION: %s' % proxy_command, host=self.get_option('remote_addr'))
            except AttributeError:
                display.warning('Paramiko ProxyCommand support unavailable. Please upgrade to Paramiko 1.9.0 or newer. Not using configured ProxyCommand')
        return sock_kwarg

    def _connect_uncached(self) -> paramiko.SSHClient:
        if False:
            for i in range(10):
                print('nop')
        ' activates the connection object '
        if paramiko is None:
            raise AnsibleError('paramiko is not installed: %s' % to_native(PARAMIKO_IMPORT_ERR))
        port = self.get_option('port')
        display.vvv('ESTABLISH PARAMIKO SSH CONNECTION FOR USER: %s on PORT %s TO %s' % (self.get_option('remote_user'), port, self.get_option('remote_addr')), host=self.get_option('remote_addr'))
        ssh = paramiko.SSHClient()
        paramiko_preferred_pubkeys = getattr(paramiko.Transport, '_preferred_pubkeys', ())
        paramiko_preferred_hostkeys = getattr(paramiko.Transport, '_preferred_keys', ())
        use_rsa_sha2_algorithms = self.get_option('use_rsa_sha2_algorithms')
        disabled_algorithms: t.Dict[str, t.Iterable[str]] = {}
        if not use_rsa_sha2_algorithms:
            if paramiko_preferred_pubkeys:
                disabled_algorithms['pubkeys'] = tuple((a for a in paramiko_preferred_pubkeys if 'rsa-sha2' in a))
            if paramiko_preferred_hostkeys:
                disabled_algorithms['keys'] = tuple((a for a in paramiko_preferred_hostkeys if 'rsa-sha2' in a))
        if self._log_channel is not None:
            ssh.set_log_channel(self._log_channel)
        self.keyfile = os.path.expanduser('~/.ssh/known_hosts')
        if self.get_option('host_key_checking'):
            for ssh_known_hosts in ('/etc/ssh/ssh_known_hosts', '/etc/openssh/ssh_known_hosts'):
                try:
                    ssh.load_system_host_keys(ssh_known_hosts)
                    break
                except IOError:
                    pass
            ssh.load_system_host_keys()
        ssh_connect_kwargs = self._parse_proxy_command(port)
        ssh.set_missing_host_key_policy(MyAddPolicy(self))
        conn_password = self.get_option('password')
        allow_agent = True
        if conn_password is not None:
            allow_agent = False
        try:
            key_filename = None
            if self.get_option('private_key_file'):
                key_filename = os.path.expanduser(self.get_option('private_key_file'))
            if LooseVersion(paramiko.__version__) >= LooseVersion('2.2.0'):
                ssh_connect_kwargs['auth_timeout'] = self.get_option('timeout')
            if LooseVersion(paramiko.__version__) >= LooseVersion('1.15.0'):
                ssh_connect_kwargs['banner_timeout'] = self.get_option('banner_timeout')
            ssh.connect(self.get_option('remote_addr').lower(), username=self.get_option('remote_user'), allow_agent=allow_agent, look_for_keys=self.get_option('look_for_keys'), key_filename=key_filename, password=conn_password, timeout=self.get_option('timeout'), port=port, disabled_algorithms=disabled_algorithms, **ssh_connect_kwargs)
        except paramiko.ssh_exception.BadHostKeyException as e:
            raise AnsibleConnectionFailure('host key mismatch for %s' % e.hostname)
        except paramiko.ssh_exception.AuthenticationException as e:
            msg = 'Failed to authenticate: {0}'.format(to_text(e))
            raise AnsibleAuthenticationFailure(msg)
        except Exception as e:
            msg = to_text(e)
            if u'PID check failed' in msg:
                raise AnsibleError('paramiko version issue, please upgrade paramiko on the machine running ansible')
            elif u'Private key file is encrypted' in msg:
                msg = 'ssh %s@%s:%s : %s\nTo connect as a different user, use -u <username>.' % (self.get_option('remote_user'), self.get_options('remote_addr'), port, msg)
                raise AnsibleConnectionFailure(msg)
            else:
                raise AnsibleConnectionFailure(msg)
        return ssh

    def exec_command(self, cmd: str, in_data: bytes | None=None, sudoable: bool=True) -> tuple[int, bytes, bytes]:
        if False:
            while True:
                i = 10
        ' run a command on the remote host '
        super(Connection, self).exec_command(cmd, in_data=in_data, sudoable=sudoable)
        if in_data:
            raise AnsibleError('Internal Error: this module does not support optimized module pipelining')
        bufsize = 4096
        try:
            self.ssh.get_transport().set_keepalive(5)
            chan = self.ssh.get_transport().open_session()
        except Exception as e:
            text_e = to_text(e)
            msg = u'Failed to open session'
            if text_e:
                msg += u': %s' % text_e
            raise AnsibleConnectionFailure(to_native(msg))
        if self.get_option('pty') and sudoable:
            chan.get_pty(term=os.getenv('TERM', 'vt100'), width=int(os.getenv('COLUMNS', 0)), height=int(os.getenv('LINES', 0)))
        display.vvv('EXEC %s' % cmd, host=self.get_option('remote_addr'))
        cmd = to_bytes(cmd, errors='surrogate_or_strict')
        no_prompt_out = b''
        no_prompt_err = b''
        become_output = b''
        try:
            chan.exec_command(cmd)
            if self.become and self.become.expect_prompt():
                passprompt = False
                become_sucess = False
                while not (become_sucess or passprompt):
                    display.debug('Waiting for Privilege Escalation input')
                    chunk = chan.recv(bufsize)
                    display.debug('chunk is: %r' % chunk)
                    if not chunk:
                        if b'unknown user' in become_output:
                            n_become_user = to_native(self.become.get_option('become_user'))
                            raise AnsibleError('user %s does not exist' % n_become_user)
                        else:
                            break
                    become_output += chunk
                    for line in become_output.splitlines(True):
                        if self.become.check_success(line):
                            become_sucess = True
                            break
                        elif self.become.check_password_prompt(line):
                            passprompt = True
                            break
                if passprompt:
                    if self.become:
                        become_pass = self.become.get_option('become_pass')
                        chan.sendall(to_bytes(become_pass, errors='surrogate_or_strict') + b'\n')
                    else:
                        raise AnsibleError('A password is required but none was supplied')
                else:
                    no_prompt_out += become_output
                    no_prompt_err += become_output
        except socket.timeout:
            raise AnsibleError('ssh timed out waiting for privilege escalation.\n' + to_text(become_output))
        stdout = b''.join(chan.makefile('rb', bufsize))
        stderr = b''.join(chan.makefile_stderr('rb', bufsize))
        return (chan.recv_exit_status(), no_prompt_out + stdout, no_prompt_out + stderr)

    def put_file(self, in_path: str, out_path: str) -> None:
        if False:
            return 10
        ' transfer a file from local to remote '
        super(Connection, self).put_file(in_path, out_path)
        display.vvv('PUT %s TO %s' % (in_path, out_path), host=self.get_option('remote_addr'))
        if not os.path.exists(to_bytes(in_path, errors='surrogate_or_strict')):
            raise AnsibleFileNotFound('file or module does not exist: %s' % in_path)
        try:
            self.sftp = self.ssh.open_sftp()
        except Exception as e:
            raise AnsibleError('failed to open a SFTP connection (%s)' % e)
        try:
            self.sftp.put(to_bytes(in_path, errors='surrogate_or_strict'), to_bytes(out_path, errors='surrogate_or_strict'))
        except IOError:
            raise AnsibleError('failed to transfer file to %s' % out_path)

    def _connect_sftp(self) -> paramiko.sftp_client.SFTPClient:
        if False:
            return 10
        cache_key = '%s__%s__' % (self.get_option('remote_addr'), self.get_option('remote_user'))
        if cache_key in SFTP_CONNECTION_CACHE:
            return SFTP_CONNECTION_CACHE[cache_key]
        else:
            result = SFTP_CONNECTION_CACHE[cache_key] = self._connect().ssh.open_sftp()
            return result

    def fetch_file(self, in_path: str, out_path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' save a remote file to the specified path '
        super(Connection, self).fetch_file(in_path, out_path)
        display.vvv('FETCH %s TO %s' % (in_path, out_path), host=self.get_option('remote_addr'))
        try:
            self.sftp = self._connect_sftp()
        except Exception as e:
            raise AnsibleError('failed to open a SFTP connection (%s)' % to_native(e))
        try:
            self.sftp.get(to_bytes(in_path, errors='surrogate_or_strict'), to_bytes(out_path, errors='surrogate_or_strict'))
        except IOError:
            raise AnsibleError('failed to transfer file from %s' % in_path)

    def _any_keys_added(self) -> bool:
        if False:
            return 10
        for (hostname, keys) in self.ssh._host_keys.items():
            for (keytype, key) in keys.items():
                added_this_time = getattr(key, '_added_by_ansible_this_time', False)
                if added_this_time:
                    return True
        return False

    def _save_ssh_host_keys(self, filename: str) -> None:
        if False:
            print('Hello World!')
        "\n        not using the paramiko save_ssh_host_keys function as we want to add new SSH keys at the bottom so folks\n        don't complain about it :)\n        "
        if not self._any_keys_added():
            return
        path = os.path.expanduser('~/.ssh')
        makedirs_safe(path)
        with open(filename, 'w') as f:
            for (hostname, keys) in self.ssh._host_keys.items():
                for (keytype, key) in keys.items():
                    added_this_time = getattr(key, '_added_by_ansible_this_time', False)
                    if not added_this_time:
                        f.write('%s %s %s\n' % (hostname, keytype, key.get_base64()))
            for (hostname, keys) in self.ssh._host_keys.items():
                for (keytype, key) in keys.items():
                    added_this_time = getattr(key, '_added_by_ansible_this_time', False)
                    if added_this_time:
                        f.write('%s %s %s\n' % (hostname, keytype, key.get_base64()))

    def reset(self) -> None:
        if False:
            while True:
                i = 10
        if not self._connected:
            return
        self.close()
        self._connect()

    def close(self) -> None:
        if False:
            print('Hello World!')
        ' terminate the connection '
        cache_key = self._cache_key()
        SSH_CONNECTION_CACHE.pop(cache_key, None)
        SFTP_CONNECTION_CACHE.pop(cache_key, None)
        if hasattr(self, 'sftp'):
            if self.sftp is not None:
                self.sftp.close()
        if self.get_option('host_key_checking') and self.get_option('record_host_keys') and self._any_keys_added():
            lockfile = self.keyfile.replace('known_hosts', '.known_hosts.lock')
            dirname = os.path.dirname(self.keyfile)
            makedirs_safe(dirname)
            KEY_LOCK = open(lockfile, 'w')
            fcntl.lockf(KEY_LOCK, fcntl.LOCK_EX)
            try:
                self.ssh.load_system_host_keys()
                self.ssh._host_keys.update(self.ssh._system_host_keys)
                key_dir = os.path.dirname(self.keyfile)
                if os.path.exists(self.keyfile):
                    key_stat = os.stat(self.keyfile)
                    mode = key_stat.st_mode
                    uid = key_stat.st_uid
                    gid = key_stat.st_gid
                else:
                    mode = 33188
                    uid = os.getuid()
                    gid = os.getgid()
                tmp_keyfile = tempfile.NamedTemporaryFile(dir=key_dir, delete=False)
                os.chmod(tmp_keyfile.name, mode & 4095)
                os.chown(tmp_keyfile.name, uid, gid)
                self._save_ssh_host_keys(tmp_keyfile.name)
                tmp_keyfile.close()
                os.rename(tmp_keyfile.name, self.keyfile)
            except Exception:
                traceback.print_exc()
            fcntl.lockf(KEY_LOCK, fcntl.LOCK_UN)
        self.ssh.close()
        self._connected = False