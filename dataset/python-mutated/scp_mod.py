"""
SCP Module
==========

.. versionadded:: 2019.2.0

Module to copy files via `SCP <https://man.openbsd.org/scp>`_
"""
import logging
try:
    import paramiko
    import scp
    HAS_SCP = True
except ImportError:
    HAS_SCP = False
__proxyenabled__ = ['*']
__virtualname__ = 'scp'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    if not HAS_SCP:
        return (False, 'Please install SCP for this modules: pip install scp')
    return __virtualname__

def _select_kwargs(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    paramiko_kwargs = {}
    scp_kwargs = {}
    paramiko_args = __utils__['args.get_function_argspec'](paramiko.SSHClient.connect)[0]
    paramiko_args.append('auto_add_policy')
    scp_args = __utils__['args.get_function_argspec'](scp.SCPClient.__init__)[0]
    scp_args.pop(0)
    for (key, val) in kwargs.items():
        if key in paramiko_args and val is not None:
            paramiko_kwargs[key] = val
        if key in scp_args and val is not None:
            scp_kwargs[key] = val
    return (paramiko_kwargs, scp_kwargs)

def _prepare_connection(**kwargs):
    if False:
        print('Hello World!')
    '\n    Prepare the underlying SSH connection with the remote target.\n    '
    (paramiko_kwargs, scp_kwargs) = _select_kwargs(**kwargs)
    ssh = paramiko.SSHClient()
    if paramiko_kwargs.pop('auto_add_policy', False):
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(**paramiko_kwargs)
    scp_client = scp.SCPClient(ssh.get_transport(), **scp_kwargs)
    return scp_client

def get(remote_path, local_path='', recursive=False, preserve_times=False, **kwargs):
    if False:
        print('Hello World!')
    "\n    Transfer files and directories from remote host to the localhost of the\n    Minion.\n\n    remote_path\n        Path to retrieve from remote host. Since this is evaluated by scp on the\n        remote host, shell wildcards and environment variables may be used.\n\n    recursive: ``False``\n        Transfer files and directories recursively.\n\n    preserve_times: ``False``\n        Preserve ``mtime`` and ``atime`` of transferred files and directories.\n\n    hostname\n        The hostname of the remote device.\n\n    port: ``22``\n        The port of the remote device.\n\n    username\n        The username required for SSH authentication on the device.\n\n    password\n        Used for password authentication. It is also used for private key\n        decryption if ``passphrase`` is not given.\n\n    passphrase\n        Used for decrypting private keys.\n\n    pkey\n        An optional private key to use for authentication.\n\n    key_filename\n        The filename, or list of filenames, of optional private key(s) and/or\n        certificates to try for authentication.\n\n    timeout\n        An optional timeout (in seconds) for the TCP connect.\n\n    socket_timeout: ``10``\n        The channel socket timeout in seconds.\n\n    buff_size: ``16384``\n        The size of the SCP send buffer.\n\n    allow_agent: ``True``\n        Set to ``False`` to disable connecting to the SSH agent.\n\n    look_for_keys: ``True``\n        Set to ``False`` to disable searching for discoverable private key\n        files in ``~/.ssh/``\n\n    banner_timeout\n        An optional timeout (in seconds) to wait for the SSH banner to be\n        presented.\n\n    auth_timeout\n        An optional timeout (in seconds) to wait for an authentication\n        response.\n\n    auto_add_policy: ``False``\n        Automatically add the host to the ``known_hosts``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' scp.get /var/tmp/file /tmp/file hostname=10.10.10.1 auto_add_policy=True\n    "
    scp_client = _prepare_connection(**kwargs)
    get_kwargs = {'recursive': recursive, 'preserve_times': preserve_times}
    if local_path:
        get_kwargs['local_path'] = local_path
    return scp_client.get(remote_path, **get_kwargs)

def put(files, remote_path=None, recursive=False, preserve_times=False, saltenv='base', **kwargs):
    if False:
        return 10
    "\n    Transfer files and directories to remote host.\n\n    files\n        A single path or a list of paths to be transferred.\n\n    remote_path\n        The path on the remote device where to store the files.\n\n    recursive: ``True``\n        Transfer files and directories recursively.\n\n    preserve_times: ``False``\n        Preserve ``mtime`` and ``atime`` of transferred files and directories.\n\n    hostname\n        The hostname of the remote device.\n\n    port: ``22``\n        The port of the remote device.\n\n    username\n        The username required for SSH authentication on the device.\n\n    password\n        Used for password authentication. It is also used for private key\n        decryption if ``passphrase`` is not given.\n\n    passphrase\n        Used for decrypting private keys.\n\n    pkey\n        An optional private key to use for authentication.\n\n    key_filename\n        The filename, or list of filenames, of optional private key(s) and/or\n        certificates to try for authentication.\n\n    timeout\n        An optional timeout (in seconds) for the TCP connect.\n\n    socket_timeout: ``10``\n        The channel socket timeout in seconds.\n\n    buff_size: ``16384``\n        The size of the SCP send buffer.\n\n    allow_agent: ``True``\n        Set to ``False`` to disable connecting to the SSH agent.\n\n    look_for_keys: ``True``\n        Set to ``False`` to disable searching for discoverable private key\n        files in ``~/.ssh/``\n\n    banner_timeout\n        An optional timeout (in seconds) to wait for the SSH banner to be\n        presented.\n\n    auth_timeout\n        An optional timeout (in seconds) to wait for an authentication\n        response.\n\n    auto_add_policy: ``False``\n        Automatically add the host to the ``known_hosts``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' scp.put /path/to/file /var/tmp/file hostname=server1 auto_add_policy=True\n    "
    scp_client = _prepare_connection(**kwargs)
    put_kwargs = {'recursive': recursive, 'preserve_times': preserve_times}
    if remote_path:
        put_kwargs['remote_path'] = remote_path
    cached_files = []
    if not isinstance(files, (list, tuple)):
        files = [files]
    for file_ in files:
        cached_file = __salt__['cp.cache_file'](file_, saltenv=saltenv)
        cached_files.append(cached_file)
    return scp_client.put(cached_files, **put_kwargs)