"""
Manage Dell DRAC from the Master

The login credentials need to be configured in the Salt master
configuration file.

.. code-block:: yaml

    drac:
      username: admin
      password: secret

"""
import logging
try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    if HAS_PARAMIKO:
        return True
    return (False, 'The drac runner module cannot be loaded: paramiko package is not installed.')

def __connect(hostname, timeout=20, username=None, password=None):
    if False:
        i = 10
        return i + 15
    '\n    Connect to the DRAC\n    '
    drac_cred = __opts__.get('drac')
    err_msg = "No drac login credentials found. Please add the 'username' and 'password' fields beneath a 'drac' key in the master configuration file. Or you can pass in a username and password as kwargs at the CLI."
    if not username:
        if drac_cred is None:
            log.error(err_msg)
            return False
        username = drac_cred.get('username', None)
    if not password:
        if drac_cred is None:
            log.error(err_msg)
            return False
        password = drac_cred.get('password', None)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname, username=username, password=password, timeout=timeout)
    except Exception as e:
        log.error('Unable to connect to %s: %s', hostname, e)
        return False
    return client

def __version(client):
    if False:
        print('Hello World!')
    '\n    Grab DRAC version\n    '
    versions = {9: 'CMC', 8: 'iDRAC6', 10: 'iDRAC6', 11: 'iDRAC6', 16: 'iDRAC7', 17: 'iDRAC7'}
    if isinstance(client, paramiko.SSHClient):
        (stdin, stdout, stderr) = client.exec_command('racadm getconfig -g idRacInfo')
        for i in stdout.readlines():
            if i[2:].startswith('idRacType'):
                return versions.get(int(i[2:].split('=')[1]), None)
    return None

def pxe(hostname, timeout=20, username=None, password=None):
    if False:
        return 10
    '\n    Connect to the Dell DRAC and have the boot order set to PXE\n    and power cycle the system to PXE boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run drac.pxe example.com\n    '
    _cmds = ['racadm config -g cfgServerInfo -o cfgServerFirstBootDevice pxe', 'racadm config -g cfgServerInfo -o cfgServerBootOnce 1', 'racadm serveraction powercycle']
    client = __connect(hostname, timeout, username, password)
    if isinstance(client, paramiko.SSHClient):
        for (i, cmd) in enumerate(_cmds, 1):
            log.info('Executing command %s', i)
            (stdin, stdout, stderr) = client.exec_command(cmd)
        if 'successful' in stdout.readline():
            log.info('Executing command: %s', cmd)
        else:
            log.error('Unable to execute: %s', cmd)
            return False
    return True

def reboot(hostname, timeout=20, username=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reboot a server using the Dell DRAC\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run drac.reboot example.com\n    '
    client = __connect(hostname, timeout, username, password)
    if isinstance(client, paramiko.SSHClient):
        (stdin, stdout, stderr) = client.exec_command('racadm serveraction powercycle')
        if 'successful' in stdout.readline():
            log.info('powercycle successful')
        else:
            log.error('powercycle racadm command failed')
            return False
    else:
        log.error('client was not of type paramiko.SSHClient')
        return False
    return True

def poweroff(hostname, timeout=20, username=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Power server off\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run drac.poweroff example.com\n    '
    client = __connect(hostname, timeout, username, password)
    if isinstance(client, paramiko.SSHClient):
        (stdin, stdout, stderr) = client.exec_command('racadm serveraction powerdown')
        if 'successful' in stdout.readline():
            log.info('powerdown successful')
        else:
            log.error('powerdown racadm command failed')
            return False
    else:
        log.error('client was not of type paramiko.SSHClient')
        return False
    return True

def poweron(hostname, timeout=20, username=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Power server on\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run drac.poweron example.com\n    '
    client = __connect(hostname, timeout, username, password)
    if isinstance(client, paramiko.SSHClient):
        (stdin, stdout, stderr) = client.exec_command('racadm serveraction powerup')
        if 'successful' in stdout.readline():
            log.info('powerup successful')
        else:
            log.error('powerup racadm command failed')
            return False
    else:
        log.error('client was not of type paramiko.SSHClient')
        return False
    return True

def version(hostname, timeout=20, username=None, password=None):
    if False:
        return 10
    '\n    Display the version of DRAC\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run drac.version example.com\n    '
    return __version(__connect(hostname, timeout, username, password))