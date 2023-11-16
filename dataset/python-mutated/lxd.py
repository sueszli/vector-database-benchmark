"""
Module for managing the LXD daemon and its containers.

.. versionadded:: 2019.2.0

`LXD(1)`_ is a container "hypervisor". This execution module provides
several functions to help manage it and its containers.

.. note::

    - `pylxd(2)`_ version >=2.2.5 is required to let this work,
      currently only available via pip.

        To install on Ubuntu:

        $ apt-get install libssl-dev python-pip
        $ pip install -U pylxd

    - you need lxd installed on the minion
      for the init() and version() methods.

    - for the config_get() and config_get() methods
      you need to have lxd-client installed.

.. _LXD(1): https://linuxcontainers.org/lxd/
.. _pylxd(2): https://github.com/lxc/pylxd/blob/master/doc/source/installation.rst

:maintainer: René Jochum <rene@jochums.at>
:maturity: new
:depends: python-pylxd
:platform: Linux
"""
import logging
import os
from datetime import datetime
import salt.utils.decorators.path
import salt.utils.files
from salt.exceptions import CommandExecutionError, SaltInvocationError
from salt.utils.versions import Version
try:
    import pylxd
    HAS_PYLXD = True
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    HAS_PYLXD = False
log = logging.getLogger(__name__)
__docformat__ = 'restructuredtext en'
_pylxd_minimal_version = '2.2.5'
_architectures = {'unknown': '0', 'i686': '1', 'x86_64': '2', 'armv7l': '3', 'aarch64': '4', 'ppc': '5', 'ppc64': '6', 'ppc64le': '7', 's390x': '8'}
CONTAINER_STATUS_RUNNING = 103
__virtualname__ = 'lxd'
_connection_pool = {}

def __virtual__():
    if False:
        while True:
            i = 10
    if HAS_PYLXD:
        if Version(pylxd_version()) < Version(_pylxd_minimal_version):
            return (False, 'The lxd execution module cannot be loaded: pylxd "{}" is not supported, you need at least pylxd "{}"'.format(pylxd_version(), _pylxd_minimal_version))
        return __virtualname__
    return (False, 'The lxd execution module cannot be loaded: the pylxd python module is not available.')

@salt.utils.decorators.path.which('lxd')
def version():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the actual lxd version.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.version\n\n    "
    return __salt__['cmd.run']('lxd --version')

def pylxd_version():
    if False:
        return 10
    "\n    Returns the actual pylxd version.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.pylxd_version\n\n    "
    return pylxd.__version__

@salt.utils.decorators.path.which('lxd')
def init(storage_backend='dir', trust_password=None, network_address=None, network_port=None, storage_create_device=None, storage_create_loop=None, storage_pool=None):
    if False:
        while True:
            i = 10
    "\n    Calls lxd init --auto -- opts\n\n    storage_backend :\n        Storage backend to use (zfs or dir, default: dir)\n\n    trust_password :\n        Password required to add new clients\n\n    network_address : None\n        Address to bind LXD to (default: none)\n\n    network_port : None\n        Port to bind LXD to (Default: 8443)\n\n    storage_create_device : None\n        Setup device based storage using this DEVICE\n\n    storage_create_loop : None\n        Setup loop based storage with this SIZE in GB\n\n    storage_pool : None\n        Storage pool to use or create\n\n    CLI Examples:\n\n    To listen on all IPv4/IPv6 Addresses:\n\n    .. code-block:: bash\n\n        salt '*' lxd.init dir PaSsW0rD [::]\n\n    To not listen on Network:\n\n    .. code-block:: bash\n\n        salt '*' lxd.init\n    "
    cmd = 'lxd init --auto --storage-backend="{}"'.format(storage_backend)
    if trust_password is not None:
        cmd = cmd + ' --trust-password="{}"'.format(trust_password)
    if network_address is not None:
        cmd = cmd + ' --network-address="{}"'.format(network_address)
    if network_port is not None:
        cmd = cmd + ' --network-port="{}"'.format(network_port)
    if storage_create_device is not None:
        cmd = cmd + ' --storage-create-device="{}"'.format(storage_create_device)
    if storage_create_loop is not None:
        cmd = cmd + ' --storage-create-loop="{}"'.format(storage_create_loop)
    if storage_pool is not None:
        cmd = cmd + ' --storage-pool="{}"'.format(storage_pool)
    try:
        output = __salt__['cmd.run'](cmd)
    except ValueError as e:
        raise CommandExecutionError("Failed to call: '{}', error was: {}".format(cmd, str(e)))
    if 'error:' in output:
        raise CommandExecutionError(output[output.index('error:') + 7:])
    return output

@salt.utils.decorators.path.which('lxd')
@salt.utils.decorators.path.which('lxc')
def config_set(key, value):
    if False:
        print('Hello World!')
    "\n    Set an LXD daemon config option\n\n    CLI Examples:\n\n    To listen on IPv4 and IPv6 port 8443,\n    you can omit the :8443 its the default:\n\n    .. code-block:: bash\n\n        salt '*' lxd.config_set core.https_address [::]:8443\n\n    To set the server trust password:\n\n    .. code-block:: bash\n\n        salt '*' lxd.config_set core.trust_password blah\n\n    "
    cmd = 'lxc config set "{}" "{}"'.format(key, value)
    output = __salt__['cmd.run'](cmd)
    if 'error:' in output:
        raise CommandExecutionError(output[output.index('error:') + 7:])
    return ('Config value "{}" successfully set.'.format(key),)

@salt.utils.decorators.path.which('lxd')
@salt.utils.decorators.path.which('lxc')
def config_get(key):
    if False:
        i = 10
        return i + 15
    "\n    Get an LXD daemon config option\n\n    key :\n        The key of the config value to retrieve\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.config_get core.https_address\n    "
    cmd = 'lxc config get "{}"'.format(key)
    output = __salt__['cmd.run'](cmd)
    if 'error:' in output:
        raise CommandExecutionError(output[output.index('error:') + 7:])
    return output

def pylxd_client_get(remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        print('Hello World!')
    '\n    Get an pyxld client, this is not meant to be run over the CLI.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    See the `requests-docs`_ for the SSL stuff.\n\n    .. _requests-docs: http://docs.python-requests.org/en/master/user/advanced/#ssl-cert-verification\n\n    '
    pool_key = '|'.join((str(remote_addr), str(cert), str(key), str(verify_cert)))
    if pool_key in _connection_pool:
        log.debug('Returning the client "%s" from our connection pool', remote_addr)
        return _connection_pool[pool_key]
    try:
        if remote_addr is None or remote_addr == '/var/lib/lxd/unix.socket':
            log.debug('Trying to connect to the local unix socket')
            client = pylxd.Client()
        elif remote_addr.startswith('/'):
            client = pylxd.Client(remote_addr)
        else:
            if cert is None or key is None:
                raise SaltInvocationError('You have to give a Cert and Key file for remote endpoints.')
            cert = os.path.expanduser(cert)
            key = os.path.expanduser(key)
            if not os.path.isfile(cert):
                raise SaltInvocationError('You have given an invalid cert path: "{}", the file does not exist or is not a file.'.format(cert))
            if not os.path.isfile(key):
                raise SaltInvocationError('You have given an invalid key path: "{}", the file does not exists or is not a file.'.format(key))
            log.debug('Trying to connect to "%s" with cert "%s", key "%s" and verify_cert "%s"', remote_addr, cert, key, verify_cert)
            client = pylxd.Client(endpoint=remote_addr, cert=(cert, key), verify=verify_cert)
    except pylxd.exceptions.ClientConnectionFailed:
        raise CommandExecutionError("Failed to connect to '{}'".format(remote_addr))
    except TypeError as e:
        raise CommandExecutionError('Failed to connect to "{}", looks like the SSL verification failed, error was: {}'.format(remote_addr, str(e)))
    _connection_pool[pool_key] = client
    return client

def pylxd_save_object(obj):
    if False:
        for i in range(10):
            print('nop')
    'Saves an object (profile/image/container) and\n        translate its execpetion on failure\n\n    obj :\n        The object to save\n\n    This is an internal method, no CLI Example.\n    '
    try:
        obj.save(wait=True)
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    return True

def authenticate(remote_addr, password, cert, key, verify_cert=True):
    if False:
        i = 10
        return i + 15
    "\n    Authenticate with a remote LXDaemon.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n\n    password :\n        The password of the remote.\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.authenticate https://srv01:8443 <yourpass> ~/.config/lxc/client.crt ~/.config/lxc/client.key false\n\n    See the `requests-docs`_ for the SSL stuff.\n\n    .. _requests-docs: http://docs.python-requests.org/en/master/user/advanced/#ssl-cert-verification\n\n    "
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    if client.trusted:
        return True
    try:
        client.authenticate(password)
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    return client.trusted

def container_list(list_names=False, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    "\n    Lists containers\n\n    list_names : False\n        Only return a list of names when True\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Examples:\n\n    Full dict with all available information:\n\n    .. code-block:: bash\n\n        salt '*' lxd.container_list\n\n    For a list of names:\n\n    .. code-block:: bash\n\n        salt '*' lxd.container_list true\n\n    See also `container-attributes`_.\n\n    .. _container-attributes: https://github.com/lxc/pylxd/blob/master/doc/source/containers.rst#container-attributes\n\n    "
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    containers = client.containers.all()
    if list_names:
        return [c.name for c in containers]
    return map(_pylxd_model_to_dict, containers)

def container_create(name, source, profiles=None, config=None, devices=None, architecture='x86_64', ephemeral=False, wait=True, remote_addr=None, cert=None, key=None, verify_cert=True, _raw=False):
    if False:
        i = 10
        return i + 15
    '\n    Create a container\n\n    name :\n        The name of the container\n\n    source :\n        Can be either a string containing an image alias:\n             "xenial/amd64"\n\n        or an dict with type "image" with alias:\n            {"type": "image",\n             "alias": "xenial/amd64"}\n\n        or image with "fingerprint":\n            {"type": "image",\n             "fingerprint": "SHA-256"}\n\n        or image with "properties":\n            {"type": "image",\n             "properties": {\n                "os": "ubuntu",\n                "release": "14.04",\n                "architecture": "x86_64"}}\n\n        or none:\n            {"type": "none"}\n\n        or copy:\n            {"type": "copy",\n             "source": "my-old-container"}\n\n    profiles : [\'default\']\n        List of profiles to apply on this container\n\n    config :\n        A config dict or None (None = unset).\n\n        Can also be a list:\n            [{\'key\': \'boot.autostart\', \'value\': 1},\n             {\'key\': \'security.privileged\', \'value\': \'1\'}]\n\n    devices :\n        A device dict or None (None = unset).\n\n    architecture : \'x86_64\'\n        Can be one of the following:\n            * unknown\n            * i686\n            * x86_64\n            * armv7l\n            * aarch64\n            * ppc\n            * ppc64\n            * ppc64le\n            * s390x\n\n    ephemeral : False\n        Destroy this container after stop?\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    _raw : False\n        Return the raw pyxld object or a dict?\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.container_create test xenial/amd64\n\n    See also the `rest-api-docs`_.\n\n    .. _rest-api-docs: https://github.com/lxc/lxd/blob/master/doc/rest-api.md#post-1\n\n    '
    if profiles is None:
        profiles = ['default']
    if config is None:
        config = {}
    if devices is None:
        devices = {}
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    if not isinstance(profiles, (list, tuple, set)):
        raise SaltInvocationError("'profiles' must be formatted as list/tuple/set.")
    if architecture not in _architectures:
        raise SaltInvocationError("Unknown architecture '{}' given for the container '{}'".format(architecture, name))
    if isinstance(source, str):
        source = {'type': 'image', 'alias': source}
    (config, devices) = normalize_input_values(config, devices)
    try:
        container = client.containers.create({'name': name, 'architecture': _architectures[architecture], 'profiles': profiles, 'source': source, 'config': config, 'ephemeral': ephemeral}, wait=wait)
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    if not wait:
        return container.json()['operation']
    if devices:
        for (dn, dargs) in devices.items():
            if 'type' in dargs:
                container_device_add(name, dn, device_type=dargs['type'], **dargs)
            else:
                container_device_add(name, dn, **dargs)
    if _raw:
        return container
    return _pylxd_model_to_dict(container)

def container_get(name=None, remote_addr=None, cert=None, key=None, verify_cert=True, _raw=False):
    if False:
        print('Hello World!')
    'Gets a container from the LXD\n\n    name :\n        The name of the container to get.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    _raw :\n        Return the pylxd object, this is internal and by states in use.\n    '
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    if name is None:
        containers = client.containers.all()
        if _raw:
            return containers
    else:
        containers = []
        try:
            containers = [client.containers.get(name)]
        except pylxd.exceptions.LXDAPIException:
            raise SaltInvocationError("Container '{}' not found".format(name))
        if _raw:
            return containers[0]
    infos = []
    for container in containers:
        infos.append(dict([(container.name, _pylxd_model_to_dict(container))]))
    return infos

def container_delete(name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    '\n    Delete a container\n\n    name :\n        Name of the container to delete\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    container.delete(wait=True)
    return True

def container_rename(name, newname, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    '\n    Rename a container\n\n    name :\n        Name of the container to Rename\n\n    newname :\n        The new name of the container\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    if container.status_code == CONTAINER_STATUS_RUNNING:
        raise SaltInvocationError("Can't rename the running container '{}'.".format(name))
    container.rename(newname, wait=True)
    return _pylxd_model_to_dict(container)

def container_state(name=None, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get container state\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    if name is None:
        containers = client.containers.all()
    else:
        try:
            containers = [client.containers.get(name)]
        except pylxd.exceptions.LXDAPIException:
            raise SaltInvocationError("Container '{}' not found".format(name))
    states = []
    for container in containers:
        state = {}
        state = container.state()
        states.append(dict([(container.name, {k: getattr(state, k) for k in dir(state) if not k.startswith('_')})]))
    return states

def container_start(name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        print('Hello World!')
    '\n    Start a container\n\n    name :\n        Name of the container to start\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    container.start(wait=True)
    return _pylxd_model_to_dict(container)

def container_stop(name, timeout=30, force=True, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        print('Hello World!')
    '\n    Stop a container\n\n    name :\n        Name of the container to stop\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    container.stop(timeout, force, wait=True)
    return _pylxd_model_to_dict(container)

def container_restart(name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    '\n    Restart a container\n\n    name :\n        Name of the container to restart\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    container.restart(wait=True)
    return _pylxd_model_to_dict(container)

def container_freeze(name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Freeze a container\n\n    name :\n        Name of the container to freeze\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    container.freeze(wait=True)
    return _pylxd_model_to_dict(container)

def container_unfreeze(name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Unfreeze a container\n\n    name :\n        Name of the container to unfreeze\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    container.unfreeze(wait=True)
    return _pylxd_model_to_dict(container)

def container_migrate(name, stop_and_start=False, remote_addr=None, cert=None, key=None, verify_cert=True, src_remote_addr=None, src_cert=None, src_key=None, src_verify_cert=None):
    if False:
        i = 10
        return i + 15
    "Migrate a container.\n\n    If the container is running, it either must be shut down\n    first (use stop_and_start=True) or criu must be installed\n    on the source and destination machines.\n\n    For this operation both certs need to be authenticated,\n    use :mod:`lxd.authenticate <salt.modules.lxd.authenticate`\n    to authenticate your cert(s).\n\n    name :\n        Name of the container to migrate\n\n    stop_and_start :\n        Stop the container on the source and start it on dest\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Authorize\n        salt '*' lxd.authenticate https://srv01:8443 <yourpass> ~/.config/lxc/client.crt ~/.config/lxc/client.key false\n        salt '*' lxd.authenticate https://srv02:8443 <yourpass> ~/.config/lxc/client.crt ~/.config/lxc/client.key false\n\n        # Migrate phpmyadmin from srv01 to srv02\n        salt '*' lxd.container_migrate phpmyadmin stop_and_start=true remote_addr=https://srv02:8443 cert=~/.config/lxc/client.crt key=~/.config/lxc/client.key verify_cert=False src_remote_addr=https://srv01:8443\n    "
    if src_cert is None:
        src_cert = cert
    if src_key is None:
        src_key = key
    if src_verify_cert is None:
        src_verify_cert = verify_cert
    container = container_get(name, src_remote_addr, src_cert, src_key, src_verify_cert, _raw=True)
    dest_client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    for pname in container.profiles:
        try:
            dest_client.profiles.get(pname)
        except pylxd.exceptions.LXDAPIException:
            raise SaltInvocationError('not all the profiles from the source exist on the target')
    was_running = container.status_code == CONTAINER_STATUS_RUNNING
    if stop_and_start and was_running:
        container.stop(wait=True)
    try:
        dest_container = container.migrate(dest_client, wait=True)
        dest_container.profiles = container.profiles
        dest_container.save()
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    container.delete(wait=True)
    if stop_and_start and was_running:
        dest_container.start(wait=True)
    return _pylxd_model_to_dict(dest_container)

def container_config_get(name, config_key, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    '\n    Get a container config value\n\n    name :\n        Name of the container\n\n    config_key :\n        The config key to retrieve\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _get_property_dict_item(container, 'config', config_key)

def container_config_set(name, config_key, config_value, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set a container config value\n\n    name :\n        Name of the container\n\n    config_key :\n        The config key to set\n\n    config_value :\n        The config value to set\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _set_property_dict_item(container, 'config', config_key, config_value)

def container_config_delete(name, config_key, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    '\n    Delete a container config value\n\n    name :\n        Name of the container\n\n    config_key :\n        The config key to delete\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _delete_property_dict_item(container, 'config', config_key)

def container_device_get(name, device_name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    '\n    Get a container device\n\n    name :\n        Name of the container\n\n    device_name :\n        The device name to retrieve\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _get_property_dict_item(container, 'devices', device_name)

def container_device_add(name, device_name, device_type='disk', remote_addr=None, cert=None, key=None, verify_cert=True, **kwargs):
    if False:
        print('Hello World!')
    '\n    Add a container device\n\n    name :\n        Name of the container\n\n    device_name :\n        The device name to add\n\n    device_type :\n        Type of the device\n\n    ** kwargs :\n        Additional device args\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    kwargs['type'] = device_type
    return _set_property_dict_item(container, 'devices', device_name, kwargs)

def container_device_delete(name, device_name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        return 10
    '\n    Delete a container device\n\n    name :\n        Name of the container\n\n    device_name :\n        The device name to delete\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _delete_property_dict_item(container, 'devices', device_name)

def container_file_put(name, src, dst, recursive=False, overwrite=False, mode=None, uid=None, gid=None, saltenv='base', remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        print('Hello World!')
    "\n    Put a file into a container\n\n    name :\n        Name of the container\n\n    src :\n        The source file or directory\n\n    dst :\n        The destination file or directory\n\n    recursive :\n        Decent into src directory\n\n    overwrite :\n        Replace destination if it exists\n\n    mode :\n        Set file mode to octal number\n\n    uid :\n        Set file uid (owner)\n\n    gid :\n        Set file gid (group)\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.container_file_put <container name> /var/tmp/foo /var/tmp/\n\n    "
    mode = str(mode)
    if not mode.startswith('0'):
        mode = '0{}'.format(mode)
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    src = os.path.expanduser(src)
    if not os.path.isabs(src):
        if src.find('://') >= 0:
            cached_file = __salt__['cp.cache_file'](src, saltenv=saltenv)
            if not cached_file:
                raise SaltInvocationError("File '{}' not found".format(src))
            if not os.path.isabs(cached_file):
                raise SaltInvocationError('File path must be absolute.')
            src = cached_file
    src = src.rstrip(os.path.sep)
    if not src:
        src = os.path.sep
    if not os.path.exists(src):
        raise CommandExecutionError("No such file or directory '{}'".format(src))
    if os.path.isdir(src) and (not recursive):
        raise SaltInvocationError('Cannot copy overwriting a directory without recursive flag set to true!')
    try:
        dst_is_directory = False
        container.files.get(os.path.join(dst, '.'))
    except pylxd.exceptions.NotFound:
        pass
    except pylxd.exceptions.LXDAPIException as why:
        if str(why).find('Is a directory') >= 0:
            dst_is_directory = True
    if os.path.isfile(src):
        if dst_is_directory:
            dst = os.path.join(dst, os.path.basename(src))
            if not overwrite:
                found = True
                try:
                    container.files.get(os.path.join(dst))
                except pylxd.exceptions.NotFound:
                    found = False
                except pylxd.exceptions.LXDAPIException as why:
                    if str(why).find('not found') >= 0:
                        found = False
                    else:
                        raise
                if found:
                    raise SaltInvocationError('Destination exists and overwrite is false')
        if mode is not None or uid is not None or gid is not None:
            stat = os.stat(src)
            if mode is None:
                mode = oct(stat.st_mode)
            if uid is None:
                uid = stat.st_uid
            if gid is None:
                gid = stat.st_gid
        with salt.utils.files.fopen(src, 'rb') as src_fp:
            container.files.put(dst, src_fp.read(), mode=mode, uid=uid, gid=gid)
        return True
    elif not os.path.isdir(src):
        raise SaltInvocationError('Source is neither file nor directory')
    if dst.endswith(os.sep):
        idx = len(os.path.dirname(src))
    elif dst_is_directory:
        idx = len(src)
    else:
        try:
            container.files.get(os.path.join(os.path.dirname(dst), '.'))
        except pylxd.exceptions.NotFound:
            pass
        except pylxd.exceptions.LXDAPIException as why:
            if str(why).find('Is a directory') >= 0:
                dst_is_directory = True
                idx = len(src)
                overwrite = True
    if not overwrite:
        raise SaltInvocationError('Destination exists and overwrite is false')
    dstdirs = []
    for (path, _, files) in os.walk(src):
        dstdir = os.path.join(dst, path[idx:].lstrip(os.path.sep))
        dstdirs.append(dstdir)
    container.execute(['mkdir', '-p'] + dstdirs)
    set_mode = mode
    set_uid = uid
    set_gid = gid
    for (path, _, files) in os.walk(src):
        dstdir = os.path.join(dst, path[idx:].lstrip(os.path.sep))
        for name in files:
            src_name = os.path.join(path, name)
            dst_name = os.path.join(dstdir, name)
            if mode is not None or uid is not None or gid is not None:
                stat = os.stat(src_name)
                if mode is None:
                    set_mode = oct(stat.st_mode)
                if uid is None:
                    set_uid = stat.st_uid
                if gid is None:
                    set_gid = stat.st_gid
            with salt.utils.files.fopen(src_name, 'rb') as src_fp:
                container.files.put(dst_name, src_fp.read(), mode=set_mode, uid=set_uid, gid=set_gid)
    return True

def container_file_get(name, src, dst, overwrite=False, mode=None, uid=None, gid=None, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    '\n    Get a file from a container\n\n    name :\n        Name of the container\n\n    src :\n        The source file or directory\n\n    dst :\n        The destination file or directory\n\n    mode :\n        Set file mode to octal number\n\n    uid :\n        Set file uid (owner)\n\n    gid :\n        Set file gid (group)\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    '
    if mode is not None:
        mode = str(mode)
        if not mode.startswith('0'):
            mode = '0{}'.format(mode)
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    dst = os.path.expanduser(dst)
    if not os.path.isabs(dst):
        raise SaltInvocationError('File path must be absolute.')
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    elif not os.path.isdir(os.path.dirname(dst)):
        raise SaltInvocationError("Parent directory for destination doesn't exist.")
    if os.path.exists(dst):
        if not overwrite:
            raise SaltInvocationError('Destination exists and overwrite is false.')
        if not os.path.isfile(dst):
            raise SaltInvocationError('Destination exists but is not a file.')
    else:
        dst_path = os.path.dirname(dst)
        if not os.path.isdir(dst_path):
            raise CommandExecutionError("No such file or directory '{}'".format(dst_path))
    with salt.utils.files.fopen(dst, 'wb') as df:
        df.write(container.files.get(src))
    if mode:
        os.chmod(dst, mode)
    if uid or uid == '0':
        uid = int(uid)
    else:
        uid = -1
    if gid or gid == '0':
        gid = int(gid)
    else:
        gid = -1
    if uid != -1 or gid != -1:
        os.chown(dst, uid, gid)
    return True

def container_execute(name, cmd, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        return 10
    '\n    Execute a command list on a container.\n\n    name :\n        Name of the container\n\n    cmd :\n        Command to be executed (as a list)\n\n        Example :\n            \'["ls", "-l"]\'\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.container_execute <container name> \'["ls", "-l"]\'\n\n    '
    container = container_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    try:
        result = container.execute(cmd)
        saltresult = {}
        if not hasattr(result, 'exit_code'):
            saltresult = dict(exit_code=0, stdout=result[0], stderr=result[1])
        else:
            saltresult = dict(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
    except pylxd.exceptions.NotFound as e:
        saltresult = dict(exit_code=0, stdout='', stderr=str(e))
    if int(saltresult['exit_code']) > 0:
        saltresult['result'] = False
    else:
        saltresult['result'] = True
    return saltresult

def profile_list(list_names=False, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        print('Hello World!')
    "Lists all profiles from the LXD.\n\n    list_names :\n\n        Return a list of names instead of full blown dicts.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_list true --out=json\n        salt '*' lxd.profile_list --out=json\n    "
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    profiles = client.profiles.all()
    if list_names:
        return [p.name for p in profiles]
    return map(_pylxd_model_to_dict, profiles)

def profile_create(name, config=None, devices=None, description=None, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    'Creates a profile.\n\n    name :\n        The name of the profile to get.\n\n    config :\n        A config dict or None (None = unset).\n\n        Can also be a list:\n            [{\'key\': \'boot.autostart\', \'value\': 1},\n             {\'key\': \'security.privileged\', \'value\': \'1\'}]\n\n    devices :\n        A device dict or None (None = unset).\n\n    description :\n        A description string or None (None = unset).\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.profile_create autostart config="{boot.autostart: 1, boot.autostart.delay: 2, boot.autostart.priority: 1}"\n        salt \'*\' lxd.profile_create shared_mounts devices="{shared_mount: {type: \'disk\', source: \'/home/shared\', path: \'/home/shared\'}}"\n\n    See the `lxd-docs`_ for the details about the config and devices dicts.\n\n    .. _lxd-docs: https://github.com/lxc/lxd/blob/master/doc/rest-api.md#post-10\n    '
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    (config, devices) = normalize_input_values(config, devices)
    try:
        profile = client.profiles.create(name, config, devices)
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    if description is not None:
        profile.description = description
        pylxd_save_object(profile)
    return _pylxd_model_to_dict(profile)

def profile_get(name, remote_addr=None, cert=None, key=None, verify_cert=True, _raw=False):
    if False:
        return 10
    "Gets a profile from the LXD\n\n    name :\n        The name of the profile to get.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    _raw :\n        Return the pylxd object, this is internal and by states in use.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_get autostart\n    "
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    profile = None
    try:
        profile = client.profiles.get(name)
    except pylxd.exceptions.LXDAPIException:
        raise SaltInvocationError("Profile '{}' not found".format(name))
    if _raw:
        return profile
    return _pylxd_model_to_dict(profile)

def profile_delete(name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        return 10
    "Deletes a profile.\n\n    name :\n        The name of the profile to delete.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_delete shared_mounts\n    "
    profile = profile_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    profile.delete()
    return True

def profile_config_get(name, config_key, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    "Get a profile config item.\n\n    name :\n        The name of the profile to get the config item from.\n\n    config_key :\n        The key for the item to retrieve.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_config_get autostart boot.autostart\n    "
    profile = profile_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _get_property_dict_item(profile, 'config', config_key)

def profile_config_set(name, config_key, config_value, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    "Set a profile config item.\n\n    name :\n        The name of the profile to set the config item to.\n\n    config_key :\n        The items key.\n\n    config_value :\n        Its items value.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_config_set autostart boot.autostart 0\n    "
    profile = profile_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _set_property_dict_item(profile, 'config', config_key, config_value)

def profile_config_delete(name, config_key, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    "Delete a profile config item.\n\n    name :\n        The name of the profile to delete the config item.\n\n    config_key :\n        The config key for the value to retrieve.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_config_delete autostart boot.autostart.delay\n    "
    profile = profile_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _delete_property_dict_item(profile, 'config', config_key)

def profile_device_get(name, device_name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    "Get a profile device.\n\n    name :\n        The name of the profile to get the device from.\n\n    device_name :\n        The name of the device to retrieve.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_device_get default eth0\n    "
    profile = profile_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _get_property_dict_item(profile, 'devices', device_name)

def profile_device_set(name, device_name, device_type='disk', remote_addr=None, cert=None, key=None, verify_cert=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Set a profile device.\n\n    name :\n        The name of the profile to set the device to.\n\n    device_name :\n        The name of the device to set.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_device_set autostart eth1 nic nictype=bridged parent=lxdbr0\n    "
    profile = profile_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    kwargs['type'] = device_type
    for (k, v) in kwargs.items():
        kwargs[k] = str(v)
    return _set_property_dict_item(profile, 'devices', device_name, kwargs)

def profile_device_delete(name, device_name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    "Delete a profile device.\n\n    name :\n        The name of the profile to delete the device.\n\n    device_name :\n        The name of the device to delete.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lxd.profile_device_delete autostart eth1\n\n    "
    profile = profile_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    return _delete_property_dict_item(profile, 'devices', device_name)

def image_list(list_aliases=False, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        return 10
    "Lists all images from the LXD.\n\n    list_aliases :\n\n        Return a dict with the fingerprint as key and\n        a list of aliases as value instead.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.image_list true --out=json\n        salt '*' lxd.image_list --out=json\n    "
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    images = client.images.all()
    if list_aliases:
        return {i.fingerprint: [a['name'] for a in i.aliases] for i in images}
    return map(_pylxd_model_to_dict, images)

def image_get(fingerprint, remote_addr=None, cert=None, key=None, verify_cert=True, _raw=False):
    if False:
        print('Hello World!')
    "Get an image by its fingerprint\n\n    fingerprint :\n        The fingerprint of the image to retrieve\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    _raw : False\n        Return the raw pylxd object or a dict of it?\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.image_get <fingerprint>\n    "
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    image = None
    try:
        image = client.images.get(fingerprint)
    except pylxd.exceptions.LXDAPIException:
        raise SaltInvocationError("Image with fingerprint '{}' not found".format(fingerprint))
    if _raw:
        return image
    return _pylxd_model_to_dict(image)

def image_get_by_alias(alias, remote_addr=None, cert=None, key=None, verify_cert=True, _raw=False):
    if False:
        return 10
    "Get an image by an alias\n\n    alias :\n        The alias of the image to retrieve\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    _raw : False\n        Return the raw pylxd object or a dict of it?\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.image_get_by_alias xenial/amd64\n    "
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    image = None
    try:
        image = client.images.get_by_alias(alias)
    except pylxd.exceptions.LXDAPIException:
        raise SaltInvocationError("Image with alias '{}' not found".format(alias))
    if _raw:
        return image
    return _pylxd_model_to_dict(image)

def image_delete(image, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    "Delete an image by an alias or fingerprint\n\n    name :\n        The alias or fingerprint of the image to delete,\n        can be a obj for the states.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.image_delete xenial/amd64\n    "
    image = _verify_image(image, remote_addr, cert, key, verify_cert)
    image.delete()
    return True

def image_from_simplestreams(server, alias, remote_addr=None, cert=None, key=None, verify_cert=True, aliases=None, public=False, auto_update=False, _raw=False):
    if False:
        print('Hello World!')
    'Create an image from simplestreams\n\n    server :\n        Simplestreams server URI\n\n    alias :\n        The alias of the image to retrieve\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    aliases : []\n        List of aliases to append to the copied image\n\n    public : False\n        Make this image public available\n\n    auto_update : False\n        Should LXD auto update that image?\n\n    _raw : False\n        Return the raw pylxd object or a dict of the image?\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.image_from_simplestreams "https://cloud-images.ubuntu.com/releases" "trusty/amd64" aliases=\'["t", "trusty/amd64"]\' auto_update=True\n    '
    if aliases is None:
        aliases = []
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    try:
        image = client.images.create_from_simplestreams(server, alias, public=public, auto_update=auto_update)
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    for alias in aliases:
        image_alias_add(image, alias)
    if _raw:
        return image
    return _pylxd_model_to_dict(image)

def image_from_url(url, remote_addr=None, cert=None, key=None, verify_cert=True, aliases=None, public=False, auto_update=False, _raw=False):
    if False:
        print('Hello World!')
    'Create an image from an url\n\n    url :\n        The URL from where to download the image\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    aliases : []\n        List of aliases to append to the copied image\n\n    public : False\n        Make this image public available\n\n    auto_update : False\n        Should LXD auto update that image?\n\n    _raw : False\n        Return the raw pylxd object or a dict of the image?\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.image_from_url https://dl.stgraber.org/lxd aliases=\'["busybox-amd64"]\'\n    '
    if aliases is None:
        aliases = []
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    try:
        image = client.images.create_from_url(url, public=public, auto_update=auto_update)
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    for alias in aliases:
        image_alias_add(image, alias)
    if _raw:
        return image
    return _pylxd_model_to_dict(image)

def image_from_file(filename, remote_addr=None, cert=None, key=None, verify_cert=True, aliases=None, public=False, saltenv='base', _raw=False):
    if False:
        while True:
            i = 10
    'Create an image from a file\n\n    filename :\n        The filename of the rootfs\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    aliases : []\n        List of aliases to append to the copied image\n\n    public : False\n        Make this image public available\n\n    saltenv : base\n        The saltenv to use for salt:// copies\n\n    _raw : False\n        Return the raw pylxd object or a dict of the image?\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.image_from_file salt://lxd/files/busybox.tar.xz aliases=["busybox-amd64"]\n    '
    if aliases is None:
        aliases = []
    cached_file = __salt__['cp.cache_file'](filename, saltenv=saltenv)
    data = b''
    with salt.utils.files.fopen(cached_file, 'r+b') as fp:
        data = fp.read()
    client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    try:
        image = client.images.create(data, public=public, wait=True)
    except pylxd.exceptions.LXDAPIException as e:
        raise CommandExecutionError(str(e))
    for alias in aliases:
        image_alias_add(image, alias)
    if _raw:
        return image
    return _pylxd_model_to_dict(image)

def image_copy_lxd(source, src_remote_addr, src_cert, src_key, src_verify_cert, remote_addr, cert, key, verify_cert=True, aliases=None, public=None, auto_update=None, _raw=False):
    if False:
        i = 10
        return i + 15
    'Copy an image from another LXD instance\n\n    source :\n        An alias or a fingerprint of the source.\n\n    src_remote_addr :\n        An URL to the source remote daemon\n\n        Examples:\n            https://mysourceserver.lan:8443\n\n    src_cert :\n        PEM Formatted SSL Certificate for the source\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    src_key :\n        PEM Formatted SSL Key for the source\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    src_verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    remote_addr :\n        Address of the destination daemon\n\n        Examples:\n            https://mydestserver.lan:8443\n\n    cert :\n        PEM Formatted SSL Certificate for the destination\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key for the destination\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    aliases : []\n        List of aliases to append to the copied image\n\n    public : None\n        Make this image public available, None = copy source\n\n    auto_update : None\n        Wherever to auto-update from the original source, None = copy source\n\n    _raw : False\n        Return the raw pylxd object or a dict of the destination image?\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.image_copy_lxd xenial/amd64 https://srv01:8443 ~/.config/lxc/client.crt ~/.config/lxc/client.key false https://srv02:8443 ~/.config/lxc/client.crt ~/.config/lxc/client.key false aliases="[\'xenial/amd64\']"\n    '
    if aliases is None:
        aliases = []
    log.debug('Trying to copy the image "%s" from "%s" to "%s"', source, src_remote_addr, remote_addr)
    src_image = None
    try:
        src_image = image_get_by_alias(source, src_remote_addr, src_cert, src_key, src_verify_cert, _raw=True)
    except SaltInvocationError:
        src_image = image_get(source, src_remote_addr, src_cert, src_key, src_verify_cert, _raw=True)
    dest_client = pylxd_client_get(remote_addr, cert, key, verify_cert)
    dest_image = src_image.copy(dest_client, public=public, auto_update=auto_update, wait=True)
    for alias in aliases:
        image_alias_add(dest_image, alias)
    if _raw:
        return dest_image
    return _pylxd_model_to_dict(dest_image)

def image_alias_add(image, alias, description='', remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        print('Hello World!')
    'Create an alias on the given image\n\n    image :\n        An image alias, a fingerprint or a image object\n\n    alias :\n        The alias to add\n\n    description :\n        Description of the alias\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.image_alias_add xenial/amd64 x "Short version of xenial/amd64"\n    '
    image = _verify_image(image, remote_addr, cert, key, verify_cert)
    for alias_info in image.aliases:
        if alias_info['name'] == alias:
            return True
    image.add_alias(alias, description)
    return True

def image_alias_delete(image, alias, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    'Delete an alias (this is currently not restricted to the image)\n\n    image :\n        An image alias, a fingerprint or a image object\n\n    alias :\n        The alias to delete\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if\n        you provide remote_addr and its a TCP Address!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' lxd.image_alias_add xenial/amd64 x "Short version of xenial/amd64"\n    '
    image = _verify_image(image, remote_addr, cert, key, verify_cert)
    try:
        image.delete_alias(alias)
    except pylxd.exceptions.LXDAPIException:
        return False
    return True

def snapshots_all(container, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    "\n    Get all snapshots for a container\n\n    container :\n        The name of the container to get.\n\n    remote_addr :\n        An URL to a remote server. The 'cert' and 'key' fields must also be\n        provided if 'remote_addr' is defined.\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Verify the ssl certificate.  Default: True\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.snapshots_all test-container\n    "
    containers = container_get(container, remote_addr, cert, key, verify_cert, _raw=True)
    if container:
        containers = [containers]
    ret = {}
    for cont in containers:
        ret.update({cont.name: [{'name': c.name} for c in cont.snapshots.all()]})
    return ret

def snapshots_create(container, name=None, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    "\n    Create a snapshot for a container\n\n    container :\n        The name of the container to get.\n\n    name :\n        The name of the snapshot.\n\n    remote_addr :\n        An URL to a remote server. The 'cert' and 'key' fields must also be\n        provided if 'remote_addr' is defined.\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Verify the ssl certificate.  Default: True\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.snapshots_create test-container test-snapshot\n    "
    cont = container_get(container, remote_addr, cert, key, verify_cert, _raw=True)
    if not name:
        name = datetime.now().strftime('%Y%m%d%H%M%S')
    cont.snapshots.create(name)
    for c in snapshots_all(container).get(container):
        if c.get('name') == name:
            return {'name': name}
    return {'name': False}

def snapshots_delete(container, name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a snapshot for a container\n\n    container :\n        The name of the container to get.\n\n    name :\n        The name of the snapshot.\n\n    remote_addr :\n        An URL to a remote server. The 'cert' and 'key' fields must also be\n        provided if 'remote_addr' is defined.\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Verify the ssl certificate.  Default: True\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.snapshots_delete test-container test-snapshot\n    "
    cont = container_get(container, remote_addr, cert, key, verify_cert, _raw=True)
    try:
        for s in cont.snapshots.all():
            if s.name == name:
                s.delete()
                return True
    except pylxd.exceptions.LXDAPIException:
        pass
    return False

def snapshots_get(container, name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    "\n    Get information about snapshot for a container\n\n    container :\n        The name of the container to get.\n\n    name :\n        The name of the snapshot.\n\n    remote_addr :\n        An URL to a remote server. The 'cert' and 'key' fields must also be\n        provided if 'remote_addr' is defined.\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Certificate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Verify the ssl certificate.  Default: True\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.snapshots_get test-container test-snapshot\n    "
    container = container_get(container, remote_addr, cert, key, verify_cert, _raw=True)
    return container.snapshots.get(name)

def normalize_input_values(config, devices):
    if False:
        return 10
    "\n    normalize config input so returns can be put into mongodb, which doesn't like `.`\n\n    This is not meant to be used on the commandline.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lxd.normalize_input_values config={} devices={}\n    "
    if isinstance(config, list):
        if config and 'key' in config[0] and ('value' in config[0]):
            config = {d['key']: d['value'] for d in config}
        else:
            config = {}
    if isinstance(config, str):
        raise SaltInvocationError("config can't be a string, validate your YAML input.")
    if isinstance(devices, str):
        raise SaltInvocationError("devices can't be a string, validate your YAML input.")
    if config is not None:
        for (k, v) in config.items():
            config[k] = str(v)
    if devices is not None:
        for dn in devices:
            for (k, v) in devices[dn].items():
                devices[dn][k] = v
    return (config, devices)

def sync_config_devices(obj, newconfig, newdevices, test=False):
    if False:
        for i in range(10):
            print('nop')
    'Syncs the given config and devices with the object\n    (a profile or a container)\n    returns a changes dict with all changes made.\n\n    obj :\n        The object to sync with / or just test with.\n\n    newconfig:\n        The new config to check with the obj.\n\n    newdevices:\n        The new devices to check with the obj.\n\n    test:\n        Wherever to not change anything and give "Would change" message.\n    '
    changes = {}
    if newconfig is None:
        newconfig = {}
    newconfig = dict(list(zip(map(str, newconfig.keys()), map(str, newconfig.values()))))
    cck = set(newconfig.keys())
    obj.config = dict(list(zip(map(str, obj.config.keys()), map(str, obj.config.values()))))
    ock = set(obj.config.keys())
    config_changes = {}
    for k in ock.difference(cck):
        if k.startswith('volatile.') or k.startswith('image.'):
            continue
        if not test:
            config_changes[k] = 'Removed config key "{}", its value was "{}"'.format(k, obj.config[k])
            del obj.config[k]
        else:
            config_changes[k] = 'Would remove config key "{} with value "{}"'.format(k, obj.config[k])
    for k in cck.intersection(ock):
        if k.startswith('volatile.') or k.startswith('image.'):
            continue
        if newconfig[k] != obj.config[k]:
            if not test:
                config_changes[k] = 'Changed config key "{}" to "{}", its value was "{}"'.format(k, newconfig[k], obj.config[k])
                obj.config[k] = newconfig[k]
            else:
                config_changes[k] = 'Would change config key "{}" to "{}", its current value is "{}"'.format(k, newconfig[k], obj.config[k])
    for k in cck.difference(ock):
        if k.startswith('volatile.') or k.startswith('image.'):
            continue
        if not test:
            config_changes[k] = 'Added config key "{}" = "{}"'.format(k, newconfig[k])
            obj.config[k] = newconfig[k]
        else:
            config_changes[k] = 'Would add config key "{}" = "{}"'.format(k, newconfig[k])
    if config_changes:
        changes['config'] = config_changes
    if newdevices is None:
        newdevices = {}
    dk = set(obj.devices.keys())
    ndk = set(newdevices.keys())
    devices_changes = {}
    for k in dk.difference(ndk):
        if k == 'root':
            continue
        if not test:
            devices_changes[k] = 'Removed device "{}"'.format(k)
            del obj.devices[k]
        else:
            devices_changes[k] = 'Would remove device "{}"'.format(k)
    for (k, v) in obj.devices.items():
        if k == 'root':
            continue
        if k not in newdevices:
            continue
        if newdevices[k] != v:
            if not test:
                devices_changes[k] = 'Changed device "{}"'.format(k)
                obj.devices[k] = newdevices[k]
            else:
                devices_changes[k] = 'Would change device "{}"'.format(k)
    for k in ndk.difference(dk):
        if k == 'root':
            continue
        if not test:
            devices_changes[k] = 'Added device "{}"'.format(k)
            obj.devices[k] = newdevices[k]
        else:
            devices_changes[k] = 'Would add device "{}"'.format(k)
    if devices_changes:
        changes['devices'] = devices_changes
    return changes

def _set_property_dict_item(obj, prop, key, value):
    if False:
        print('Hello World!')
    'Sets the dict item key of the attr from obj.\n\n    Basicaly it does getattr(obj, prop)[key] = value.\n\n\n    For the disk device we added some checks to make\n    device changes on the CLI saver.\n    '
    attr = getattr(obj, prop)
    if prop == 'devices':
        device_type = value['type']
        if device_type == 'disk':
            if 'path' not in value:
                raise SaltInvocationError('path must be given as parameter')
            if value['path'] != '/' and 'source' not in value:
                raise SaltInvocationError('source must be given as parameter')
        for k in value.keys():
            if k.startswith('__'):
                del value[k]
        attr[key] = value
    else:
        attr[key] = str(value)
    pylxd_save_object(obj)
    return _pylxd_model_to_dict(obj)

def _get_property_dict_item(obj, prop, key):
    if False:
        print('Hello World!')
    attr = getattr(obj, prop)
    if key not in attr:
        raise SaltInvocationError("'{}' doesn't exists".format(key))
    return attr[key]

def _delete_property_dict_item(obj, prop, key):
    if False:
        i = 10
        return i + 15
    attr = getattr(obj, prop)
    if key not in attr:
        raise SaltInvocationError("'{}' doesn't exists".format(key))
    del attr[key]
    pylxd_save_object(obj)
    return True

def _verify_image(image, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        i = 10
        return i + 15
    if isinstance(image, str):
        name = image
        image = None
        try:
            image = image_get_by_alias(name, remote_addr, cert, key, verify_cert, _raw=True)
        except SaltInvocationError:
            image = image_get(name, remote_addr, cert, key, verify_cert, _raw=True)
    elif not hasattr(image, 'fingerprint'):
        raise SaltInvocationError("Invalid image '{}'".format(image))
    return image

def _pylxd_model_to_dict(obj):
    if False:
        while True:
            i = 10
    'Translates a plyxd model object to a dict'
    marshalled = {}
    for key in obj.__attributes__.keys():
        if hasattr(obj, key):
            marshalled[key] = getattr(obj, key)
    return marshalled