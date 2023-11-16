"""
Support for haproxy

.. versionadded:: 2014.7.0
"""
import logging
import os
import stat
import time
try:
    import haproxy.cmds
    import haproxy.conn
    HAS_HAPROXY = True
except ImportError:
    HAS_HAPROXY = False
log = logging.getLogger(__name__)
__virtualname__ = 'haproxy'
DEFAULT_SOCKET_URL = '/var/run/haproxy.sock'
FIELD_NUMERIC = ['weight', 'bin', 'bout']
FIELD_NODE_NAME = 'name'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load the module if haproxyctl is installed\n    '
    if HAS_HAPROXY:
        return __virtualname__
    return (False, 'The haproxyconn execution module cannot be loaded: haproxyctl module not available')

def _get_conn(socket=DEFAULT_SOCKET_URL):
    if False:
        print('Hello World!')
    '\n    Get connection to haproxy socket.\n    '
    assert os.path.exists(socket), '{} does not exist.'.format(socket)
    issock = os.stat(socket).st_mode
    assert stat.S_ISSOCK(issock), '{} is not a socket.'.format(socket)
    ha_conn = haproxy.conn.HaPConn(socket)
    return ha_conn

def list_servers(backend, socket=DEFAULT_SOCKET_URL, objectify=False):
    if False:
        print('Hello World!')
    "\n    List servers in haproxy backend.\n\n    backend\n        haproxy backend\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.list_servers mysql\n    "
    ha_conn = _get_conn(socket)
    ha_cmd = haproxy.cmds.listServers(backend=backend)
    return ha_conn.sendCmd(ha_cmd, objectify=objectify)

def wait_state(backend, server, value='up', timeout=60 * 5, socket=DEFAULT_SOCKET_URL):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Wait for a specific server state\n\n    backend\n        haproxy backend\n\n    server\n        targeted server\n\n    value\n        state value\n\n    timeout\n        timeout before giving up state value, default 5 min\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.wait_state mysql server01 up 60\n    "
    t = time.time() + timeout
    while time.time() < t:
        if get_backend(backend=backend, socket=socket)[server]['status'].lower() == value.lower():
            return True
    return False

def get_backend(backend, socket=DEFAULT_SOCKET_URL):
    if False:
        i = 10
        return i + 15
    "\n\n    Receive information about a specific backend.\n\n    backend\n        haproxy backend\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.get_backend mysql\n    "
    backend_data = list_servers(backend=backend, socket=socket).replace('\n', ' ').split(' ')
    result = {}

    def num(s):
        if False:
            print('Hello World!')
        try:
            return int(s)
        except ValueError:
            return s
    for data in backend_data:
        if ':' in data:
            active_field = data.replace(':', '').lower()
            continue
        elif active_field.lower() == FIELD_NODE_NAME:
            active_server = data
            result[active_server] = {}
            continue
        if active_field in FIELD_NUMERIC:
            if data == '':
                result[active_server][active_field] = 0
            else:
                result[active_server][active_field] = num(data)
        else:
            result[active_server][active_field] = data
    return result

def enable_server(name, backend, socket=DEFAULT_SOCKET_URL):
    if False:
        return 10
    '\n    Enable Server in haproxy\n\n    name\n        Server to enable\n\n    backend\n        haproxy backend, or all backends if "*" is supplied\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' haproxy.enable_server web1.example.com www\n    '
    if backend == '*':
        backends = show_backends(socket=socket).split('\n')
    else:
        backends = [backend]
    results = {}
    for backend in backends:
        ha_conn = _get_conn(socket)
        ha_cmd = haproxy.cmds.enableServer(server=name, backend=backend)
        ha_conn.sendCmd(ha_cmd)
        results[backend] = list_servers(backend, socket=socket)
    return results

def disable_server(name, backend, socket=DEFAULT_SOCKET_URL):
    if False:
        i = 10
        return i + 15
    '\n    Disable server in haproxy.\n\n    name\n        Server to disable\n\n    backend\n        haproxy backend, or all backends if "*" is supplied\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' haproxy.disable_server db1.example.com mysql\n    '
    if backend == '*':
        backends = show_backends(socket=socket).split('\n')
    else:
        backends = [backend]
    results = {}
    for backend in backends:
        ha_conn = _get_conn(socket)
        ha_cmd = haproxy.cmds.disableServer(server=name, backend=backend)
        ha_conn.sendCmd(ha_cmd)
        results[backend] = list_servers(backend, socket=socket)
    return results

def get_weight(name, backend, socket=DEFAULT_SOCKET_URL):
    if False:
        return 10
    "\n    Get server weight\n\n    name\n        Server name\n\n    backend\n        haproxy backend\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.get_weight web1.example.com www\n    "
    ha_conn = _get_conn(socket)
    ha_cmd = haproxy.cmds.getWeight(server=name, backend=backend)
    return ha_conn.sendCmd(ha_cmd)

def set_weight(name, backend, weight=0, socket=DEFAULT_SOCKET_URL):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set server weight\n\n    name\n        Server name\n\n    backend\n        haproxy backend\n\n    weight\n        Server Weight\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.set_weight web1.example.com www 13\n    "
    ha_conn = _get_conn(socket)
    ha_cmd = haproxy.cmds.getWeight(server=name, backend=backend, weight=weight)
    ha_conn.sendCmd(ha_cmd)
    return get_weight(name, backend, socket=socket)

def set_state(name, backend, state, socket=DEFAULT_SOCKET_URL):
    if False:
        while True:
            i = 10
    '\n    Force a server\'s administrative state to a new state. This can be useful to\n    disable load balancing and/or any traffic to a server. Setting the state to\n    "ready" puts the server in normal mode, and the command is the equivalent of\n    the "enable server" command. Setting the state to "maint" disables any traffic\n    to the server as well as any health checks. This is the equivalent of the\n    "disable server" command. Setting the mode to "drain" only removes the server\n    from load balancing but still allows it to be checked and to accept new\n    persistent connections. Changes are propagated to tracking servers if any.\n\n    name\n        Server name\n\n    backend\n        haproxy backend\n\n    state\n        A string of the state to set. Must be \'ready\', \'drain\', or \'maint\'\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' haproxy.set_state my_proxy_server my_backend ready\n\n    '

    class setServerState(haproxy.cmds.Cmd):
        """Set server state command."""
        cmdTxt = 'set server %(backend)s/%(server)s state %(value)s\r\n'
        p_args = ['backend', 'server', 'value']
        helpTxt = "Force a server's administrative state to a new state."
    ha_conn = _get_conn(socket)
    ha_cmd = setServerState(server=name, backend=backend, value=state)
    return ha_conn.sendCmd(ha_cmd)

def show_frontends(socket=DEFAULT_SOCKET_URL):
    if False:
        i = 10
        return i + 15
    "\n    Show HaProxy frontends\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.show_frontends\n    "
    ha_conn = _get_conn(socket)
    ha_cmd = haproxy.cmds.showFrontends()
    return ha_conn.sendCmd(ha_cmd)

def list_frontends(socket=DEFAULT_SOCKET_URL):
    if False:
        return 10
    "\n\n    List HaProxy frontends\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.list_frontends\n    "
    return show_frontends(socket=socket).split('\n')

def show_backends(socket=DEFAULT_SOCKET_URL):
    if False:
        while True:
            i = 10
    "\n    Show HaProxy Backends\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.show_backends\n    "
    ha_conn = _get_conn(socket)
    ha_cmd = haproxy.cmds.showBackends()
    return ha_conn.sendCmd(ha_cmd)

def list_backends(servers=True, socket=DEFAULT_SOCKET_URL):
    if False:
        return 10
    "\n\n    List HaProxy Backends\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    servers\n        list backends with servers\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.list_backends\n    "
    if not servers:
        return show_backends(socket=socket).split('\n')
    else:
        result = {}
        for backend in list_backends(servers=False, socket=socket):
            result[backend] = get_backend(backend=backend, socket=socket)
        return result

def get_sessions(name, backend, socket=DEFAULT_SOCKET_URL):
    if False:
        return 10
    "\n    .. versionadded:: 2016.11.0\n\n    Get number of current sessions on server in backend (scur)\n\n    name\n        Server name\n\n    backend\n        haproxy backend\n\n    socket\n        haproxy stats socket, default ``/var/run/haproxy.sock``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' haproxy.get_sessions web1.example.com www\n    "

    class getStats(haproxy.cmds.Cmd):
        p_args = ['backend', 'server']
        cmdTxt = 'show stat\r\n'
        helpText = 'Fetch all statistics'
    ha_conn = _get_conn(socket)
    ha_cmd = getStats(server=name, backend=backend)
    result = ha_conn.sendCmd(ha_cmd)
    for line in result.split('\n'):
        if line.startswith(backend):
            outCols = line.split(',')
            if outCols[1] == name:
                return outCols[4]