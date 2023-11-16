"""
Apache Traffic Server execution module.

.. versionadded:: 2015.8.0

``traffic_ctl`` is used to execute individual Traffic Server commands and to
script multiple commands in a shell.
"""
import logging
import subprocess
import salt.utils.path
import salt.utils.stringutils
__virtualname__ = 'trafficserver'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    if salt.utils.path.which('traffic_ctl') or salt.utils.path.which('traffic_line'):
        return __virtualname__
    return (False, 'trafficserver execution module not loaded: neither traffic_ctl nor traffic_line was found.')
_TRAFFICLINE = salt.utils.path.which('traffic_line')
_TRAFFICCTL = salt.utils.path.which('traffic_ctl')

def _traffic_ctl(*args):
    if False:
        while True:
            i = 10
    return [_TRAFFICCTL] + list(args)

def _traffic_line(*args):
    if False:
        print('Hello World!')
    return [_TRAFFICLINE] + list(args)

def _statuscmd():
    if False:
        print('Hello World!')
    if _TRAFFICCTL:
        cmd = _traffic_ctl('server', 'status')
    else:
        cmd = _traffic_line('--status')
    return _subprocess(cmd)

def _subprocess(cmd):
    if False:
        while True:
            i = 10
    '\n    Function to standardize the subprocess call\n    '
    log.debug('Running: "%s"', ' '.join(cmd))
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        ret = salt.utils.stringutils.to_unicode(proc.communicate()[0]).strip()
        retcode = proc.wait()
        if ret:
            return ret
        elif retcode != 1:
            return True
        else:
            return False
    except OSError as err:
        log.error(err)
        return False

def bounce_cluster():
    if False:
        return 10
    "\n    Bounce all Traffic Server nodes in the cluster. Bouncing Traffic Server\n    shuts down and immediately restarts Traffic Server, node-by-node.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.bounce_cluster\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('cluster', 'restart')
    else:
        cmd = _traffic_line('-B')
    return _subprocess(cmd)

def bounce_local(drain=False):
    if False:
        i = 10
        return i + 15
    "\n    Bounce Traffic Server on the local node. Bouncing Traffic Server shuts down\n    and immediately restarts the Traffic Server node.\n\n    drain\n        This option modifies the restart behavior such that traffic_server\n        is not shut down until the number of active client connections\n        drops to the number given by the\n        proxy.config.restart.active_client_threshold configuration\n        variable.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.bounce_local\n        salt '*' trafficserver.bounce_local drain=True\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('server', 'restart')
    else:
        cmd = _traffic_line('-b')
    if drain:
        cmd = cmd + ['--drain']
    return _subprocess(cmd)

def clear_cluster():
    if False:
        print('Hello World!')
    "\n    Clears accumulated statistics on all nodes in the cluster.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.clear_cluster\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('metric', 'clear', '--cluster')
    else:
        cmd = _traffic_line('-C')
    return _subprocess(cmd)

def clear_node():
    if False:
        print('Hello World!')
    "\n    Clears accumulated statistics on the local node.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.clear_node\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('metric', 'clear')
    else:
        cmd = _traffic_line('-c')
    return _subprocess(cmd)

def restart_cluster():
    if False:
        for i in range(10):
            print('nop')
    "\n    Restart the traffic_manager process and the traffic_server process on all\n    the nodes in a cluster.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.restart_cluster\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('cluster', 'restart', '--manager')
    else:
        cmd = _traffic_line('-M')
    return _subprocess(cmd)

def restart_local(drain=False):
    if False:
        return 10
    "\n    Restart the traffic_manager and traffic_server processes on the local node.\n\n    drain\n        This option modifies the restart behavior such that\n        ``traffic_server`` is not shut down until the number of\n        active client connections drops to the number given by the\n        ``proxy.config.restart.active_client_threshold`` configuration\n        variable.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.restart_local\n        salt '*' trafficserver.restart_local drain=True\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('server', 'restart', '--manager')
    else:
        cmd = _traffic_line('-L')
    if drain:
        cmd = cmd + ['--drain']
    return _subprocess(cmd)

def match_metric(regex):
    if False:
        return 10
    "\n    Display the current values of all metrics whose names match the\n    given regular expression.\n\n    .. versionadded:: 2016.11.0\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.match_metric regex\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('metric', 'match', regex)
    else:
        cmd = _traffic_ctl('-m', regex)
    return _subprocess(cmd)

def match_config(regex):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display the current values of all configuration variables whose\n    names match the given regular expression.\n\n    .. versionadded:: 2016.11.0\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.match_config regex\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('config', 'match', regex)
    else:
        cmd = _traffic_line('-m', regex)
    return _subprocess(cmd)

def read_config(*args):
    if False:
        return 10
    "\n    Read Traffic Server configuration variable definitions.\n\n    .. versionadded:: 2016.11.0\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.read_config proxy.config.http.keep_alive_post_out\n    "
    ret = {}
    if _TRAFFICCTL:
        cmd = _traffic_ctl('config', 'get')
    else:
        cmd = _traffic_line('-r')
    try:
        for arg in args:
            log.debug('Querying: %s', arg)
            ret[arg] = _subprocess(cmd + [arg])
    except KeyError:
        pass
    return ret

def read_metric(*args):
    if False:
        return 10
    "\n    Read Traffic Server one or more metrics.\n\n    .. versionadded:: 2016.11.0\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.read_metric proxy.process.http.tcp_hit_count_stat\n    "
    ret = {}
    if _TRAFFICCTL:
        cmd = _traffic_ctl('metric', 'get')
    else:
        cmd = _traffic_line('-r')
    try:
        for arg in args:
            log.debug('Querying: %s', arg)
            ret[arg] = _subprocess(cmd + [arg])
    except KeyError:
        pass
    return ret

def set_config(variable, value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the value of a Traffic Server configuration variable.\n\n    variable\n        Name of a Traffic Server configuration variable.\n\n    value\n        The new value to set.\n\n    .. versionadded:: 2016.11.0\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.set_config proxy.config.http.keep_alive_post_out 0\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('config', 'set', variable, value)
    else:
        cmd = _traffic_line('-s', variable, '-v', value)
    log.debug('Setting %s to %s', variable, value)
    return _subprocess(cmd)

def shutdown():
    if False:
        i = 10
        return i + 15
    "\n    Shut down Traffic Server on the local node.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.shutdown\n    "
    if _TRAFFICLINE:
        cmd = _traffic_line('-S')
    else:
        cmd = _traffic_ctl('server', 'stop')
    _subprocess(cmd)
    return _statuscmd()

def startup():
    if False:
        return 10
    "\n    Start Traffic Server on the local node.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.start\n    "
    if _TRAFFICLINE:
        cmd = _traffic_line('-U')
    else:
        cmd = _traffic_ctl('server', 'start')
    _subprocess(cmd)
    return _statuscmd()

def refresh():
    if False:
        return 10
    "\n    Initiate a Traffic Server configuration file reread. Use this command to\n    update the running configuration after any configuration file modification.\n\n    The timestamp of the last reconfiguration event (in seconds since epoch) is\n    published in the proxy.node.config.reconfigure_time metric.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.refresh\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('config', 'reload')
    else:
        cmd = _traffic_line('-x')
    return _subprocess(cmd)

def zero_cluster():
    if False:
        i = 10
        return i + 15
    "\n    Reset performance statistics to zero across the cluster.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.zero_cluster\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('metric', 'clear', '--cluster')
    else:
        cmd = _traffic_line('-Z')
    return _subprocess(cmd)

def zero_node():
    if False:
        i = 10
        return i + 15
    "\n    Reset performance statistics to zero on the local node.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.zero_cluster\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('metric', 'clear')
    else:
        cmd = _traffic_line('-z')
    return _subprocess(cmd)

def offline(path):
    if False:
        print('Hello World!')
    "\n    Mark a cache storage device as offline. The storage is identified by a path\n    which must match exactly a path specified in storage.config. This removes\n    the storage from the cache and redirects requests that would have used this\n    storage to other storage. This has exactly the same effect as a disk\n    failure for that storage. This does not persist across restarts of the\n    traffic_server process.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.offline /path/to/cache\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('storage', 'offline', path)
    else:
        cmd = _traffic_line('--offline', path)
    return _subprocess(cmd)

def alarms():
    if False:
        print('Hello World!')
    "\n    List all alarm events that have not been acknowledged (cleared).\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.alarms\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('alarm', 'list')
    else:
        cmd = _traffic_line('--alarms')
    return _subprocess(cmd)

def clear_alarms(alarm):
    if False:
        return 10
    "\n    Clear (acknowledge) an alarm event. The arguments are “all” for all current\n    alarms, a specific alarm number (e.g. ‘‘1’‘), or an alarm string identifier\n    (e.g. ‘’MGMT_ALARM_PROXY_CONFIG_ERROR’‘).\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.clear_alarms [all | #event | name]\n    "
    if _TRAFFICCTL:
        cmd = _traffic_ctl('alarm', 'clear', alarm)
    else:
        cmd = _traffic_line('--clear_alarms', alarm)
    return _subprocess(cmd)

def status():
    if False:
        while True:
            i = 10
    "\n    Show the current proxy server status, indicating if we’re running or not.\n\n    .. code-block:: bash\n\n        salt '*' trafficserver.status\n    "
    return _statuscmd()