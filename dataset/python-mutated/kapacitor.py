"""
Kapacitor execution module.

:configuration: This module accepts connection configuration details either as
    parameters or as configuration settings in /etc/salt/minion on the relevant
    minions::

        kapacitor.host: 'localhost'
        kapacitor.port: 9092

    .. versionadded:: 2016.11.0

    Also protocol and SSL settings could be configured::

        kapacitor.unsafe_ssl: 'false'
        kapacitor.protocol: 'http'

    .. versionadded:: 2019.2.0

    This data can also be passed into pillar. Options passed into opts will
    overwrite options passed into pillar.

"""
import logging as logger
import salt.utils.http
import salt.utils.json
import salt.utils.path
from salt.utils.decorators import memoize
log = logger.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    if salt.utils.path.which('kapacitor'):
        return 'kapacitor'
    else:
        return (False, 'Missing dependency: kapacitor')

@memoize
def version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the kapacitor version.\n    '
    version = __salt__['pkg.version']('kapacitor')
    if not version:
        version = str(__salt__['config.option']('kapacitor.version', 'latest'))
    return version

def _get_url():
    if False:
        i = 10
        return i + 15
    '\n    Get the kapacitor URL.\n    '
    protocol = __salt__['config.option']('kapacitor.protocol', 'http')
    host = __salt__['config.option']('kapacitor.host', 'localhost')
    port = __salt__['config.option']('kapacitor.port', 9092)
    return '{}://{}:{}'.format(protocol, host, port)

def get_task(name):
    if False:
        while True:
            i = 10
    "\n    Get a dict of data on a task.\n\n    name\n        Name of the task to get information about.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kapacitor.get_task cpu\n    "
    url = _get_url()
    if version() < '0.13':
        task_url = '{}/task?name={}'.format(url, name)
    else:
        task_url = '{}/kapacitor/v1/tasks/{}?skip-format=true'.format(url, name)
    response = salt.utils.http.query(task_url, status=True)
    if response['status'] == 404:
        return None
    data = salt.utils.json.loads(response['body'])
    if version() < '0.13':
        return {'script': data['TICKscript'], 'type': data['Type'], 'dbrps': data['DBRPs'], 'enabled': data['Enabled']}
    return {'script': data['script'], 'type': data['type'], 'dbrps': data['dbrps'], 'enabled': data['status'] == 'enabled'}

def _run_cmd(cmd):
    if False:
        return 10
    '\n    Run a Kapacitor task and return a dictionary of info.\n    '
    ret = {}
    env_vars = {'KAPACITOR_URL': _get_url(), 'KAPACITOR_UNSAFE_SSL': __salt__['config.option']('kapacitor.unsafe_ssl', 'false')}
    result = __salt__['cmd.run_all'](cmd, env=env_vars)
    if result.get('stdout'):
        ret['stdout'] = result['stdout']
    if result.get('stderr'):
        ret['stderr'] = result['stderr']
    ret['success'] = result['retcode'] == 0
    return ret

def define_task(name, tick_script, task_type='stream', database=None, retention_policy='default', dbrps=None):
    if False:
        return 10
    '\n    Define a task. Serves as both create/update.\n\n    name\n        Name of the task.\n\n    tick_script\n        Path to the TICK script for the task. Can be a salt:// source.\n\n    task_type\n        Task type. Defaults to \'stream\'\n\n    dbrps\n        A list of databases and retention policies in "dbname"."rpname" format\n        to fetch data from. For backward compatibility, the value of\n        \'database\' and \'retention_policy\' will be merged as part of dbrps.\n\n        .. versionadded:: 2019.2.0\n\n    database\n        Which database to fetch data from.\n\n    retention_policy\n        Which retention policy to fetch data from. Defaults to \'default\'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' kapacitor.define_task cpu salt://kapacitor/cpu.tick database=telegraf\n    '
    if not database and (not dbrps):
        log.error('Providing database name or dbrps is mandatory.')
        return False
    if version() < '0.13':
        cmd = 'kapacitor define -name {}'.format(name)
    else:
        cmd = 'kapacitor define {}'.format(name)
    if tick_script.startswith('salt://'):
        tick_script = __salt__['cp.cache_file'](tick_script, __env__)
    cmd += ' -tick {}'.format(tick_script)
    if task_type:
        cmd += ' -type {}'.format(task_type)
    if not dbrps:
        dbrps = []
    if database and retention_policy:
        dbrp = '{}.{}'.format(database, retention_policy)
        dbrps.append(dbrp)
    if dbrps:
        for dbrp in dbrps:
            cmd += ' -dbrp {}'.format(dbrp)
    return _run_cmd(cmd)

def delete_task(name):
    if False:
        return 10
    "\n    Delete a kapacitor task.\n\n    name\n        Name of the task to delete.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kapacitor.delete_task cpu\n    "
    return _run_cmd('kapacitor delete tasks {}'.format(name))

def enable_task(name):
    if False:
        i = 10
        return i + 15
    "\n    Enable a kapacitor task.\n\n    name\n        Name of the task to enable.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kapacitor.enable_task cpu\n    "
    return _run_cmd('kapacitor enable {}'.format(name))

def disable_task(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Disable a kapacitor task.\n\n    name\n        Name of the task to disable.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kapacitor.disable_task cpu\n    "
    return _run_cmd('kapacitor disable {}'.format(name))