"""
Monit service module. This module will create a monit type
service watcher.
"""
import re
import salt.utils.path
__func_alias__ = {'id_': 'id', 'reload_': 'reload'}

def __virtual__():
    if False:
        while True:
            i = 10
    if salt.utils.path.which('monit') is not None:
        return True
    return (False, 'The monit execution module cannot be loaded: the monit binary is not in the path.')

def start(name):
    if False:
        return 10
    "\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.start <service name>\n    "
    cmd = 'monit start {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def stop(name):
    if False:
        while True:
            i = 10
    "\n    Stops service via monit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.stop <service name>\n    "
    cmd = 'monit stop {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def restart(name):
    if False:
        return 10
    "\n    Restart service via monit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.restart <service name>\n    "
    cmd = 'monit restart {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def unmonitor(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Unmonitor service via monit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.unmonitor <service name>\n    "
    cmd = 'monit unmonitor {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def monitor(name):
    if False:
        while True:
            i = 10
    "\n    monitor service via monit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.monitor <service name>\n    "
    cmd = 'monit monitor {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def summary(svc_name=''):
    if False:
        print('Hello World!')
    "\n    Display a summary from monit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.summary\n        salt '*' monit.summary <service name>\n    "
    ret = {}
    cmd = 'monit summary'
    res = __salt__['cmd.run'](cmd).splitlines()
    for line in res:
        if 'daemon is not running' in line:
            return dict(monit='daemon is not running', result=False)
        elif not line or svc_name not in line or 'The Monit daemon' in line:
            continue
        else:
            parts = line.split("'")
            if len(parts) == 3:
                (resource, name, status_) = (parts[0].strip(), parts[1], parts[2].strip())
                if svc_name != '' and svc_name != name:
                    continue
                if resource not in ret:
                    ret[resource] = {}
                ret[resource][name] = status_
    return ret

def status(svc_name=''):
    if False:
        while True:
            i = 10
    "\n    Display a process status from monit\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.status\n        salt '*' monit.status <service name>\n    "
    cmd = 'monit status'
    res = __salt__['cmd.run'](cmd)
    if version() < '5.18.0':
        fieldlength = 33
    else:
        fieldlength = 28
    separator = 3 + fieldlength
    prostr = 'Process' + ' ' * fieldlength
    s = res.replace('Process', prostr).replace("'", '').split('\n\n')
    entries = {}
    for process in s[1:-1]:
        pro = process.splitlines()
        tmp = {}
        for items in pro:
            key = items[:separator].strip()
            tmp[key] = items[separator - 1:].strip()
        entries[pro[0].split()[1]] = tmp
    if svc_name == '':
        ret = entries
    else:
        ret = entries.get(svc_name, 'No such service')
    return ret

def reload_():
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.3.0\n\n    Reload monit configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.reload\n    "
    cmd = 'monit reload'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def configtest():
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2016.3.0\n\n    Test monit configuration syntax\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.configtest\n    "
    ret = {}
    cmd = 'monit -t'
    out = __salt__['cmd.run_all'](cmd)
    if out['retcode'] != 0:
        ret['comment'] = 'Syntax Error'
        ret['stderr'] = out['stderr']
        ret['result'] = False
        return ret
    ret['comment'] = 'Syntax OK'
    ret['stdout'] = out['stdout']
    ret['result'] = True
    return ret

def version():
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.3.0\n\n    Return version from monit -V\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.version\n    "
    cmd = 'monit -V'
    out = __salt__['cmd.run'](cmd).splitlines()
    ret = out[0].split()
    return ret[-1]

def id_(reset=False):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2016.3.0\n\n    Return monit unique id.\n\n    reset : False\n        Reset current id and generate a new id when it's True.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.id [reset=True]\n    "
    if reset:
        id_pattern = re.compile('Monit id (?P<id>[^ ]+)')
        cmd = 'echo y|monit -r'
        out = __salt__['cmd.run_all'](cmd, python_shell=True)
        ret = id_pattern.search(out['stdout']).group('id')
        return ret if ret else False
    else:
        cmd = 'monit -i'
        out = __salt__['cmd.run'](cmd)
        ret = out.split(':')[-1].strip()
    return ret

def validate():
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.0\n\n    Check all services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' monit.validate\n    "
    cmd = 'monit validate'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)