"""
Module for rebootmgr
:maintainer:    Alberto Planas <aplanas@suse.com>
:maturity:      new
:depends:       None
:platform:      Linux

.. versionadded:: 3004
"""
import logging
import re
import salt.exceptions
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    'rebootmgrctl command is required.'
    if __utils__['path.which']('rebootmgrctl') is not None:
        return True
    else:
        return (False, 'Module rebootmgt requires the command rebootmgrctl')

def _cmd(cmd, retcode=False):
    if False:
        print('Hello World!')
    'Utility function to run commands.'
    result = __salt__['cmd.run_all'](cmd)
    if retcode:
        return result['retcode']
    if result['retcode']:
        raise salt.exceptions.CommandExecutionError(result['stderr'])
    return result['stdout']

def version():
    if False:
        i = 10
        return i + 15
    'Return the version of rebootmgrd\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr version\n\n    '
    cmd = ['rebootmgrctl', '--version']
    return _cmd(cmd).split()[-1]

def is_active():
    if False:
        print('Hello World!')
    'Check if the rebootmgrd is running and active or not.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr is_active\n\n    '
    cmd = ['rebootmgrctl', 'is_active', '--quiet']
    return _cmd(cmd, retcode=True) == 0

def reboot(order=None):
    if False:
        return 10
    'Tells rebootmgr to schedule a reboot.\n\n    With the [now] option, a forced reboot is done, no lock from etcd\n    is requested and a set maintenance window is ignored. With the\n    [fast] option, a lock from etcd is requested if needed, but a\n    defined maintenance window is ignored.\n\n    order\n        If specified, can be "now" or "fast"\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr reboot\n        salt microos rebootmgt reboot order=now\n\n    '
    if order and order not in ('now', 'fast'):
        raise salt.exceptions.CommandExecutionError("Order parameter, if specified, must be 'now' or 'fast'")
    cmd = ['rebootmgrctl', 'reboot']
    if order:
        cmd.append(order)
    return _cmd(cmd)

def cancel():
    if False:
        return 10
    'Cancels an already running reboot.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr cancel\n\n    '
    cmd = ['rebootmgrctl', 'cancel']
    return _cmd(cmd)

def status():
    if False:
        while True:
            i = 10
    'Returns the current status of rebootmgrd.\n\n    Valid returned values are:\n      0 - No reboot requested\n      1 - Reboot requested\n      2 - Reboot requested, waiting for maintenance window\n      3 - Reboot requested, waiting for etcd lock.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr status\n\n    '
    cmd = ['rebootmgrctl', 'status', '--quiet']
    return _cmd(cmd, retcode=True)

def set_strategy(strategy=None):
    if False:
        for i in range(10):
            print('nop')
    'A new strategy to reboot the machine is set and written into\n    /etc/rebootmgr.conf.\n\n    strategy\n        If specified, must be one of those options:\n\n        best-effort - This is the default strategy. If etcd is\n            running, etcd-lock is used. If no etcd is running, but a\n            maintenance window is specified, the strategy will be\n            maint-window. If no maintenance window is specified, the\n            machine is immediately rebooted (instantly).\n\n        etcd-lock - A lock at etcd for the specified lock-group will\n            be acquired before reboot. If a maintenance window is\n            specified, the lock is only acquired during this window.\n\n        maint-window - Reboot does happen only during a specified\n            maintenance window. If no window is specified, the\n            instantly strategy is followed.\n\n        instantly - Other services will be informed that a reboot will\n            happen. Reboot will be done without getting any locks or\n            waiting for a maintenance window.\n\n        off - Reboot requests are temporary\n            ignored. /etc/rebootmgr.conf is not modified.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr set_strategy stragegy=off\n\n    '
    if strategy and strategy not in ('best-effort', 'etcd-lock', 'maint-window', 'instantly', 'off'):
        raise salt.exceptions.CommandExecutionError('Strategy parameter not valid')
    cmd = ['rebootmgrctl', 'set-strategy']
    if strategy:
        cmd.append(strategy)
    return _cmd(cmd)

def get_strategy():
    if False:
        return 10
    'The currently used reboot strategy of rebootmgrd will be printed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr get_strategy\n\n    '
    cmd = ['rebootmgrctl', 'get-strategy']
    return _cmd(cmd).split(':')[-1].strip()

def set_window(time, duration):
    if False:
        while True:
            i = 10
    'Set\'s the maintenance window.\n\n    time\n        The format of time is the same as described in\n        systemd.time(7).\n\n    duration\n         The format of duration is "[XXh][YYm]".\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr set_window time="Thu,Fri 2020-*-1,5 11:12:13" duration=1h\n\n    '
    cmd = ['rebootmgrctl', 'set-window', time, duration]
    return _cmd(cmd)

def get_window():
    if False:
        return 10
    'The currently set maintenance window will be printed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr get_window\n\n    '
    cmd = ['rebootmgrctl', 'get-window']
    window = _cmd(cmd)
    return dict(zip(('time', 'duration'), re.search('Maintenance window is set to (.*), lasting (.*).', window).groups()))

def set_group(group):
    if False:
        return 10
    'Set the group, to which this machine belongs to get a reboot lock\n       from etcd.\n\n    group\n        Group name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr set_group group=group_1\n\n    '
    cmd = ['rebootmgrctl', 'set-group', group]
    return _cmd(cmd)

def get_group():
    if False:
        while True:
            i = 10
    'The currently set lock group for etcd.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr get_group\n\n    '
    cmd = ['rebootmgrctl', 'get-group']
    group = _cmd(cmd)
    return re.search('Etcd lock group is set to (.*)', group).groups()[0]

def set_max(max_locks, group=None):
    if False:
        return 10
    'Set the maximal number of hosts in a group, which are allowed to\n       reboot at the same time.\n\n    number\n        Maximal number of hosts in a group\n\n    group\n        Group name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr set_max 4\n\n    '
    cmd = ['rebootmgrctl', 'set-max']
    if group:
        cmd.extend(['--group', group])
    cmd.append(max_locks)
    return _cmd(cmd)

def lock(machine_id=None, group=None):
    if False:
        print('Hello World!')
    'Lock a machine. If no group is specified, the local default group\n       will be used. If no machine-id is specified, the local machine\n       will be locked.\n\n    machine_id\n        The machine-id is a network wide, unique ID. Per default the\n        ID from /etc/machine-id is used.\n\n    group\n        Group name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr lock group=group1\n\n    '
    cmd = ['rebootmgrctl', 'lock']
    if group:
        cmd.extend(['--group', group])
    if machine_id:
        cmd.append(machine_id)
    return _cmd(cmd)

def unlock(machine_id=None, group=None):
    if False:
        while True:
            i = 10
    'Unlock a machine. If no group is specified, the local default group\n       will be used. If no machine-id is specified, the local machine\n       will be locked.\n\n    machine_id\n        The machine-id is a network wide, unique ID. Per default the\n        ID from /etc/machine-id is used.\n\n    group\n        Group name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt microos rebootmgr unlock group=group1\n\n    '
    cmd = ['rebootmgrctl', 'unlock']
    if group:
        cmd.extend(['--group', group])
    if machine_id:
        cmd.append(machine_id)
    return _cmd(cmd)