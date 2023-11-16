"""
Manage account locks on AIX systems

.. versionadded:: 2018.3.0

:depends: none
"""
import logging
log = logging.getLogger(__name__)
__virtualname__ = 'shadow'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if kernel is AIX\n    '
    if __grains__['kernel'] == 'AIX':
        return __virtualname__
    return (False, 'The aix_shadow execution module failed to load: only available on AIX systems.')

def login_failures(user):
    if False:
        while True:
            i = 10
    '\n    Query for all accounts which have 3 or more login failures.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <minion_id> shadow.login_failures ALL\n    '
    cmd = 'lsuser -a unsuccessful_login_count {}'.format(user)
    cmd += " | grep -E 'unsuccessful_login_count=([3-9]|[0-9][0-9]+)'"
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=True)
    ret = []
    lines = out['stdout'].splitlines()
    for line in lines:
        ret.append(line.split()[0])
    return ret

def locked(user):
    if False:
        while True:
            i = 10
    '\n    Query for all accounts which are flagged as locked.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <minion_id> shadow.locked ALL\n    '
    cmd = 'lsuser -a account_locked {}'.format(user)
    cmd += ' | grep "account_locked=true"'
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=True)
    ret = []
    lines = out['stdout'].splitlines()
    for line in lines:
        ret.append(line.split()[0])
    return ret

def unlock(user):
    if False:
        print('Hello World!')
    '\n    Unlock user for locked account\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt <minion_id> shadow.unlock user\n    '
    cmd = 'chuser account_locked=false {0} | chsec -f /etc/security/lastlog -a "unsuccessful_login_count=0" -s {0}'.format(user)
    ret = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=True)
    return ret