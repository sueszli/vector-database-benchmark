"""
Interface to Red Hat tuned-adm module

:maintainer:    Syed Ali <alicsyed@gmail.com>
:maturity:      new
:depends:       tuned-adm
:platform:      Linux
"""
import re
import salt.utils.path
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'tuned'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Check to see if tuned-adm binary is installed on the system\n\n    '
    tuned_adm = salt.utils.path.which('tuned-adm')
    if not tuned_adm:
        return (False, 'The tuned execution module failed to load: the tuned-adm binary is not in the path.')
    return __virtualname__

def list_():
    if False:
        for i in range(10):
            print('nop')
    "\n    List the profiles available\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' tuned.list\n    "
    result = __salt__['cmd.run']('tuned-adm list').splitlines()
    result.pop(0)
    try:
        result = result[:result.index('** COLLECTED WARNINGS **') - 1]
    except ValueError:
        pass
    result.pop()
    result = [i.split('- ')[1].strip() for i in result]
    return result

def active():
    if False:
        while True:
            i = 10
    "\n    Return current active profile\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' tuned.active\n    "
    result = __salt__['cmd.run_all']('tuned-adm active', ignore_retcode=True)
    if result['retcode'] != 0:
        return 'none'
    pattern = re.compile('(?P<stmt>Current active profile:) (?P<profile>\\w+.*)')
    match = re.match(pattern, result['stdout'])
    return '{}'.format(match.group('profile'))

def off():
    if False:
        for i in range(10):
            print('nop')
    "\n    Turn off all profiles\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' tuned.off\n    "
    result = __salt__['cmd.retcode']('tuned-adm off')
    if int(result) != 0:
        return False
    return True

def profile(profile_name):
    if False:
        print('Hello World!')
    "\n    Activate specified profile\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' tuned.profile virtual-guest\n    "
    result = __salt__['cmd.retcode']('tuned-adm profile {}'.format(profile_name))
    if int(result) != 0:
        return False
    return '{}'.format(profile_name)