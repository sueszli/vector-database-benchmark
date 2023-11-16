"""
Set grains describing the minion process.
"""
import os
import salt.utils.platform
import salt.utils.user

def _uid():
    if False:
        return 10
    '\n    Grain for the minion User ID\n    '
    return salt.utils.user.get_uid()

def _username():
    if False:
        for i in range(10):
            print('nop')
    '\n    Grain for the minion username\n    '
    return salt.utils.user.get_user()

def _gid():
    if False:
        return 10
    '\n    Grain for the minion Group ID\n    '
    return salt.utils.user.get_gid()

def _groupname():
    if False:
        print('Hello World!')
    '\n    Grain for the minion groupname\n    '
    try:
        return salt.utils.user.get_default_group(_username()) or ''
    except KeyError:
        return ''

def _pid():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the current process pid\n    '
    return os.getpid()

def grains():
    if False:
        while True:
            i = 10
    '\n    Return the grains dictionary\n    '
    ret = {'username': _username(), 'groupname': _groupname(), 'pid': _pid()}
    if not salt.utils.platform.is_windows():
        ret['gid'] = _gid()
        ret['uid'] = _uid()
    return ret