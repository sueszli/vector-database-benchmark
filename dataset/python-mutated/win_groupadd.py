"""
Manage groups on Windows

.. important::
    If you feel that Salt should be using this module to manage groups on a
    minion, and it is using a different module (or gives an error similar to
    *'group.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import logging
import salt.utils.platform
import salt.utils.win_functions
import salt.utils.winapi
try:
    import pywintypes
    import win32api
    import win32com.client
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
log = logging.getLogger(__name__)
__virtualname__ = 'group'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set the group module if the kernel is Windows\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'win_groupadd: only works on Windows systems')
    if not HAS_DEPENDENCIES:
        return (False, 'win_groupadd: missing dependencies')
    return __virtualname__

def _get_computer_object():
    if False:
        return 10
    '\n    A helper function to get the object for the local machine\n\n    Returns:\n        object: Returns the computer object for the local machine\n    '
    with salt.utils.winapi.Com():
        nt = win32com.client.Dispatch('AdsNameSpaces')
        return nt.GetObject('', 'WinNT://.,computer')

def _get_group_object(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    A helper function to get a specified group object\n\n    Args:\n\n        name (str): The name of the object\n\n    Returns:\n        object: The specified group object\n    '
    with salt.utils.winapi.Com():
        nt = win32com.client.Dispatch('AdsNameSpaces')
        return nt.GetObject('', 'WinNT://./' + name + ',group')

def _get_all_groups():
    if False:
        for i in range(10):
            print('nop')
    '\n    A helper function that gets a list of group objects for all groups on the\n    machine\n\n    Returns:\n        iter: A list of objects for all groups on the machine\n    '
    with salt.utils.winapi.Com():
        nt = win32com.client.Dispatch('AdsNameSpaces')
        results = nt.GetObject('', 'WinNT://.')
        results.Filter = ['group']
        return results

def _get_username(member):
    if False:
        while True:
            i = 10
    '\n    Resolve the username from the member object returned from a group query\n\n    Returns:\n        str: The username converted to domain\\username format\n    '
    return member.ADSPath.replace('WinNT://', '').replace('/', '\\')

def add(name, **kwargs):
    if False:
        return 10
    "\n    Add the specified group\n\n    Args:\n\n        name (str):\n            The name of the group to add\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.add foo\n    "
    if not info(name):
        comp_obj = _get_computer_object()
        try:
            new_group = comp_obj.Create('group', name)
            new_group.SetInfo()
            log.info('Successfully created group %s', name)
        except pywintypes.com_error as exc:
            log.error('Failed to create group %s. %s', name, win32api.FormatMessage(exc.excepinfo[5]))
            return False
    else:
        log.warning('The group %s already exists.', name)
        return False
    return True

def delete(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove the named group\n\n    Args:\n\n        name (str):\n            The name of the group to remove\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.delete foo\n    "
    if info(name):
        comp_obj = _get_computer_object()
        try:
            comp_obj.Delete('group', name)
            log.info('Successfully removed group %s', name)
        except pywintypes.com_error as exc:
            log.error('Failed to remove group %s. %s', name, win32api.FormatMessage(exc.excepinfo[5]))
            return False
    else:
        log.warning('The group %s does not exist.', name)
        return False
    return True

def info(name):
    if False:
        i = 10
        return i + 15
    "\n    Return information about a group\n\n    Args:\n\n        name (str):\n            The name of the group for which to get information\n\n    Returns:\n        dict: A dictionary of information about the group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.info foo\n    "
    try:
        groupObj = _get_group_object(name)
        gr_name = groupObj.Name
        gr_mem = [_get_username(x) for x in groupObj.members()]
    except pywintypes.com_error as exc:
        log.debug('Failed to access group %s. %s', name, win32api.FormatMessage(exc.excepinfo[5]))
        return False
    if not gr_name:
        return False
    return {'name': gr_name, 'passwd': None, 'gid': None, 'members': gr_mem}

def getent(refresh=False):
    if False:
        while True:
            i = 10
    "\n    Return info on all groups\n\n    Args:\n\n        refresh (bool):\n            Refresh the info for all groups in ``__context__``. If False only\n            the groups in ``__context__`` will be returned. If True the\n            ``__context__`` will be refreshed with current data and returned.\n            Default is False\n\n    Returns:\n        A list of groups and their information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.getent\n    "
    if 'group.getent' in __context__ and (not refresh):
        return __context__['group.getent']
    ret = []
    results = _get_all_groups()
    for result in results:
        group = {'gid': __salt__['file.group_to_gid'](result.Name), 'members': [_get_username(x) for x in result.members()], 'name': result.Name, 'passwd': 'x'}
        ret.append(group)
    __context__['group.getent'] = ret
    return ret

def adduser(name, username, **kwargs):
    if False:
        print('Hello World!')
    "\n    Add a user to a group\n\n    Args:\n\n        name (str):\n            The name of the group to modify\n\n        username (str):\n            The name of the user to add to the group\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.adduser foo username\n    "
    try:
        group_obj = _get_group_object(name)
    except pywintypes.com_error as exc:
        log.error('Failed to access group %s. %s', name, win32api.FormatMessage(exc.excepinfo[5]))
        return False
    existing_members = [_get_username(x) for x in group_obj.members()]
    username = salt.utils.win_functions.get_sam_name(username)
    try:
        if username not in existing_members:
            group_obj.Add('WinNT://' + username.replace('\\', '/'))
            log.info('Added user %s', username)
        else:
            log.warning('User %s is already a member of %s', username, name)
            return False
    except pywintypes.com_error as exc:
        log.error('Failed to add %s to group %s. %s', username, name, exc.excepinfo[2])
        return False
    return True

def deluser(name, username, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove a user from a group\n\n    Args:\n\n        name (str):\n            The name of the group to modify\n\n        username (str):\n            The name of the user to remove from the group\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.deluser foo username\n    "
    try:
        group_obj = _get_group_object(name)
    except pywintypes.com_error as exc:
        log.error('Failed to access group %s. %s', name, win32api.FormatMessage(exc.excepinfo[5]))
        return False
    existing_members = [_get_username(x) for x in group_obj.members()]
    try:
        if salt.utils.win_functions.get_sam_name(username) in existing_members:
            group_obj.Remove('WinNT://' + username.replace('\\', '/'))
            log.info('Removed user %s', username)
        else:
            log.warning('User %s is not a member of %s', username, name)
            return False
    except pywintypes.com_error as exc:
        log.error('Failed to remove %s from group %s. %s', username, name, win32api.FormatMessage(exc.excepinfo[5]))
        return False
    return True

def members(name, members_list, **kwargs):
    if False:
        return 10
    "\n    Ensure a group contains only the members in the list\n\n    Args:\n\n        name (str):\n            The name of the group to modify\n\n        members_list (str):\n            A single user or a comma separated list of users. The group will\n            contain only the users specified in this list.\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.members foo 'user1,user2,user3'\n    "
    members_list = [salt.utils.win_functions.get_sam_name(m) for m in members_list.split(',')]
    if not isinstance(members_list, list):
        log.debug('member_list is not a list')
        return False
    try:
        obj_group = _get_group_object(name)
    except pywintypes.com_error as exc:
        log.error('Failed to access group %s. %s', name, win32api.FormatMessage(exc.excepinfo[5]))
        return False
    existing_members = [_get_username(x) for x in obj_group.members()]
    existing_members.sort()
    members_list.sort()
    if existing_members == members_list:
        log.info('%s membership is correct', name)
        return True
    success = True
    for member in members_list:
        if member not in existing_members:
            try:
                obj_group.Add('WinNT://' + member.replace('\\', '/'))
                log.info('User added: %s', member)
            except pywintypes.com_error as exc:
                log.error('Failed to add %s to %s. %s', member, name, win32api.FormatMessage(exc.excepinfo[5]))
                success = False
    for member in existing_members:
        if member not in members_list:
            try:
                obj_group.Remove('WinNT://' + member.replace('\\', '/'))
                log.info('User removed: %s', member)
            except pywintypes.com_error as exc:
                log.error('Failed to remove %s from %s. %s', member, name, win32api.FormatMessage(exc.excepinfo[5]))
                success = False
    return success

def list_groups(refresh=False):
    if False:
        print('Hello World!')
    "\n    Return a list of groups\n\n    Args:\n\n        refresh (bool):\n            Refresh the info for all groups in ``__context__``. If False only\n            the groups in ``__context__`` will be returned. If True, the\n            ``__context__`` will be refreshed with current data and returned.\n            Default is False\n\n    Returns:\n        list: A list of groups on the machine\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.list_groups\n    "
    if 'group.list_groups' in __context__ and (not refresh):
        return __context__['group.list_groups']
    results = _get_all_groups()
    ret = []
    for result in results:
        ret.append(result.Name)
    __context__['group.list_groups'] = ret
    return ret