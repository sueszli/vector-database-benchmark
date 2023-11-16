"""
Support for the softwareupdate command on MacOS.
"""
import os
import re
import salt.utils.data
import salt.utils.files
import salt.utils.mac_utils
import salt.utils.path
import salt.utils.platform
from salt.exceptions import CommandExecutionError, SaltInvocationError
__virtualname__ = 'softwareupdate'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only for MacOS\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'The softwareupdate module could not be loaded: module only works on MacOS systems.')
    return __virtualname__

def _get_available(recommended=False, restart=False, shut_down=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Utility function to get all available update packages.\n\n    Sample return date:\n    { 'updatename': '1.2.3-45', ... }\n    "
    cmd = ['softwareupdate', '--list']
    out = salt.utils.mac_utils.execute_return_result(cmd)
    if __grains__['osrelease_info'][0] > 10 or __grains__['osrelease_info'][1] >= 15:
        rexp = re.compile('(?m)^\\s*[*-] Label: (?P<name>[^ ].*)[\\r\\n].*Version: (?P<version>[^,]*), Size: (?P<size>[^,]*),\\s*(?P<recommended>Recommended: YES,)?\\s*(?P<action>Action: (?:restart|shut down),)?')
    else:
        rexp = re.compile('(?m)^\\s+[*-] (?P<name>.*)[\\r\\n].*\\((?P<version>[^ \\)]*)[^\\r\\n\\[]*(?P<recommended>\\[recommended\\])?\\s?(?P<action>\\[(?:restart|shut down)\\])?')
    conditions = []
    if salt.utils.data.is_true(recommended):
        conditions.append(lambda m: m.group('recommended'))
    if salt.utils.data.is_true(restart):
        conditions.append(lambda m: 'restart' in (m.group('action') or ''))
    if salt.utils.data.is_true(shut_down):
        conditions.append(lambda m: 'shut down' in (m.group('action') or ''))
    return {m.group('name'): m.group('version') for m in rexp.finditer(out) if all((f(m) for f in conditions))}

def list_available(recommended=False, restart=False, shut_down=False):
    if False:
        print('Hello World!')
    "\n    List all available updates.\n\n    :param bool recommended: Show only recommended updates.\n\n    :param bool restart: Show only updates that require a restart.\n\n    :return: Returns a dictionary containing the updates\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.list_available\n    "
    return _get_available(recommended, restart, shut_down)

def ignore(name):
    if False:
        i = 10
        return i + 15
    '\n    Ignore a specific program update. When an update is ignored the \'-\' and\n    version number at the end will be omitted, so "SecUpd2014-001-1.0" becomes\n    "SecUpd2014-001". It will be removed automatically if present. An update\n    is successfully ignored when it no longer shows up after list_updates.\n\n    :param name: The name of the update to add to the ignore list.\n    :ptype: str\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' softwareupdate.ignore <update-name>\n    '
    to_ignore = name.rsplit('-', 1)[0]
    cmd = ['softwareupdate', '--ignore', to_ignore]
    salt.utils.mac_utils.execute_return_success(cmd)
    return to_ignore in list_ignored()

def list_ignored():
    if False:
        while True:
            i = 10
    "\n    List all updates that have been ignored. Ignored updates are shown\n    without the '-' and version number at the end, this is how the\n    softwareupdate command works.\n\n    :return: The list of ignored updates\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.list_ignored\n    "
    cmd = ['softwareupdate', '--list', '--ignore']
    out = salt.utils.mac_utils.execute_return_result(cmd)
    rexp = re.compile('(?m)^    ["]?([^,|\\s].*[^"|\\n|,])[,|"]?')
    return rexp.findall(out)

def reset_ignored():
    if False:
        return 10
    "\n    Make sure the ignored updates are not ignored anymore,\n    returns a list of the updates that are no longer ignored.\n\n    :return: True if the list was reset, Otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.reset_ignored\n    "
    cmd = ['softwareupdate', '--reset-ignored']
    salt.utils.mac_utils.execute_return_success(cmd)
    return list_ignored() == []

def schedule_enabled():
    if False:
        for i in range(10):
            print('nop')
    "\n    Check the status of automatic update scheduling.\n\n    :return: True if scheduling is enabled, False if disabled\n\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.schedule_enabled\n    "
    cmd = ['softwareupdate', '--schedule']
    ret = salt.utils.mac_utils.execute_return_result(cmd)
    enabled = ret.split()[-1]
    return salt.utils.mac_utils.validate_enabled(enabled) == 'on'

def schedule_enable(enable):
    if False:
        return 10
    "\n    Enable/disable automatic update scheduling.\n\n    :param enable: True/On/Yes/1 to turn on automatic updates. False/No/Off/0\n        to turn off automatic updates. If this value is empty, the current\n        status will be returned.\n\n    :type: bool str\n\n    :return: True if scheduling is enabled, False if disabled\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.schedule_enable on|off\n    "
    status = salt.utils.mac_utils.validate_enabled(enable)
    cmd = ['softwareupdate', '--schedule', salt.utils.mac_utils.validate_enabled(status)]
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.validate_enabled(schedule_enabled()) == status

def update_all(recommended=False, restart=True):
    if False:
        return 10
    "\n    Install all available updates. Returns a dictionary containing the name\n    of the update and the status of its installation.\n\n    :param bool recommended: If set to True, only install the recommended\n        updates. If set to False (default) all updates are installed.\n\n    :param bool restart: Set this to False if you do not want to install updates\n        that require a restart. Default is True\n\n    :return: A dictionary containing the updates that were installed and the\n        status of its installation. If no updates were installed an empty\n        dictionary is returned.\n\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.update_all\n    "
    to_update = _get_available(recommended, restart)
    if not to_update:
        return {}
    for _update in to_update:
        cmd = ['softwareupdate', '--install', _update]
        salt.utils.mac_utils.execute_return_success(cmd)
    ret = {}
    updates_left = _get_available()
    for _update in to_update:
        ret[_update] = True if _update not in updates_left else False
    return ret

def update(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Install a named update.\n\n    :param str name: The name of the of the update to install.\n\n    :return: True if successfully updated, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.update <update-name>\n    "
    if not update_available(name):
        raise SaltInvocationError('Update not available: {}'.format(name))
    cmd = ['softwareupdate', '--install', name]
    salt.utils.mac_utils.execute_return_success(cmd)
    return not update_available(name)

def update_available(name):
    if False:
        print('Hello World!')
    '\n    Check whether or not an update is available with a given name.\n\n    :param str name: The name of the update to look for\n\n    :return: True if available, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt \'*\' softwareupdate.update_available <update-name>\n       salt \'*\' softwareupdate.update_available "<update with whitespace>"\n    '
    return name in _get_available()

def list_downloads():
    if False:
        i = 10
        return i + 15
    "\n    Return a list of all updates that have been downloaded locally.\n\n    :return: A list of updates that have been downloaded\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.list_downloads\n    "
    outfiles = []
    for (root, subFolder, files) in salt.utils.path.os_walk('/Library/Updates'):
        for f in files:
            outfiles.append(os.path.join(root, f))
    dist_files = []
    for f in outfiles:
        if f.endswith('.dist'):
            dist_files.append(f)
    ret = []
    for update in _get_available():
        for f in dist_files:
            with salt.utils.files.fopen(f) as fhr:
                if update.rsplit('-', 1)[0] in salt.utils.stringutils.to_unicode(fhr.read()):
                    ret.append(update)
    return ret

def download(name):
    if False:
        i = 10
        return i + 15
    "\n    Download a named update so that it can be installed later with the\n    ``update`` or ``update_all`` functions\n\n    :param str name: The update to download.\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.download <update name>\n    "
    if not update_available(name):
        raise SaltInvocationError('Update not available: {}'.format(name))
    if name in list_downloads():
        return True
    cmd = ['softwareupdate', '--download', name]
    salt.utils.mac_utils.execute_return_success(cmd)
    return name in list_downloads()

def download_all(recommended=False, restart=True):
    if False:
        i = 10
        return i + 15
    "\n    Download all available updates so that they can be installed later with the\n    ``update`` or ``update_all`` functions. It returns a list of updates that\n    are now downloaded.\n\n    :param bool recommended: If set to True, only install the recommended\n        updates. If set to False (default) all updates are installed.\n\n    :param bool restart: Set this to False if you do not want to install updates\n        that require a restart. Default is True\n\n    :return: A list containing all downloaded updates on the system.\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' softwareupdate.download_all\n    "
    to_download = _get_available(recommended, restart)
    for name in to_download:
        download(name)
    return list_downloads()

def get_catalog():
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2016.3.0\n\n    Get the current catalog being used for update lookups. Will return a url if\n    a custom catalog has been specified. Otherwise the word 'Default' will be\n    returned\n\n    :return: The catalog being used for update lookups\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' softwareupdates.get_catalog\n    "
    cmd = ['defaults', 'read', '/Library/Preferences/com.apple.SoftwareUpdate.plist']
    out = salt.utils.mac_utils.execute_return_result(cmd)
    if 'AppleCatalogURL' in out:
        cmd.append('AppleCatalogURL')
        out = salt.utils.mac_utils.execute_return_result(cmd)
        return out
    elif 'CatalogURL' in out:
        cmd.append('CatalogURL')
        out = salt.utils.mac_utils.execute_return_result(cmd)
        return out
    else:
        return 'Default'

def set_catalog(url):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2016.3.0\n\n    Set the Software Update Catalog to the URL specified\n\n    :param str url: The url to the update catalog\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' softwareupdates.set_catalog http://swupd.local:8888/index.sucatalog\n    "
    cmd = ['softwareupdate', '--set-catalog', url]
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        pass
    return get_catalog() == url

def reset_catalog():
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.3.0\n\n    Reset the Software Update Catalog to the default.\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' softwareupdates.reset_catalog\n    "
    cmd = ['softwareupdate', '--clear-catalog']
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        pass
    return get_catalog() == 'Default'