"""
Manage the Windows registry

-----
Hives
-----
Hives are the main sections of the registry and all begin with the word HKEY.

    - HKEY_LOCAL_MACHINE
    - HKEY_CURRENT_USER
    - HKEY_USER

----
Keys
----
Keys are the folders in the registry. Keys can have many nested subkeys. Keys
can have a value assigned to them under the (Default) value name

-----------------
Values or Entries
-----------------
Values/Entries are name/data pairs. There can be many values in a key. The
(Default) value corresponds to the Key itself, the rest are their own name/value
pairs.

:depends:   - PyWin32
"""
import logging
import salt.utils.platform
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
try:
    import win32api
    import win32con
    import win32gui
    HAS_WINDOWS_MODULES = True
except ImportError:
    HAS_WINDOWS_MODULES = False
log = logging.getLogger(__name__)
__virtualname__ = 'reg'

def __virtual__():
    if False:
        return 10
    '\n    Only works on Windows systems with the PyWin32\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'reg execution module failed to load: The module will only run on Windows systems')
    if not HAS_WINDOWS_MODULES:
        return (False, 'reg execution module failed to load: One of the following libraries did not load: win32gui, win32con, win32api')
    return __virtualname__

def _to_mbcs(vdata):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts unicode to to current users character encoding. Use this for values\n    returned by reg functions\n    '
    return salt.utils.stringutils.to_unicode(vdata, 'mbcs')

def _to_unicode(vdata):
    if False:
        while True:
            i = 10
    '\n    Converts from current users character encoding to unicode. Use this for\n    parameters being pass to reg functions\n    '
    if vdata is None:
        vdata = ''
    if isinstance(vdata, int):
        vdata = str(vdata)
    return salt.utils.stringutils.to_unicode(vdata, 'utf-8')

class Registry:
    """
    This was put in a class to delay usage until this module is actually used
    This class contains all the lookup dicts for working with the registry
    """

    def __init__(self):
        if False:
            return 10
        self.hkeys = {'HKEY_CURRENT_CONFIG': win32con.HKEY_CURRENT_CONFIG, 'HKEY_CLASSES_ROOT': win32con.HKEY_CLASSES_ROOT, 'HKEY_CURRENT_USER': win32con.HKEY_CURRENT_USER, 'HKEY_LOCAL_MACHINE': win32con.HKEY_LOCAL_MACHINE, 'HKEY_USERS': win32con.HKEY_USERS, 'HKCC': win32con.HKEY_CURRENT_CONFIG, 'HKCR': win32con.HKEY_CLASSES_ROOT, 'HKCU': win32con.HKEY_CURRENT_USER, 'HKLM': win32con.HKEY_LOCAL_MACHINE, 'HKU': win32con.HKEY_USERS}
        self.vtype = {'REG_NONE': 0, 'REG_BINARY': win32con.REG_BINARY, 'REG_DWORD': win32con.REG_DWORD, 'REG_EXPAND_SZ': win32con.REG_EXPAND_SZ, 'REG_MULTI_SZ': win32con.REG_MULTI_SZ, 'REG_SZ': win32con.REG_SZ, 'REG_QWORD': win32con.REG_QWORD}
        self.opttype = {'REG_OPTION_NON_VOLATILE': 0, 'REG_OPTION_VOLATILE': 1}
        self.vtype_reverse = {0: 'REG_NONE', win32con.REG_BINARY: 'REG_BINARY', win32con.REG_DWORD: 'REG_DWORD', win32con.REG_EXPAND_SZ: 'REG_EXPAND_SZ', win32con.REG_MULTI_SZ: 'REG_MULTI_SZ', win32con.REG_SZ: 'REG_SZ', win32con.REG_QWORD: 'REG_QWORD'}
        self.opttype_reverse = {0: 'REG_OPTION_NON_VOLATILE', 1: 'REG_OPTION_VOLATILE'}
        self.subkey_slash_check = {win32con.HKEY_CURRENT_USER: 0, win32con.HKEY_LOCAL_MACHINE: 1, win32con.HKEY_USERS: 1, win32con.HKEY_CURRENT_CONFIG: 1, win32con.HKEY_CLASSES_ROOT: 1}
        self.registry_32 = {True: win32con.KEY_READ | win32con.KEY_WOW64_32KEY, False: win32con.KEY_READ}

    def __getattr__(self, k):
        if False:
            print('Hello World!')
        try:
            return self.hkeys[k]
        except KeyError:
            msg = "No hkey named '{0}. Try one of {1}'"
            hkeys = ', '.join(self.hkeys)
            raise CommandExecutionError(msg.format(k, hkeys))

def key_exists(hive, key, use_32bit_registry=False):
    if False:
        while True:
            i = 10
    "\n    Check that the key is found in the registry. This refers to keys and not\n    value/data pairs. To check value/data pairs, use ``value_exists``\n\n    Args:\n\n        hive (str): The hive to connect to\n\n        key (str): The key to check\n\n        use_32bit_registry (bool): Look in the 32bit portion of the registry\n\n    Returns:\n        bool: True if exists, otherwise False\n\n    Usage:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg as reg\n            reg.key_exists(hive='HKLM', key='SOFTWARE\\Microsoft')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    access_mask = registry.registry_32[use_32bit_registry]
    handle = None
    try:
        handle = win32api.RegOpenKeyEx(hkey, local_key, 0, access_mask)
        return True
    except win32api.error as exc:
        if exc.winerror == 2:
            return False
        if exc.winerror == 5:
            return True
        raise
    finally:
        if handle:
            win32api.RegCloseKey(handle)

def value_exists(hive, key, vname, use_32bit_registry=False):
    if False:
        return 10
    "\n    Check that the value/data pair is found in the registry.\n\n    .. versionadded:: 2018.3.4\n\n    Args:\n\n        hive (str): The hive to connect to\n\n        key (str): The key to check in\n\n        vname (str): The name of the value/data pair you're checking\n\n        use_32bit_registry (bool): Look in the 32bit portion of the registry\n\n    Returns:\n        bool: True if exists, otherwise False\n\n    Usage:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg as reg\n            reg.value_exists(hive='HKLM',\n                                key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion',\n                                vname='CommonFilesDir')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    local_vname = _to_unicode(vname)
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    access_mask = registry.registry_32[use_32bit_registry]
    try:
        handle = win32api.RegOpenKeyEx(hkey, local_key, 0, access_mask)
    except win32api.error as exc:
        if exc.winerror == 2:
            return False
        raise
    try:
        (_, _) = win32api.RegQueryValueEx(handle, local_vname)
        return True
    except win32api.error as exc:
        if exc.winerror == 2 and vname is None:
            return True
        else:
            return False
    finally:
        if handle:
            win32api.RegCloseKey(handle)

def broadcast_change():
    if False:
        i = 10
        return i + 15
    '\n    Refresh the windows environment.\n\n    .. note::\n        This will only effect new processes and windows. Services will not see\n        the change until the system restarts.\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    Usage:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.broadcast_change()\n    '
    (_, res) = win32gui.SendMessageTimeout(win32con.HWND_BROADCAST, win32con.WM_SETTINGCHANGE, 0, 0, win32con.SMTO_ABORTIFHUNG, 5000)
    return not bool(res)

def list_keys(hive, key=None, use_32bit_registry=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Enumerates the subkeys in a registry key or hive.\n\n    Args:\n\n       hive (str):\n            The name of the hive. Can be one of the following:\n\n                - HKEY_LOCAL_MACHINE or HKLM\n                - HKEY_CURRENT_USER or HKCU\n                - HKEY_USERS or HKU\n                - HKEY_CLASSES_ROOT or HKCR\n                - HKEY_CURRENT_CONFIG or HKCC\n\n        key (str):\n            The key (looks like a path) to the value name. If a key is not\n            passed, the keys under the hive will be returned.\n\n        use_32bit_registry (bool):\n            Accesses the 32bit portion of the registry on 64 bit installations.\n            On 32bit machines this is ignored.\n\n    Returns:\n        list: A list of keys/subkeys under the hive or key.\n\n    Usage:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.list_keys(hive='HKLM', key='SOFTWARE\\Microsoft')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    access_mask = registry.registry_32[use_32bit_registry]
    subkeys = []
    handle = None
    try:
        handle = win32api.RegOpenKeyEx(hkey, local_key, 0, access_mask)
        for i in range(win32api.RegQueryInfoKey(handle)[0]):
            subkey = win32api.RegEnumKey(handle, i)
            subkeys.append(subkey)
    except win32api.error as exc:
        if exc.winerror == 2:
            log.debug('Cannot find key: %s\\%s', hive, key, exc_info=True)
            return (False, 'Cannot find key: {}\\{}'.format(hive, key))
        if exc.winerror == 5:
            log.debug('Access is denied: %s\\%s', hive, key, exc_info=True)
            return (False, 'Access is denied: {}\\{}'.format(hive, key))
        raise
    finally:
        if handle:
            win32api.RegCloseKey(handle)
    return subkeys

def list_values(hive, key=None, use_32bit_registry=False):
    if False:
        return 10
    "\n    Enumerates the values in a registry key or hive.\n\n    .. note::\n        The ``(Default)`` value will only be returned if it is set, otherwise it\n        will not be returned in the list of values.\n\n    Args:\n\n        hive (str):\n            The name of the hive. Can be one of the following:\n\n                - HKEY_LOCAL_MACHINE or HKLM\n                - HKEY_CURRENT_USER or HKCU\n                - HKEY_USER or HKU\n                - HKEY_CLASSES_ROOT or HKCR\n                - HKEY_CURRENT_CONFIG or HKCC\n\n        key (str):\n            The key (looks like a path) to the value name. If a key is not\n            passed, the values under the hive will be returned.\n\n        use_32bit_registry (bool):\n            Accesses the 32bit portion of the registry on 64 bit installations.\n            On 32bit machines this is ignored.\n\n    Returns:\n        list: A list of values under the hive or key.\n\n    Usage:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.list_values(hive='HKLM', key='SYSTEM\\CurrentControlSet\\Services\\Tcpip')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    access_mask = registry.registry_32[use_32bit_registry]
    handle = None
    values = list()
    try:
        handle = win32api.RegOpenKeyEx(hkey, local_key, 0, access_mask)
        for i in range(win32api.RegQueryInfoKey(handle)[1]):
            (vname, vdata, vtype) = win32api.RegEnumValue(handle, i)
            if not vname:
                vname = '(Default)'
            value = {'hive': local_hive, 'key': local_key, 'vname': _to_mbcs(vname), 'vtype': registry.vtype_reverse[vtype], 'success': True}
            if vtype == win32con.REG_MULTI_SZ:
                value['vdata'] = [_to_mbcs(i) for i in vdata]
            elif vtype in [win32con.REG_SZ, win32con.REG_EXPAND_SZ]:
                value['vdata'] = _to_mbcs(vdata)
            else:
                value['vdata'] = vdata
            values.append(value)
    except win32api.error as exc:
        if exc.winerror == 2:
            log.debug('Cannot find key: %s\\%s', hive, key)
            return (False, 'Cannot find key: {}\\{}'.format(hive, key))
        elif exc.winerror == 5:
            log.debug('Access is denied: %s\\%s', hive, key)
            return (False, 'Access is denied: {}\\{}'.format(hive, key))
        raise
    finally:
        if handle:
            win32api.RegCloseKey(handle)
    return values

def read_value(hive, key, vname=None, use_32bit_registry=False):
    if False:
        while True:
            i = 10
    "\n    Reads a registry value entry or the default value for a key. To read the\n    default value, don't pass ``vname``\n\n    Args:\n\n        hive (str): The name of the hive. Can be one of the following:\n\n            - HKEY_LOCAL_MACHINE or HKLM\n            - HKEY_CURRENT_USER or HKCU\n            - HKEY_USER or HKU\n            - HKEY_CLASSES_ROOT or HKCR\n            - HKEY_CURRENT_CONFIG or HKCC\n\n        key (str):\n            The key (looks like a path) to the value name.\n\n        vname (str):\n            The value name. These are the individual name/data pairs under the\n            key. If not passed, the key (Default) value will be returned.\n\n        use_32bit_registry (bool):\n            Accesses the 32bit portion of the registry on 64bit installations.\n            On 32bit machines this is ignored.\n\n    Returns:\n        dict: A dictionary containing the passed settings as well as the\n            value_data if successful. If unsuccessful, sets success to False.\n\n        bool: Returns False if the key is not found\n\n        If vname is not passed:\n\n            - Returns the first unnamed value (Default) as a string.\n            - Returns none if first unnamed value is empty.\n\n    Usage:\n\n        The following will get the value of the ``version`` value name in the\n        ``HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Salt`` key\n\n        .. code-block:: python\n\n            import salt.utils.win_reg as reg\n            reg.read_value(hive='HKLM', key='SOFTWARE\\\\Salt', vname='version')\n\n    Usage:\n\n        The following will get the default value of the\n        ``HKEY_LOCAL_MACHINE\\\\SOFTWARE\\\\Salt`` key\n\n        .. code-block:: python\n\n            import salt.utils.win_reg as reg\n            reg.read_value(hive='HKLM', key='SOFTWARE\\\\Salt')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    local_vname = _to_unicode(vname)
    ret = {'hive': local_hive, 'key': local_key, 'vname': local_vname, 'vdata': None, 'vtype': None, 'success': True}
    if not vname:
        ret['vname'] = '(Default)'
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    access_mask = registry.registry_32[use_32bit_registry]
    handle = None
    try:
        handle = win32api.RegOpenKeyEx(hkey, local_key, 0, access_mask)
        try:
            (vdata, vtype) = win32api.RegQueryValueEx(handle, local_vname)
            if vdata or vdata in [0, '', []]:
                ret['vtype'] = registry.vtype_reverse[vtype]
                if vtype == win32con.REG_MULTI_SZ:
                    ret['vdata'] = [_to_mbcs(i) for i in vdata]
                elif vtype in [win32con.REG_SZ, win32con.REG_EXPAND_SZ]:
                    ret['vdata'] = _to_mbcs(vdata)
                else:
                    ret['vdata'] = vdata
            else:
                ret['comment'] = 'Empty Value'
        except win32api.error as exc:
            if exc.winerror == 2 and vname is None:
                ret['vdata'] = '(value not set)'
                ret['vtype'] = 'REG_SZ'
            elif exc.winerror == 2:
                msg = 'Cannot find {} in {}\\{}'.format(local_vname, local_hive, local_key)
                log.trace(exc)
                log.trace(msg)
                ret['comment'] = msg
                ret['success'] = False
            else:
                raise
    except win32api.error as exc:
        if exc.winerror == 2:
            msg = 'Cannot find key: {}\\{}'.format(local_hive, local_key)
            log.trace(exc)
            log.trace(msg)
            ret['comment'] = msg
            ret['success'] = False
        elif exc.winerror == 5:
            msg = 'Access is denied: {}\\{}'.format(local_hive, local_key)
            log.trace(exc)
            log.trace(msg)
            ret['comment'] = msg
            ret['success'] = False
        else:
            raise
    finally:
        if handle:
            win32api.RegCloseKey(handle)
    return ret

def set_value(hive, key, vname=None, vdata=None, vtype='REG_SZ', use_32bit_registry=False, volatile=False):
    if False:
        return 10
    "\n    Sets a value in the registry. If ``vname`` is passed, it will be the value\n    for that value name, otherwise it will be the default value for the\n    specified key\n\n    Args:\n\n        hive (str):\n            The name of the hive. Can be one of the following\n\n                - HKEY_LOCAL_MACHINE or HKLM\n                - HKEY_CURRENT_USER or HKCU\n                - HKEY_USER or HKU\n                - HKEY_CLASSES_ROOT or HKCR\n                - HKEY_CURRENT_CONFIG or HKCC\n\n        key (str):\n            The key (looks like a path) to the value name.\n\n        vname (str):\n            The value name. These are the individual name/data pairs under the\n            key. If not passed, the key (Default) value will be set.\n\n        vdata (str, int, list, bytes):\n            The value you'd like to set. If a value name (vname) is passed, this\n            will be the data for that value name. If not, this will be the\n            (Default) value for the key.\n\n            The type of data this parameter expects is determined by the value\n            type specified in ``vtype``. The correspondence is as follows:\n\n                - REG_BINARY: Binary data (bytes)\n                - REG_DWORD: int\n                - REG_EXPAND_SZ: str\n                - REG_MULTI_SZ: list of str\n                - REG_QWORD: int\n                - REG_SZ: str\n\n                .. note::\n                    When setting REG_BINARY, string data will be converted to\n                    binary. You can pass base64 encoded using the ``binascii``\n                    built-in module. Use ``binascii.b2a_base64('your data')``\n\n            .. note::\n                The type for the (Default) value is always REG_SZ and cannot be\n                changed.\n\n            .. note::\n                This parameter is optional. If not passed, the Key will be\n                created with no associated item/value pairs.\n\n        vtype (str):\n            The value type. The possible values of the vtype parameter are\n            indicated above in the description of the vdata parameter.\n\n        use_32bit_registry (bool):\n            Sets the 32bit portion of the registry on 64bit installations. On\n            32bit machines this is ignored.\n\n        volatile (bool):\n            When this parameter has a value of True, the registry key will be\n            made volatile (i.e. it will not persist beyond a system reset or\n            shutdown). This parameter only has an effect when a key is being\n            created and at no other time.\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    Usage:\n\n        This will set the version value to 2015.5.2 in the SOFTWARE\\Salt key in\n        the HKEY_LOCAL_MACHINE hive\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.set_value(hive='HKLM', key='SOFTWARE\\Salt', vname='version', vdata='2015.5.2')\n\n    Usage:\n\n        This function is strict about the type of vdata. For instance this\n        example will fail because vtype has a value of REG_SZ and vdata has a\n        type of int (as opposed to str as expected).\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.set_value(hive='HKLM', key='SOFTWARE\\Salt', vname='str_data', vdata=1.2)\n\n    Usage:\n\n        In this next example vdata is properly quoted and should succeed.\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.set_value(hive='HKLM', key='SOFTWARE\\Salt', vname='str_data', vdata='1.2')\n\n    Usage:\n\n        This is an example of using vtype REG_BINARY. Both ``set_value``\n        commands will set the same value ``Salty Test``\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.set_value(hive='HKLM', key='SOFTWARE\\Salt', vname='bin_data', vdata='Salty Test', vtype='REG_BINARY')\n\n            import binascii\n            bin_data = binascii.b2a_base64('Salty Test')\n            winreg.set_value(hive='HKLM', key='SOFTWARE\\Salt', vname='bin_data_encoded', vdata=bin_data, vtype='REG_BINARY')\n\n    Usage:\n\n        An example using vtype REG_MULTI_SZ is as follows:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.set_value(hive='HKLM', key='SOFTWARE\\Salt', vname='list_data', vdata=['Salt', 'is', 'great'], vtype='REG_MULTI_SZ')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    local_vname = _to_unicode(vname)
    local_vtype = _to_unicode(vtype)
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    vtype_value = registry.vtype[local_vtype]
    access_mask = registry.registry_32[use_32bit_registry] | win32con.KEY_ALL_ACCESS
    local_vdata = cast_vdata(vdata=vdata, vtype=local_vtype)
    if volatile:
        create_options = registry.opttype['REG_OPTION_VOLATILE']
    else:
        create_options = registry.opttype['REG_OPTION_NON_VOLATILE']
    handle = None
    try:
        (handle, result) = win32api.RegCreateKeyEx(hkey, local_key, access_mask, Options=create_options)
        msg = 'Created new key: %s\\%s' if result == 1 else 'Opened existing key: %s\\%s'
        log.debug(msg, local_hive, local_key)
        try:
            win32api.RegSetValueEx(handle, local_vname, 0, vtype_value, local_vdata)
            win32api.RegFlushKey(handle)
            broadcast_change()
            return True
        except TypeError as exc:
            log.exception('"vdata" does not match the expected data type.\n%s', exc)
            return False
        except (SystemError, ValueError) as exc:
            log.exception('Encountered error setting registry value.\n%s', exc)
            return False
    except win32api.error as exc:
        log.exception('Error creating/opening key: %s\\%s\n%s\n%s', local_hive, local_key, exc.winerror, exc.strerror)
        return False
    finally:
        if handle:
            win32api.RegCloseKey(handle)

def cast_vdata(vdata=None, vtype='REG_SZ'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Cast the ``vdata` value to the appropriate data type for the registry type\n    specified in ``vtype``\n\n    Args:\n\n        vdata (str, int, list, bytes): The data to cast\n\n        vtype (str):\n            The type of data to be written to the registry. Must be one of the\n            following:\n\n                - REG_BINARY\n                - REG_DWORD\n                - REG_EXPAND_SZ\n                - REG_MULTI_SZ\n                - REG_QWORD\n                - REG_SZ\n\n    Returns:\n        The vdata cast to the appropriate type. Will be unicode string, binary,\n        list of unicode strings, or int\n\n    Usage:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.cast_vdata(vdata='This is the string', vtype='REG_SZ')\n    "
    registry = Registry()
    vtype_value = registry.vtype[vtype]
    if vtype_value in [win32con.REG_SZ, win32con.REG_EXPAND_SZ]:
        return _to_unicode(vdata)
    elif vtype_value == win32con.REG_BINARY:
        if isinstance(vdata, str):
            return vdata.encode('utf-8')
        return vdata
    elif vtype_value == win32con.REG_MULTI_SZ:
        return [_to_unicode(i) for i in vdata]
    elif vtype_value == win32con.REG_QWORD:
        return int(vdata)
    else:
        return int(vdata)

def delete_key_recursive(hive, key, use_32bit_registry=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2015.5.4\n\n    Delete a registry key to include all subkeys and value/data pairs.\n\n    Args:\n\n        hive (str):\n            The name of the hive. Can be one of the following\n\n                - HKEY_LOCAL_MACHINE or HKLM\n                - HKEY_CURRENT_USER or HKCU\n                - HKEY_USER or HKU\n                - HKEY_CLASSES_ROOT or HKCR\n                - HKEY_CURRENT_CONFIG or HKCC\n\n            key (str):\n                The key to remove (looks like a path)\n\n            use_32bit_registry (bool):\n                Deletes the 32bit portion of the registry on 64bit\n                installations. On 32bit machines this is ignored.\n\n    Returns:\n        dict: A dictionary listing the keys that deleted successfully as well as\n            those that failed to delete.\n\n    Usage:\n\n        The following example will remove ``salt`` and all its subkeys from the\n        ``SOFTWARE`` key in ``HKEY_LOCAL_MACHINE``:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.delete_key_recursive(hive='HKLM', key='SOFTWARE\\DeleteMe')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    key_path = local_key
    access_mask = registry.registry_32[use_32bit_registry] | win32con.KEY_ALL_ACCESS
    if not key_exists(local_hive, local_key, use_32bit_registry):
        log.debug('"%s\\%s" not found', hive, key)
        return False
    if len(key) > 1 and key.count('\\', 1) < registry.subkey_slash_check[hkey]:
        log.error('"%s\\%s" is too close to root, not safe to remove', hive, key)
        return False

    def _subkeys(_key):
        if False:
            while True:
                i = 10
        '\n        Enumerate keys\n        '
        i = 0
        while True:
            try:
                subkey = win32api.RegEnumKey(_key, i)
                yield _to_mbcs(subkey)
                i += 1
            except win32api.error:
                break

    def _traverse_registry_tree(_hkey, _keypath, _ret, _access_mask):
        if False:
            while True:
                i = 10
        '\n        Traverse the registry tree i.e. dive into the tree\n        '
        _key = win32api.RegOpenKeyEx(_hkey, _keypath, 0, _access_mask)
        for subkeyname in _subkeys(_key):
            subkeypath = '{}\\{}'.format(_keypath, subkeyname)
            _ret = _traverse_registry_tree(_hkey, subkeypath, _ret, access_mask)
            _ret.append(subkeypath)
        return _ret
    key_list = []
    key_list = _traverse_registry_tree(hkey, key_path, key_list, access_mask)
    key_list.append(key_path)
    ret = {'Deleted': [], 'Failed': []}
    for sub_key_path in key_list:
        key_handle = None
        try:
            key_handle = win32api.RegOpenKeyEx(hkey, sub_key_path, 0, access_mask)
            try:
                win32api.RegDeleteKey(key_handle, '')
                ret['Deleted'].append('{}\\{}'.format(hive, sub_key_path))
            except OSError as exc:
                log.error(exc, exc_info=True)
                ret['Failed'].append('{}\\{} {}'.format(hive, sub_key_path, exc))
        except win32api.error as exc:
            log.error(exc, exc_info=True)
            ret['Failed'].append('{}\\{} {}'.format(hive, sub_key_path, exc.strerror))
        finally:
            if key_handle:
                win32api.RegCloseKey(key_handle)
    broadcast_change()
    return ret

def delete_value(hive, key, vname=None, use_32bit_registry=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a registry value entry or the default value for a key.\n\n    Args:\n\n        hive (str):\n            The name of the hive. Can be one of the following\n\n                - HKEY_LOCAL_MACHINE or HKLM\n                - HKEY_CURRENT_USER or HKCU\n                - HKEY_USER or HKU\n                - HKEY_CLASSES_ROOT or HKCR\n                - HKEY_CURRENT_CONFIG or HKCC\n\n        key (str):\n            The key (looks like a path) to the value name.\n\n        vname (str):\n            The value name. These are the individual name/data pairs under the\n            key. If not passed, the key (Default) value will be deleted.\n\n        use_32bit_registry (bool):\n            Deletes the 32bit portion of the registry on 64bit installations. On\n            32bit machines this is ignored.\n\n    Return:\n        bool: True if successful, otherwise False\n\n    Usage:\n\n        .. code-block:: python\n\n            import salt.utils.win_reg\n            winreg.delete_value(hive='HKLM', key='SOFTWARE\\SaltTest', vname='version')\n    "
    local_hive = _to_unicode(hive)
    local_key = _to_unicode(key)
    local_vname = _to_unicode(vname)
    registry = Registry()
    try:
        hkey = registry.hkeys[local_hive]
    except KeyError:
        raise CommandExecutionError('Invalid Hive: {}'.format(local_hive))
    access_mask = registry.registry_32[use_32bit_registry] | win32con.KEY_ALL_ACCESS
    handle = None
    try:
        handle = win32api.RegOpenKeyEx(hkey, local_key, 0, access_mask)
        win32api.RegDeleteValue(handle, local_vname)
        broadcast_change()
        return True
    except win32api.error as exc:
        if exc.winerror == 2:
            return None
        raise
    finally:
        if handle:
            win32api.RegCloseKey(handle)