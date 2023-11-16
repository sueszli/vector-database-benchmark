"""
Utility module for dealing with Windows Registry.
"""
import os
import re
import sys
from contextlib import contextmanager
from llnl.util import tty
if sys.platform == 'win32':
    import winreg

class RegistryValue:
    """
    Class defining a Windows registry entry
    """

    def __init__(self, name, value, parent_key):
        if False:
            i = 10
            return i + 15
        self.path = name
        self.value = value
        self.key = parent_key

class RegistryKey:
    """
    Class wrapping a Windows registry key
    """

    def __init__(self, name, handle):
        if False:
            for i in range(10):
                print('nop')
        self.path = name
        self.name = os.path.split(name)[-1]
        self._handle = handle
        self._keys = []
        self._values = {}

    @property
    def values(self):
        if False:
            i = 10
            return i + 15
        'Returns all subvalues of this key as RegistryValue objects in dictionary\n        of value name : RegistryValue object\n        '
        self._gather_value_info()
        return self._values

    @property
    def subkeys(self):
        if False:
            i = 10
            return i + 15
        'Returns list of all subkeys of this key as RegistryKey objects'
        self._gather_subkey_info()
        return self._keys

    @property
    def hkey(self):
        if False:
            for i in range(10):
                print('nop')
        return self._handle

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def _gather_subkey_info(self):
        if False:
            while True:
                i = 10
        'Composes all subkeys into a list for access'
        if self._keys:
            return
        (sub_keys, _, _) = winreg.QueryInfoKey(self.hkey)
        for i in range(sub_keys):
            sub_name = winreg.EnumKey(self.hkey, i)
            try:
                sub_handle = winreg.OpenKeyEx(self.hkey, sub_name, access=winreg.KEY_READ)
                self._keys.append(RegistryKey(os.path.join(self.path, sub_name), sub_handle))
            except OSError as e:
                if hasattr(e, 'winerror'):
                    if e.winerror == 5:
                        pass
                    else:
                        raise
                else:
                    raise

    def _gather_value_info(self):
        if False:
            print('Hello World!')
        'Compose all values for this key into a dict of form value name: RegistryValue Object'
        if self._values:
            return
        (_, values, _) = winreg.QueryInfoKey(self.hkey)
        for i in range(values):
            (value_name, value_data, _) = winreg.EnumValue(self.hkey, i)
            self._values[value_name] = RegistryValue(value_name, value_data, self.hkey)

    def get_subkey(self, sub_key):
        if False:
            i = 10
            return i + 15
        'Returns subkey of name sub_key in a RegistryKey objects'
        return RegistryKey(os.path.join(self.path, sub_key), winreg.OpenKeyEx(self.hkey, sub_key, access=winreg.KEY_READ))

    def get_value(self, val_name):
        if False:
            i = 10
            return i + 15
        'Returns value associated with this key in RegistryValue object'
        return RegistryValue(val_name, winreg.QueryValueEx(self.hkey, val_name)[0], self.hkey)

class _HKEY_CONSTANT(RegistryKey):
    """Subclass of RegistryKey to represent the prebaked, always open registry HKEY constants"""

    def __init__(self, hkey_constant):
        if False:
            for i in range(10):
                print('nop')
        hkey_name = hkey_constant
        super(_HKEY_CONSTANT, self).__init__(hkey_name, None)

    def _get_hkey(self, key):
        if False:
            return 10
        return getattr(winreg, key)

    @property
    def hkey(self):
        if False:
            print('Hello World!')
        if not self._handle:
            self._handle = self._get_hkey(self.path)
        return self._handle

class HKEY:
    """
    Predefined, open registry HKEYs
    From the Microsoft docs:
    An application must open a key before it can read data from the registry.
    To open a key, an application must supply a handle to another key in
    the registry that is already open. The system defines predefined keys
    that are always open. Predefined keys help an application navigate in
    the registry."""
    HKEY_CLASSES_ROOT = _HKEY_CONSTANT('HKEY_CLASSES_ROOT')
    HKEY_CURRENT_USER = _HKEY_CONSTANT('HKEY_CURRENT_USER')
    HKEY_USERS = _HKEY_CONSTANT('HKEY_USERS')
    HKEY_LOCAL_MACHINE = _HKEY_CONSTANT('HKEY_LOCAL_MACHINE')
    HKEY_CURRENT_CONFIG = _HKEY_CONSTANT('HKEY_CURRENT_CONFIG')
    HKEY_PERFORMANCE_DATA = _HKEY_CONSTANT('HKEY_PERFORMANCE_DATA')

class WindowsRegistryView:
    """
    Interface to provide access, querying, and searching to Windows registry entries.
    This class represents a single key entrypoint into the Windows registry
    and provides an interface to this key's values, its subkeys, and those subkey's values.
    This class cannot be used to move freely about the registry, only subkeys/values of
    the root key used to instantiate this class.
    """

    def __init__(self, key, root_key=HKEY.HKEY_CURRENT_USER):
        if False:
            i = 10
            return i + 15
        'Constructs a Windows Registry entrypoint to key provided\n        root_key should be an already open root key or an hkey constant if provided\n\n        Args:\n            key (str): registry key to provide root for registry key for this clas\n            root_key: Already open registry key or HKEY constant to provide access into\n                         the Windows registry. Registry access requires an already open key\n                         to get an entrypoint, the HKEY constants are always open, or an already\n                         open key can be used instead.\n        '
        if sys.platform != 'win32':
            raise RuntimeError('Cannot instantiate Windows Registry class on non Windows platforms')
        self.key = key
        self.root = root_key
        self._reg = None

    class KeyMatchConditions:

        @staticmethod
        def regex_matcher(subkey_name):
            if False:
                for i in range(10):
                    print('nop')
            return lambda x: re.match(subkey_name, x.name)

        @staticmethod
        def name_matcher(subkey_name):
            if False:
                return 10
            return lambda x: subkey_name == x.name

    @contextmanager
    def invalid_reg_ref_error_handler(self):
        if False:
            while True:
                i = 10
        try:
            yield
        except FileNotFoundError as e:
            if sys.platform == 'win32' and e.winerror == 2:
                tty.debug('Key %s at position %s does not exist' % (self.key, str(self.root)))
            else:
                raise e

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return self.reg != -1

    def _load_key(self):
        if False:
            i = 10
            return i + 15
        try:
            self._reg = RegistryKey(os.path.join(str(self.root), self.key), winreg.OpenKeyEx(self.root.hkey, self.key, access=winreg.KEY_READ))
        except FileNotFoundError as e:
            if sys.platform == 'win32' and e.winerror == 2:
                self._reg = -1
                tty.debug('Key %s at position %s does not exist' % (self.key, str(self.root)))
            else:
                raise e

    def _valid_reg_check(self):
        if False:
            return 10
        if self.reg == -1:
            tty.debug('Cannot perform operation for nonexistent key %s' % self.key)
            return False
        return True

    def _regex_match_subkeys(self, subkey):
        if False:
            print('Hello World!')
        r_subkey = re.compile(subkey)
        return [key for key in self.get_subkeys() if r_subkey.match(key.name)]

    @property
    def reg(self):
        if False:
            return 10
        if not self._reg:
            self._load_key()
        return self._reg

    def get_value(self, value_name):
        if False:
            print('Hello World!')
        'Return registry value corresponding to provided argument (if it exists)'
        if not self._valid_reg_check():
            raise RegistryError('Cannot query value from invalid key %s' % self.key)
        with self.invalid_reg_ref_error_handler():
            return self.reg.get_value(value_name)

    def get_subkey(self, subkey_name):
        if False:
            return 10
        if not self._valid_reg_check():
            raise RegistryError('Cannot query subkey from invalid key %s' % self.key)
        with self.invalid_reg_ref_error_handler():
            return self.reg.get_subkey(subkey_name)

    def get_subkeys(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._valid_reg_check():
            raise RegistryError('Cannot query subkeys from invalid key %s' % self.key)
        with self.invalid_reg_ref_error_handler():
            return self.reg.subkeys

    def get_matching_subkeys(self, subkey_name):
        if False:
            print('Hello World!')
        'Returns all subkeys regex matching subkey name\n\n        Note: this method obtains only direct subkeys of the given key and does not\n        desced to transtitve subkeys. For this behavior, see `find_matching_subkeys`'
        self._regex_match_subkeys(subkey_name)

    def get_values(self):
        if False:
            return 10
        if not self._valid_reg_check():
            raise RegistryError('Cannot query values from invalid key %s' % self.key)
        with self.invalid_reg_ref_error_handler():
            return self.reg.values

    def _traverse_subkeys(self, stop_condition, collect_all_matching=False):
        if False:
            i = 10
            return i + 15
        "Perform simple BFS of subkeys, returning the key\n        that successfully triggers the stop condition.\n        Args:\n            stop_condition: lambda or function pointer that takes a single argument\n                            a key and returns a boolean value based on that key\n            collect_all_matching: boolean value, if True, the traversal collects and returns\n                            all keys meeting stop condition. If false, once stop\n                            condition is met, the key that triggered the condition '\n                            is returned.\n        Return:\n            the key if stop_condition is triggered, or None if not\n        "
        collection = []
        if not self._valid_reg_check():
            raise RegistryError('Cannot query values from invalid key %s' % self.key)
        with self.invalid_reg_ref_error_handler():
            queue = self.reg.subkeys
            for key in queue:
                if stop_condition(key):
                    if collect_all_matching:
                        collection.append(key)
                    else:
                        return key
                queue.extend(key.subkeys)
            return collection if collection else None

    def _find_subkey_s(self, search_key, collect_all_matching=False):
        if False:
            i = 10
            return i + 15
        'Retrieve one or more keys regex matching `search_key`.\n        One key will be returned unless `collect_all_matching` is enabled,\n        in which case call matches are returned.\n\n        Args:\n            search_key (str): regex string represeting a subkey name structure\n                              to be matched against.\n                              Cannot be provided alongside `direct_subkey`\n            collect_all_matching (bool): No-op if `direct_subkey` is specified\n        Return:\n            the desired subkey as a RegistryKey object, or none\n        '
        return self._traverse_subkeys(search_key, collect_all_matching=collect_all_matching)

    def find_subkey(self, subkey_name):
        if False:
            i = 10
            return i + 15
        'Perform a BFS of subkeys until desired key is found\n        Returns None or RegistryKey object corresponding to requested key name\n\n        Args:\n            subkey_name (str)\n        Return:\n            the desired subkey as a RegistryKey object, or none\n\n        For more details, see the WindowsRegistryView._find_subkey_s method docstring\n        '
        return self._find_subkey_s(WindowsRegistryView.KeyMatchConditions.name_matcher(subkey_name))

    def find_matching_subkey(self, subkey_name):
        if False:
            for i in range(10):
                print('nop')
        'Perform a BFS of subkeys until a key matching subkey name regex is found\n        Returns None or the first RegistryKey object corresponding to requested key name\n\n        Args:\n            subkey_name (str)\n        Return:\n            the desired subkey as a RegistryKey object, or none\n\n        For more details, see the WindowsRegistryView._find_subkey_s method docstring\n        '
        return self._find_subkey_s(WindowsRegistryView.KeyMatchConditions.regex_matcher(subkey_name))

    def find_subkeys(self, subkey_name):
        if False:
            for i in range(10):
                print('nop')
        'Exactly the same as find_subkey, except this function tries to match\n        a regex to multiple keys\n\n        Args:\n            subkey_name (str)\n        Return:\n            the desired subkeys as a list of RegistryKey object, or none\n\n        For more details, see the WindowsRegistryView._find_subkey_s method docstring\n        '
        kwargs = {'collect_all_matching': True}
        return self._find_subkey_s(WindowsRegistryView.KeyMatchConditions.regex_matcher(subkey_name), **kwargs)

    def find_value(self, val_name, recursive=True):
        if False:
            i = 10
            return i + 15
        '\n        If non recursive, return RegistryValue object corresponding to name\n\n        Args:\n            val_name (str): name of value desired from registry\n            recursive (bool): optional argument, if True, the registry is searched recursively\n                              for the value of name val_name, else only the current key is searched\n        Return:\n            The desired registry value as a RegistryValue object if it exists, otherwise, None\n        '
        if not recursive:
            return self.get_value(val_name)
        else:
            key = self._traverse_subkeys(lambda x: val_name in x.values)
            if not key:
                return None
            else:
                return key.values[val_name]

class RegistryError(RuntimeError):
    """Runtime Error describing issue with invalid key access to Windows registry"""