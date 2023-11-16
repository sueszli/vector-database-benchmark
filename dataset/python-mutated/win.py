"""
Collect information about software installed on Windows OS
================

:maintainer: Salt Stack <https://github.com/saltstack>
:codeauthor: Damon Atkins <https://github.com/damon-atkins>
:maturity: new
:depends: pywin32
:platform: windows

Known Issue: install_date may not match Control Panel\\Programs\\Programs and Features
"""
import collections
import datetime
import locale
import logging
import os.path
import platform
import re
import sys
import time
from functools import cmp_to_key
__version__ = '0.1'
try:
    import pywintypes
    import win32api
    import win32con
    import win32process
    import win32security
    import winerror
except ImportError:
    if __name__ == '__main__':
        raise ImportError('Please install pywin32/pypiwin32')
    else:
        raise
if __name__ == '__main__':
    LOG_CONSOLE = logging.StreamHandler()
    LOG_CONSOLE.setFormatter(logging.Formatter('[%(levelname)s]: %(message)s'))
    log = logging.getLogger(__name__)
    log.addHandler(LOG_CONSOLE)
    log.setLevel(logging.DEBUG)
else:
    log = logging.getLogger(__name__)
try:
    from salt.utils.odict import OrderedDict
except ImportError:
    from collections import OrderedDict
from salt.utils.versions import Version

class RegSoftwareInfo:
    """
    Retrieve Registry data on a single installed software item or component.

    Attribute:
        None

    :codeauthor: Damon Atkins <https://github.com/damon-atkins>
    """
    __guid_pattern = re.compile('^\\{(\\w{8})-(\\w{4})-(\\w{4})-(\\w\\w)(\\w\\w)-(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)\\}$')
    __squid_pattern = re.compile('^(\\w{8})(\\w{4})(\\w{4})(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)$')
    __version_pattern = re.compile('\\d+\\.\\d+\\.\\d+[\\w.-]*|\\d+\\.\\d+[\\w.-]*')
    __upgrade_codes = {}
    __upgrade_code_have_scan = {}
    __reg_types = {'str': (win32con.REG_EXPAND_SZ, win32con.REG_SZ), 'list': win32con.REG_MULTI_SZ, 'int': (win32con.REG_DWORD, win32con.REG_DWORD_BIG_ENDIAN, win32con.REG_QWORD), 'bytes': win32con.REG_BINARY}
    if platform.architecture()[0] == '32bit':
        if win32process.IsWow64Process():
            __use_32bit_lookup = {True: 0, False: win32con.KEY_WOW64_64KEY}
        else:
            __use_32bit_lookup = {True: 0, False: None}
    else:
        __use_32bit_lookup = {True: win32con.KEY_WOW64_32KEY, False: 0}

    def __init__(self, key_guid, sid=None, use_32bit=False):
        if False:
            print('Hello World!')
        '\n        Initialise against a software item or component.\n\n        All software has a unique "Identifer" within the registry. This can be free\n        form text/numbers e.g. "MySoftware" or\n        GUID e.g. "{0EAF0D8F-C9CF-4350-BD9A-07EC66929E04}"\n\n        Args:\n            key_guid (str): Identifer.\n            sid (str): Security IDentifier of the User or None for Computer/Machine.\n            use_32bit (bool):\n                Regisrty location of the Identifer. ``True`` 32 bit registry only\n                meaning fully on 64 bit OS.\n        '
        self.__reg_key_guid = key_guid
        self.__squid = ''
        self.__reg_products_path = ''
        self.__reg_upgradecode_path = ''
        self.__patch_list = None
        guid_match = self.__guid_pattern.match(key_guid)
        if guid_match is not None:
            for index in range(1, 12):
                self.__squid += guid_match.group(index)[::-1]
        if sid:
            self.__reg_hive = 'HKEY_USERS'
            self.__reg_32bit = False
            self.__reg_32bit_access = 0
            self.__reg_uninstall_path = '{}\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{}'.format(sid, key_guid)
            if self.__squid:
                self.__reg_products_path = '{}\\Software\\Classes\\Installer\\Products\\{}'.format(sid, self.__squid)
                self.__reg_upgradecode_path = '{}\\Software\\Microsoft\\Installer\\UpgradeCodes'.format(sid)
                self.__reg_patches_path = 'Software\\Microsoft\\Windows\\CurrentVersion\\Installer\\UserData\\{}\\Products\\{}\\Patches'.format(sid, self.__squid)
        else:
            self.__reg_hive = 'HKEY_LOCAL_MACHINE'
            self.__reg_32bit = use_32bit
            self.__reg_32bit_access = self.__use_32bit_lookup[use_32bit]
            self.__reg_uninstall_path = 'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{}'.format(key_guid)
            if self.__squid:
                self.__reg_products_path = 'Software\\Classes\\Installer\\Products\\{}'.format(self.__squid)
                self.__reg_upgradecode_path = 'Software\\Classes\\Installer\\UpgradeCodes'
                self.__reg_patches_path = 'Software\\Microsoft\\Windows\\CurrentVersion\\Installer\\UserData\\S-1-5-18\\Products\\{}\\Patches'.format(self.__squid)
        try:
            self.__reg_uninstall_handle = win32api.RegOpenKeyEx(getattr(win32con, self.__reg_hive), self.__reg_uninstall_path, 0, win32con.KEY_READ | self.__reg_32bit_access)
        except pywintypes.error as exc:
            if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                log.error("Software/Component Not Found  key_guid: '%s', sid: '%s' , use_32bit: '%s'", key_guid, sid, use_32bit)
            raise
        self.__reg_products_handle = None
        if self.__squid:
            try:
                self.__reg_products_handle = win32api.RegOpenKeyEx(getattr(win32con, self.__reg_hive), self.__reg_products_path, 0, win32con.KEY_READ | self.__reg_32bit_access)
            except pywintypes.error as exc:
                if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                    log.debug("Software/Component Not Found in Products section of registry key_guid: '%s', sid: '%s', use_32bit: '%s'", key_guid, sid, use_32bit)
                    self.__squid = None
                else:
                    raise
        self.__mod_time1970 = 0
        mod_win_time = win32api.RegQueryInfoKeyW(self.__reg_uninstall_handle).get('LastWriteTime', None)
        if mod_win_time:
            if hasattr(mod_win_time, 'utctimetuple'):
                self.__mod_time1970 = time.mktime(mod_win_time.utctimetuple())
            elif hasattr(mod_win_time, '__int__'):
                self.__mod_time1970 = int(mod_win_time)

    def __squid_to_guid(self, squid):
        if False:
            print('Hello World!')
        '\n        Squished GUID (SQUID) to GUID.\n\n        A SQUID is a Squished/Compressed version of a GUID to use up less space\n        in the registry.\n\n        Args:\n            squid (str): Squished GUID.\n\n        Returns:\n            str: the GUID if a valid SQUID provided.\n        '
        if not squid:
            return ''
        squid_match = self.__squid_pattern.match(squid)
        guid = ''
        if squid_match is not None:
            guid = '{' + squid_match.group(1)[::-1] + '-' + squid_match.group(2)[::-1] + '-' + squid_match.group(3)[::-1] + '-' + squid_match.group(4)[::-1] + squid_match.group(5)[::-1] + '-'
            for index in range(6, 12):
                guid += squid_match.group(index)[::-1]
            guid += '}'
        return guid

    @staticmethod
    def __one_equals_true(value):
        if False:
            return 10
        '\n        Test for ``1`` as a number or a string and return ``True`` if it is.\n\n        Args:\n            value: string or number or None.\n\n        Returns:\n            bool: ``True`` if 1 otherwise ``False``.\n        '
        if isinstance(value, int) and value == 1:
            return True
        elif isinstance(value, str) and re.match('\\d+', value, flags=re.IGNORECASE + re.UNICODE) is not None and (str(value) == '1'):
            return True
        return False

    @staticmethod
    def __reg_query_value(handle, value_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calls RegQueryValueEx\n\n        If PY2 ensure unicode string and expand REG_EXPAND_SZ before returning\n        Remember to catch not found exceptions when calling.\n\n        Args:\n            handle (object): open registry handle.\n            value_name (str): Name of the value you wished returned\n\n        Returns:\n            tuple: type, value\n        '
        (item_value, item_type) = win32api.RegQueryValueEx(handle, value_name)
        if item_type == win32con.REG_EXPAND_SZ:
            win32api.ExpandEnvironmentStrings(item_value)
            item_type = win32con.REG_SZ
        return (item_value, item_type)

    @property
    def install_time(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the install time, or provide an estimate of install time.\n\n        Installers or even self upgrading software must/should update the date\n        held within InstallDate field when they change versions. Some installers\n        do not set ``InstallDate`` at all so we use the last modified time on the\n        registry key.\n\n        Returns:\n            int: Seconds since 1970 UTC.\n        '
        time1970 = self.__mod_time1970
        try:
            (date_string, item_type) = win32api.RegQueryValueEx(self.__reg_uninstall_handle, 'InstallDate')
        except pywintypes.error as exc:
            if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                return time1970
            else:
                raise
        if item_type == win32con.REG_SZ:
            try:
                date_object = datetime.datetime.strptime(date_string, '%Y%m%d')
                time1970 = time.mktime(date_object.timetuple())
            except ValueError:
                pass
        return time1970

    def get_install_value(self, value_name, wanted_type=None):
        if False:
            print('Hello World!')
        '\n        For the uninstall section of the registry return the name value.\n\n        Args:\n            value_name (str): Registry value name.\n            wanted_type (str):\n                The type of value wanted if the type does not match\n                None is return. wanted_type support values are\n                ``str`` ``int`` ``list`` ``bytes``.\n\n        Returns:\n            value: Value requested or None if not found.\n        '
        try:
            (item_value, item_type) = self.__reg_query_value(self.__reg_uninstall_handle, value_name)
        except pywintypes.error as exc:
            if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                return None
            raise
        if wanted_type and item_type not in self.__reg_types[wanted_type]:
            item_value = None
        return item_value

    def is_install_true(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        For the uninstall section check if name value is ``1``.\n\n        Args:\n            value_name (str): Registry value name.\n\n        Returns:\n            bool: ``True`` if ``1`` otherwise ``False``.\n        '
        return self.__one_equals_true(self.get_install_value(key))

    def get_product_value(self, value_name, wanted_type=None):
        if False:
            return 10
        '\n        For the product section of the registry return the name value.\n\n        Args:\n            value_name (str): Registry value name.\n            wanted_type (str):\n                The type of value wanted if the type does not match\n                None is return. wanted_type support values are\n                ``str`` ``int`` ``list`` ``bytes``.\n\n        Returns:\n            value: Value requested or ``None`` if not found.\n        '
        if not self.__reg_products_handle:
            return None
        (subkey, search_value_name) = os.path.split(value_name)
        try:
            if subkey:
                handle = win32api.RegOpenKeyEx(self.__reg_products_handle, subkey, 0, win32con.KEY_READ | self.__reg_32bit_access)
                (item_value, item_type) = self.__reg_query_value(handle, search_value_name)
                win32api.RegCloseKey(handle)
            else:
                (item_value, item_type) = win32api.RegQueryValueEx(self.__reg_products_handle, value_name)
        except pywintypes.error as exc:
            if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                return None
            raise
        if wanted_type and item_type not in self.__reg_types[wanted_type]:
            item_value = None
        return item_value

    @property
    def upgrade_code(self):
        if False:
            i = 10
            return i + 15
        '\n        For installers which follow the Microsoft Installer standard, returns\n        the ``Upgrade code``.\n\n        Returns:\n            value (str): ``Upgrade code`` GUID for installed software.\n        '
        if not self.__squid:
            return ''
        have_scan_key = '{}\\{}\\{}'.format(self.__reg_hive, self.__reg_upgradecode_path, self.__reg_32bit)
        if not self.__upgrade_codes or self.__reg_key_guid not in self.__upgrade_codes:
            try:
                uc_handle = win32api.RegOpenKeyEx(getattr(win32con, self.__reg_hive), self.__reg_upgradecode_path, 0, win32con.KEY_READ | self.__reg_32bit_access)
            except pywintypes.error as exc:
                if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                    log.warning('Not Found %s\\%s 32bit %s', self.__reg_hive, self.__reg_upgradecode_path, self.__reg_32bit)
                    return ''
                raise
            (squid_upgrade_code_all, _, _, suc_pytime) = zip(*win32api.RegEnumKeyEx(uc_handle))
            if have_scan_key in self.__upgrade_code_have_scan and self.__upgrade_code_have_scan[have_scan_key] == (squid_upgrade_code_all, suc_pytime):
                log.debug('Scan skipped for upgrade codes, no changes (%s)', have_scan_key)
                return ''
            log.debug('Scan for upgrade codes (%s) for product codes', have_scan_key)
            for upgrade_code_squid in squid_upgrade_code_all:
                upgrade_code_guid = self.__squid_to_guid(upgrade_code_squid)
                pc_handle = win32api.RegOpenKeyEx(uc_handle, upgrade_code_squid, 0, win32con.KEY_READ | self.__reg_32bit_access)
                (_, pc_val_count, _) = win32api.RegQueryInfoKey(pc_handle)
                for item_index in range(pc_val_count):
                    product_code_guid = self.__squid_to_guid(win32api.RegEnumValue(pc_handle, item_index)[0])
                    if product_code_guid:
                        self.__upgrade_codes[product_code_guid] = upgrade_code_guid
                win32api.RegCloseKey(pc_handle)
            win32api.RegCloseKey(uc_handle)
            self.__upgrade_code_have_scan[have_scan_key] = (squid_upgrade_code_all, suc_pytime)
        return self.__upgrade_codes.get(self.__reg_key_guid, '')

    @property
    def list_patches(self):
        if False:
            print('Hello World!')
        '\n        For installers which follow the Microsoft Installer standard, returns\n        a list of patches applied.\n\n        Returns:\n            value (list): Long name of the patch.\n        '
        if not self.__squid:
            return []
        if self.__patch_list is None:
            try:
                pat_all_handle = win32api.RegOpenKeyEx(getattr(win32con, self.__reg_hive), self.__reg_patches_path, 0, win32con.KEY_READ | self.__reg_32bit_access)
            except pywintypes.error as exc:
                if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                    log.warning('Not Found %s\\%s 32bit %s', self.__reg_hive, self.__reg_patches_path, self.__reg_32bit)
                    return []
                raise
            (pc_sub_key_cnt, _, _) = win32api.RegQueryInfoKey(pat_all_handle)
            if not pc_sub_key_cnt:
                return []
            (squid_patch_all, _, _, _) = zip(*win32api.RegEnumKeyEx(pat_all_handle))
            ret = []
            for patch_squid in squid_patch_all:
                try:
                    patch_squid_handle = win32api.RegOpenKeyEx(pat_all_handle, patch_squid, 0, win32con.KEY_READ | self.__reg_32bit_access)
                    (patch_display_name, patch_display_name_type) = self.__reg_query_value(patch_squid_handle, 'DisplayName')
                    (patch_state, patch_state_type) = self.__reg_query_value(patch_squid_handle, 'State')
                    if patch_state_type != win32con.REG_DWORD or not isinstance(patch_state_type, int) or patch_state != 1 or (patch_display_name_type != win32con.REG_SZ):
                        continue
                    win32api.RegCloseKey(patch_squid_handle)
                    ret.append(patch_display_name)
                except pywintypes.error as exc:
                    if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                        log.debug('skipped patch, not found %s', patch_squid)
                        continue
                    raise
        return ret

    @property
    def registry_path_text(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the uninstall path this object is associated with.\n\n        Returns:\n            str: <hive>\\<uninstall registry entry>\n        '
        return '{}\\{}'.format(self.__reg_hive, self.__reg_uninstall_path)

    @property
    def registry_path(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the uninstall path this object is associated with.\n\n        Returns:\n            tuple: hive, uninstall registry entry path.\n        '
        return (self.__reg_hive, self.__reg_uninstall_path)

    @property
    def guid(self):
        if False:
            i = 10
            return i + 15
        '\n        Return GUID or Key.\n\n        Returns:\n            str: GUID or Key\n        '
        return self.__reg_key_guid

    @property
    def squid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return SQUID of the GUID if a valid GUID.\n\n        Returns:\n            str: GUID\n        '
        return self.__squid

    @property
    def package_code(self):
        if False:
            while True:
                i = 10
        '\n        Return package code of the software.\n\n        Returns:\n            str: GUID\n        '
        return self.__squid_to_guid(self.get_product_value('PackageCode'))

    @property
    def version_binary(self):
        if False:
            i = 10
            return i + 15
        '\n        Return version number which is stored in binary format.\n\n        Returns:\n            str: <major 0-255>.<minior 0-255>.<build 0-65535> or None if not found\n        '
        try:
            (item_value, item_type) = self.__reg_query_value(self.__reg_uninstall_handle, 'version')
        except pywintypes.error as exc:
            if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                return ('', '')
        version_binary_text = ''
        version_src = ''
        if item_value:
            if item_type == win32con.REG_DWORD:
                if isinstance(item_value, int):
                    version_binary_raw = item_value
                if version_binary_raw:
                    version_binary_text = '{}.{}.{}'.format(version_binary_raw >> 24 & 255, version_binary_raw >> 16 & 255, version_binary_raw & 65535)
                    version_src = 'binary-version'
            elif item_type == win32con.REG_SZ and isinstance(item_value, str) and (self.__version_pattern.match(item_value) is not None):
                version_binary_text = item_value.strip(' ')
                version_src = 'binary-version (string)'
        return (version_binary_text, version_src)

class WinSoftware:
    """
    Point in time snapshot of the software and components installed on
    a system.

    Attributes:
        None

    :codeauthor: Damon Atkins <https://github.com/damon-atkins>
    """
    __sid_pattern = re.compile('^S-\\d-\\d-\\d+$|^S-\\d-\\d-\\d+-\\d+-\\d+-\\d+-\\d+$')
    __whitespace_pattern = re.compile('^\\s*$', flags=re.UNICODE)
    __uninstall_search_list = [('url', 'str', ['URLInfoAbout', 'HelpLink', 'MoreInfoUrl', 'UrlUpdateInfo']), ('size', 'int', ['Size', 'EstimatedSize']), ('win_comments', 'str', ['Comments']), ('win_release_type', 'str', ['ReleaseType']), ('win_product_id', 'str', ['ProductID']), ('win_product_codes', 'str', ['ProductCodes']), ('win_package_refs', 'str', ['PackageRefs']), ('win_install_location', 'str', ['InstallLocation']), ('win_install_src_dir', 'str', ['InstallSource']), ('win_parent_pkg_uid', 'str', ['ParentKeyName']), ('win_parent_name', 'str', ['ParentDisplayName'])]
    __products_search_list = [('win_advertise_flags', 'int', ['AdvertiseFlags']), ('win_redeployment_flags', 'int', ['DeploymentFlags']), ('win_instance_type', 'int', ['InstanceType']), ('win_package_name', 'str', ['SourceList\\PackageName'])]

    def __init__(self, version_only=False, user_pkgs=False, pkg_obj=None):
        if False:
            return 10
        '\n        Point in time snapshot of the software and components installed on\n        a system.\n\n        Args:\n            version_only (bool): Provide list of versions installed instead of detail.\n            user_pkgs (bool): Include software/components installed with user space.\n            pkg_obj (object):\n                If None (default) return default package naming standard and use\n                default version capture methods (``DisplayVersion`` then\n                ``Version``, otherwise ``0.0.0.0``)\n        '
        self.__pkg_obj = pkg_obj
        self.__version_only = version_only
        self.__reg_software = {}
        self.__get_software_details(user_pkgs=user_pkgs)
        self.__pkg_cnt = len(self.__reg_software)
        self.__iter_list = None

    @property
    def data(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the raw data\n\n        Returns:\n            dict: contents of the dict are dependent on the parameters passed\n                when the class was initiated.\n        '
        return self.__reg_software

    @property
    def version_only(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if class initiated with ``version_only=True``\n\n        Returns:\n            bool: The value of ``version_only``\n        '
        return self.__version_only

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Returns total number of software/components installed.\n\n        Returns:\n            int: total number of software/components installed.\n        '
        return self.__pkg_cnt

    def __getitem__(self, pkg_id):
        if False:
            return 10
        '\n        Returns information on a package.\n\n        Args:\n            pkg_id (str): Package Id of the software/component\n\n        Returns:\n            dict or list: List if ``version_only`` is ``True`` otherwise dict\n        '
        if pkg_id in self.__reg_software:
            return self.__reg_software[pkg_id]
        else:
            raise KeyError(pkg_id)

    def __iter__(self):
        if False:
            print('Hello World!')
        '\n        Standard interation class initialisation over package information.\n        '
        if self.__iter_list is not None:
            raise RuntimeError('Can only perform one iter at a time')
        self.__iter_list = collections.deque(sorted(self.__reg_software.keys()))
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        '\n        Returns next Package Id.\n\n        Returns:\n            str: Package Id\n        '
        try:
            return self.__iter_list.popleft()
        except IndexError:
            self.__iter_list = None
            raise StopIteration

    def next(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns next Package Id.\n\n        Returns:\n            str: Package Id\n        '
        return self.__next__()

    def get(self, pkg_id, default_value=None):
        if False:
            while True:
                i = 10
        '\n        Returns information on a package.\n\n        Args:\n            pkg_id (str): Package Id of the software/component.\n            default_value: Value to return when the Package Id is not found.\n\n        Returns:\n            dict or list: List if ``version_only`` is ``True`` otherwise dict\n        '
        return self.__reg_software.get(pkg_id, default_value)

    @staticmethod
    def __oldest_to_latest_version(ver1, ver2):
        if False:
            return 10
        '\n        Used for sorting version numbers oldest to latest\n        '
        return 1 if Version(ver1) > Version(ver2) else -1

    @staticmethod
    def __latest_to_oldest_version(ver1, ver2):
        if False:
            print('Hello World!')
        '\n        Used for sorting version numbers, latest to oldest\n        '
        return 1 if Version(ver1) < Version(ver2) else -1

    def pkg_version_list(self, pkg_id):
        if False:
            i = 10
            return i + 15
        '\n        Returns information on a package.\n\n        Args:\n            pkg_id (str): Package Id of the software/component.\n\n        Returns:\n            list: List of version numbers installed.\n        '
        pkg_data = self.__reg_software.get(pkg_id, None)
        if not pkg_data:
            return []
        if isinstance(pkg_data, list):
            return pkg_data
        installed_versions = list(pkg_data.get('version').keys())
        return sorted(installed_versions, key=cmp_to_key(self.__oldest_to_latest_version))

    def pkg_version_latest(self, pkg_id):
        if False:
            print('Hello World!')
        '\n        Returns a package latest version installed out of all the versions\n        currently installed.\n\n        Args:\n            pkg_id (str): Package Id of the software/component.\n\n        Returns:\n            str: Latest/Newest version number installed.\n        '
        return self.pkg_version_list(pkg_id)[-1]

    def pkg_version_oldest(self, pkg_id):
        if False:
            return 10
        '\n        Returns a package oldest version installed out of all the versions\n        currently installed.\n\n        Args:\n            pkg_id (str): Package Id of the software/component.\n\n        Returns:\n            str: Oldest version number installed.\n        '
        return self.pkg_version_list(pkg_id)[0]

    @staticmethod
    def __sid_to_username(sid):
        if False:
            return 10
        '\n        Provided with a valid Windows Security Identifier (SID) and returns a Username\n\n        Args:\n            sid (str): Security Identifier (SID).\n\n        Returns:\n            str: Username in the format of username@realm or username@computer.\n        '
        if sid is None or sid == '':
            return ''
        try:
            sid_bin = win32security.GetBinarySid(sid)
        except pywintypes.error as exc:
            raise ValueError('pkg: Software owned by {} is not valid: [{}] {}'.format(sid, exc.winerror, exc.strerror))
        try:
            (name, domain, _account_type) = win32security.LookupAccountSid(None, sid_bin)
            user_name = '{}\\{}'.format(domain, name)
        except pywintypes.error as exc:
            if exc.winerror == winerror.ERROR_NONE_MAPPED:
                return sid
            else:
                raise ValueError("Failed looking up sid '{}' username: [{}] {}".format(sid, exc.winerror, exc.strerror))
        try:
            user_principal = win32security.TranslateName(user_name, win32api.NameSamCompatible, win32api.NameUserPrincipal)
        except pywintypes.error as exc:
            if exc.winerror in (winerror.ERROR_NO_SUCH_DOMAIN, winerror.ERROR_INVALID_DOMAINNAME, winerror.ERROR_NONE_MAPPED):
                return '{}@{}'.format(name.lower(), domain.lower())
            else:
                raise
        return user_principal

    def __software_to_pkg_id(self, publisher, name, is_component, is_32bit):
        if False:
            print('Hello World!')
        '\n        Determine the Package ID of a software/component using the\n        software/component ``publisher``, ``name``, whether its a software or a\n        component, and if its 32bit or 64bit archiecture.\n\n        Args:\n            publisher (str): Publisher of the software/component.\n            name (str): Name of the software.\n            is_component (bool): True if package is a component.\n            is_32bit (bool): True if the software/component is 32bit architecture.\n\n        Returns:\n            str: Package Id\n        '
        if publisher:
            pub_lc = publisher.replace(',', '').lower()
        else:
            pub_lc = 'NoValue'
        if name:
            name_lc = name.replace(',', '').lower()
        else:
            name_lc = 'NoValue'
        if is_component:
            soft_type = 'comp'
        else:
            soft_type = 'soft'
        if is_32bit:
            soft_type += '32'
        default_pkg_id = pub_lc + '\\\\' + name_lc + '\\\\' + soft_type
        if self.__pkg_obj and hasattr(self.__pkg_obj, 'to_pkg_id'):
            pkg_id = self.__pkg_obj.to_pkg_id(publisher, name, is_component, is_32bit)
            if pkg_id:
                return pkg_id
        return default_pkg_id

    def __version_capture_slp(self, pkg_id, version_binary, version_display, display_name):
        if False:
            while True:
                i = 10
        '\n        This returns the version and where the version string came from, based on instructions\n        under ``version_capture``, if ``version_capture`` is missing, it defaults to\n        value of display-version.\n\n        Args:\n            pkg_id (str): Publisher of the software/component.\n            version_binary (str): Name of the software.\n            version_display (str): True if package is a component.\n            display_name (str): True if the software/component is 32bit architecture.\n\n        Returns:\n            str: Package Id\n        '
        if self.__pkg_obj and hasattr(self.__pkg_obj, 'version_capture'):
            (version_str, src, version_user_str) = self.__pkg_obj.version_capture(pkg_id, version_binary, version_display, display_name)
            if src != 'use-default' and version_str and src:
                return (version_str, src, version_user_str)
            elif src != 'use-default':
                raise ValueError("version capture within object '{}' failed for pkg id: '{}' it returned '{}' '{}' '{}'".format(str(self.__pkg_obj), pkg_id, version_str, src, version_user_str))
        if version_display and re.match('\\d+', version_display, flags=re.IGNORECASE + re.UNICODE) is not None:
            version_str = version_display
            src = 'display-version'
        elif version_binary and re.match('\\d+', version_binary, flags=re.IGNORECASE + re.UNICODE) is not None:
            version_str = version_binary
            src = 'version-binary'
        else:
            src = 'none'
            version_str = '0.0.0.0.0'
        return (version_str, src, version_str)

    def __collect_software_info(self, sid, key_software, use_32bit):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update data with the next software found\n        '
        reg_soft_info = RegSoftwareInfo(key_software, sid, use_32bit)
        display_name = reg_soft_info.get_install_value('DisplayName', wanted_type='str')
        if display_name is None or self.__whitespace_pattern.match(display_name):
            return
        default_value = reg_soft_info.get_install_value('', wanted_type='str')
        release_type = reg_soft_info.get_install_value('ReleaseType', wanted_type='str')
        if re.match('^{.*\\}\\.KB\\d{6,}$', key_software, flags=re.IGNORECASE + re.UNICODE) is not None or (default_value and default_value.startswith(('KB', 'kb', 'Kb'))) or (release_type and release_type in ('Hotfix', 'Update Rollup', 'Security Update', 'ServicePack')):
            log.debug('skipping hotfix/update/service pack %s', key_software)
            return
        uninstall_no_remove = reg_soft_info.is_install_true('NoRemove')
        uninstall_string = reg_soft_info.get_install_value('UninstallString')
        uninstall_quiet_string = reg_soft_info.get_install_value('QuietUninstallString')
        uninstall_modify_path = reg_soft_info.get_install_value('ModifyPath')
        windows_installer = reg_soft_info.is_install_true('WindowsInstaller')
        system_component = reg_soft_info.is_install_true('SystemComponent')
        publisher = reg_soft_info.get_install_value('Publisher', wanted_type='str')
        if uninstall_string is None and uninstall_quiet_string is None and (uninstall_modify_path is None) and (not windows_installer):
            return
        if sid:
            username = self.__sid_to_username(sid)
        else:
            username = None
        pkg_id = self.__software_to_pkg_id(publisher, display_name, system_component, use_32bit)
        (version_binary, version_src) = reg_soft_info.version_binary
        version_display = reg_soft_info.get_install_value('DisplayVersion', wanted_type='str')
        (version_text, version_src, user_version) = self.__version_capture_slp(pkg_id, version_binary, version_display, display_name)
        if not user_version:
            user_version = version_text
        if username:
            dict_key = '{};{}'.format(username, pkg_id)
        else:
            dict_key = pkg_id
        if self.__version_only:
            if dict_key in self.__reg_software:
                if version_text not in self.__reg_software[dict_key]:
                    insert_point = 0
                    for ver_item in self.__reg_software[dict_key]:
                        if Version(version_text) <= Version(ver_item):
                            break
                        insert_point += 1
                    self.__reg_software[dict_key].insert(insert_point, version_text)
                else:
                    log.debug("Found extra entries for '%s' with same version '%s', skipping entry '%s'", dict_key, version_text, key_software)
            else:
                self.__reg_software[dict_key] = [version_text]
            return
        if dict_key in self.__reg_software:
            data = self.__reg_software[dict_key]
        else:
            data = self.__reg_software[dict_key] = OrderedDict()
        if sid:
            data.update({'arch': 'unknown'})
        else:
            arch_str = 'x86' if use_32bit else 'x64'
            if 'arch' in data:
                if data['arch'] != arch_str:
                    data['arch'] = 'many'
            else:
                data.update({'arch': arch_str})
        if publisher:
            if 'vendor' in data:
                if data['vendor'].lower() != publisher.lower():
                    data['vendor'] = 'many'
            else:
                data['vendor'] = publisher
        if 'win_system_component' in data:
            if data['win_system_component'] != system_component:
                data['win_system_component'] = None
        else:
            data['win_system_component'] = system_component
        data.update({'win_version_src': version_src})
        data.setdefault('version', {})
        if version_text in data['version']:
            if 'win_install_count' in data['version'][version_text]:
                data['version'][version_text]['win_install_count'] += 1
            else:
                data['version'][version_text]['win_install_count'] = 2
        else:
            data['version'][version_text] = OrderedDict()
        version_data = data['version'][version_text]
        version_data.update({'win_display_name': display_name})
        if uninstall_string:
            version_data.update({'win_uninstall_cmd': uninstall_string})
        if uninstall_quiet_string:
            version_data.update({'win_uninstall_quiet_cmd': uninstall_quiet_string})
        if uninstall_no_remove:
            version_data.update({'win_uninstall_no_remove': uninstall_no_remove})
        version_data.update({'win_product_code': key_software})
        if version_display:
            version_data.update({'win_version_display': version_display})
        if version_binary:
            version_data.update({'win_version_binary': version_binary})
        if user_version:
            version_data.update({'win_version_user': user_version})
        if windows_installer or (uninstall_string and re.search('MsiExec.exe\\s|MsiExec\\s', uninstall_string, flags=re.IGNORECASE + re.UNICODE)):
            version_data.update({'win_installer_type': 'winmsi'})
        elif re.match('InstallShield_', key_software, re.IGNORECASE) is not None or (uninstall_string and (re.search('InstallShield', uninstall_string, flags=re.IGNORECASE + re.UNICODE) is not None or re.search('isuninst\\.exe.*\\.isu', uninstall_string, flags=re.IGNORECASE + re.UNICODE) is not None)):
            version_data.update({'win_installer_type': 'installshield'})
        elif key_software.endswith('_is1') and reg_soft_info.get_install_value('Inno Setup: Setup Version', wanted_type='str'):
            version_data.update({'win_installer_type': 'inno'})
        elif uninstall_string and re.search('.*\\\\uninstall.exe|.*\\\\uninst.exe', uninstall_string, flags=re.IGNORECASE + re.UNICODE):
            version_data.update({'win_installer_type': 'nsis'})
        else:
            version_data.update({'win_installer_type': 'unknown'})
        language_number = reg_soft_info.get_install_value('Language')
        if isinstance(language_number, int) and language_number in locale.windows_locale:
            version_data.update({'win_language': locale.windows_locale[language_number]})
        package_code = reg_soft_info.package_code
        if package_code:
            version_data.update({'win_package_code': package_code})
        upgrade_code = reg_soft_info.upgrade_code
        if upgrade_code:
            version_data.update({'win_upgrade_code': upgrade_code})
        is_minor_upgrade = reg_soft_info.is_install_true('IsMinorUpgrade')
        if is_minor_upgrade:
            version_data.update({'win_is_minor_upgrade': is_minor_upgrade})
        install_time = reg_soft_info.install_time
        if install_time:
            version_data.update({'install_date': datetime.datetime.fromtimestamp(install_time).isoformat()})
            version_data.update({'install_date_time_t': int(install_time)})
        for (infokey, infotype, regfield_list) in self.__uninstall_search_list:
            for regfield in regfield_list:
                strvalue = reg_soft_info.get_install_value(regfield, wanted_type=infotype)
                if strvalue:
                    version_data.update({infokey: strvalue})
                    break
        for (infokey, infotype, regfield_list) in self.__products_search_list:
            for regfield in regfield_list:
                data = reg_soft_info.get_product_value(regfield, wanted_type=infotype)
                if data is not None:
                    version_data.update({infokey: data})
                    break
        patch_list = reg_soft_info.list_patches
        if patch_list:
            version_data.update({'win_patches': patch_list})

    def __get_software_details(self, user_pkgs):
        if False:
            i = 10
            return i + 15
        '\n        This searches the uninstall keys in the registry to find\n        a match in the sub keys, it will return a dict with the\n        display name as the key and the version as the value\n        .. sectionauthor:: Damon Atkins <https://github.com/damon-atkins>\n        .. versionadded:: 2016.11.0\n        '
        if platform.architecture()[0] == '32bit':
            if win32process.IsWow64Process():
                use_32bit_lookup = {True: 0, False: win32con.KEY_WOW64_64KEY}
                arch_list = [True, False]
            else:
                use_32bit_lookup = {True: 0, False: None}
                arch_list = [True]
        else:
            use_32bit_lookup = {True: win32con.KEY_WOW64_32KEY, False: 0}
            arch_list = [True, False]
        for arch_flag in arch_list:
            key_search = 'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall'
            log.debug('SYSTEM processing 32bit:%s', arch_flag)
            handle = win32api.RegOpenKeyEx(win32con.HKEY_LOCAL_MACHINE, key_search, 0, win32con.KEY_READ | use_32bit_lookup[arch_flag])
            (reg_key_all, _, _, _) = zip(*win32api.RegEnumKeyEx(handle))
            win32api.RegCloseKey(handle)
            for reg_key in reg_key_all:
                self.__collect_software_info(None, reg_key, arch_flag)
        if not user_pkgs:
            return
        log.debug('Processing user software... please wait')
        handle_sid = win32api.RegOpenKeyEx(win32con.HKEY_USERS, '', 0, win32con.KEY_READ)
        sid_all = []
        for index in range(win32api.RegQueryInfoKey(handle_sid)[0]):
            sid_all.append(win32api.RegEnumKey(handle_sid, index))
        for sid in sid_all:
            if self.__sid_pattern.match(sid) is not None:
                user_uninstall_path = '{}\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall'.format(sid)
                try:
                    handle = win32api.RegOpenKeyEx(handle_sid, user_uninstall_path, 0, win32con.KEY_READ)
                except pywintypes.error as exc:
                    if exc.winerror == winerror.ERROR_FILE_NOT_FOUND:
                        log.debug('Not Found %s', user_uninstall_path)
                        continue
                    else:
                        raise
                try:
                    (reg_key_all, _, _, _) = zip(*win32api.RegEnumKeyEx(handle))
                except ValueError:
                    log.debug('No Entries Found %s', user_uninstall_path)
                    reg_key_all = []
                win32api.RegCloseKey(handle)
                for reg_key in reg_key_all:
                    self.__collect_software_info(sid, reg_key, False)
        win32api.RegCloseKey(handle_sid)
        return

def __main():
    if False:
        while True:
            i = 10
    'This module can also be run directly for testing\n    Args:\n        detail|list : Provide ``detail`` or version ``list``.\n        system|system+user: System installed and System and User installs.\n    '
    if len(sys.argv) < 3:
        sys.stderr.write('usage: {} <detail|list> <system|system+user>\n'.format(sys.argv[0]))
        sys.exit(64)
    user_pkgs = False
    version_only = False
    if str(sys.argv[1]) == 'list':
        version_only = True
    if str(sys.argv[2]) == 'system+user':
        user_pkgs = True
    import timeit
    import salt.utils.json

    def run():
        if False:
            for i in range(10):
                print('nop')
        '\n        Main run code, when this module is run directly\n        '
        pkg_list = WinSoftware(user_pkgs=user_pkgs, version_only=version_only)
        print(salt.utils.json.dumps(pkg_list.data, sort_keys=True, indent=4))
        print('Total: {}'.format(len(pkg_list)))
    print('Time Taken: {}'.format(timeit.timeit(run, number=1)))
if __name__ == '__main__':
    __main()