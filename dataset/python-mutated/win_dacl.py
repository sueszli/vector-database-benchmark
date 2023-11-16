"""
============
Windows DACL
============

This salt utility contains objects and functions for setting permissions to
objects in Windows. You can use the built in functions or access the objects
directly to create your own custom functionality. There are two objects, Flags
and Dacl.

If you need access only to flags, use the Flags object.

.. code-block:: python

    import salt.utils.win_dacl
    flags = salt.utils.win_dacl.Flags()
    flag_full_control = flags.ace_perms['file']['basic']['full_control']

The Dacl object inherits Flags. To use the Dacl object:

.. code-block:: python

    import salt.utils.win_dacl
    dacl = salt.utils.win_dacl.Dacl(obj_type='file')
    dacl.add_ace('Administrators', 'grant', 'full_control')
    dacl.save('C:\\temp')

Object types are used by setting the `obj_type` parameter to a valid Windows
object. Valid object types are as follows:

- file
- service
- printer
- registry
- registry32 (for WOW64)
- share

Each object type has its own set up permissions and 'applies to' properties as
follows. At this time only basic permissions are used for setting. Advanced
permissions are listed for displaying the permissions of an object that don't
match the basic permissions, ie. Special permissions. These should match the
permissions you see when you look at the security for an object.

**Basic Permissions**

    ================  ====  ========  =====  =======  =======
    Permissions       File  Registry  Share  Printer  Service
    ================  ====  ========  =====  =======  =======
    full_control      X     X         X               X
    modify            X
    read_execute      X
    read              X     X         X               X
    write             X     X                         X
    read_write                                        X
    change                            X
    print                                    X
    manage_printer                           X
    manage_documents                         X
    ================  ====  ========  =====  =======  =======

**Advanced Permissions**

    =======================  ====  ========  =======  =======
    Permissions              File  Registry  Printer  Service
    =======================  ====  ========  =======  =======
    *** folder permissions
    list_folder              X
    create_files             X
    create_folders           X
    traverse_folder          X
    delete_subfolders_files  X

    *** file permissions
    read_data                X
    write_data               X
    append_data              X
    execute_file             X

    *** common permissions
    read_ea                  X
    write_ea                 X
    read_attributes          X
    write_attributes         X
    delete                   X     X
    read_permissions         X               X        X
    change_permissions       X               X        X
    take_ownership           X               X
    query_value                    X
    set_value                      X
    create_subkey                  X
    enum_subkeys                   X
    notify                         X
    create_link                    X
    read_control                   X
    write_dac                      X
    write_owner                    X
    manage_printer                           X
    print                                    X
    query_config                                      X
    change_config                                     X
    query_status                                      X
    enum_dependents                                   X
    start                                             X
    stop                                              X
    pause_resume                                      X
    interrogate                                       X
    user_defined                                      X
    change_owner                                      X
    =======================  ====  ========  =======  =======

Only the registry and file object types have 'applies to' properties. These
should match what you see when you look at the properties for an object.

    **File types:**

        - this_folder_only: Applies only to this object
        - this_folder_subfolders_files (default): Applies to this object
          and all sub containers and objects
        - this_folder_subfolders: Applies to this object and all sub
          containers, no files
        - this_folder_files: Applies to this object and all file
          objects, no containers
        - subfolders_files: Applies to all containers and objects
          beneath this object
        - subfolders_only: Applies to all containers beneath this object
        - files_only: Applies to all file objects beneath this object

    .. NOTE::

        'applies to' properties can only be modified on directories. Files
        will always be ``this_folder_only``.

    **Registry types:**

        - this_key_only: Applies only to this key
        - this_key_subkeys: Applies to this key and all subkeys
        - subkeys_only: Applies to all subkeys beneath this object

"""
import logging
import salt.utils.platform
import salt.utils.win_functions
from salt.exceptions import CommandExecutionError, SaltInvocationError
HAS_WIN32 = False
try:
    import pywintypes
    import win32api
    import win32con
    import win32security
    HAS_WIN32 = True
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'dacl'

def __virtual__():
    if False:
        return 10
    '\n    Only load if Win32 Libraries are installed\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'win_dacl: Requires Windows')
    if not HAS_WIN32:
        return (False, 'win_dacl: Requires pywin32')
    return __virtualname__

def flags(instantiated=True):
    if False:
        while True:
            i = 10
    '\n    Helper function for instantiating a Flags object\n\n    Args:\n\n        instantiated (bool):\n            True to return an instantiated object, False to return the object\n            definition. Use False if inherited by another class. Default is\n            True.\n\n    Returns:\n        object: An instance of the Flags object or its definition\n    '
    if not HAS_WIN32:
        return

    class Flags:
        """
        Object containing all the flags for dealing with Windows permissions
        """
        ace_perms = {'file': {'basic': {2032127: 'Full control', 1245631: 'Modify', 1180095: 'Read & execute with write', 1179817: 'Read & execute', 1179785: 'Read', 1048854: 'Write', 'full_control': 2032127, 'modify': 1245631, 'read_execute': 1179817, 'read': 1179785, 'write': 1048854}, 'advanced': {1: 'List folder / read data', 2: 'Create files / write data', 4: 'Create folders / append data', 8: 'Read extended attributes', 16: 'Write extended attributes', 32: 'Traverse folder / execute file', 64: 'Delete subfolders and files', 128: 'Read attributes', 256: 'Write attributes', 65536: 'Delete', 131072: 'Read permissions', 262144: 'Change permissions', 524288: 'Take ownership', 'list_folder': 1, 'create_files': 2, 'create_folders': 4, 'traverse_folder': 32, 'delete_subfolders_files': 64, 'read_data': 1, 'write_data': 2, 'append_data': 4, 'execute_file': 32, 'read_ea': 8, 'write_ea': 16, 'read_attributes': 128, 'write_attributes': 256, 'delete': 65536, 'read_permissions': 131072, 'change_permissions': 262144, 'take_ownership': 524288}}, 'registry': {'basic': {983103: 'Full Control', 131097: 'Read', 131078: 'Write', 268435456: 'Full Control', 536870912: 'Execute', 1073741824: 'Write', 18446744071562067968: 'Read', 'full_control': 983103, 'read': 131097, 'write': 131078}, 'advanced': {1: 'Query Value', 2: 'Set Value', 4: 'Create Subkey', 8: 'Enumerate Subkeys', 16: 'Notify', 32: 'Create Link', 65536: 'Delete', 131072: 'Read Control', 262144: 'Write DAC', 524288: 'Write Owner', 'query_value': 1, 'set_value': 2, 'create_subkey': 4, 'enum_subkeys': 8, 'notify': 16, 'create_link': 32, 'delete': 65536, 'read_control': 131072, 'write_dac': 262144, 'write_owner': 524288}}, 'share': {'basic': {2032127: 'Full control', 1245631: 'Change', 1179817: 'Read', 'full_control': 2032127, 'change': 1245631, 'read': 1179817}, 'advanced': {}}, 'printer': {'basic': {131080: 'Print', 983052: 'Manage this printer', 983088: 'Manage documents', 'print': 131080, 'manage_printer': 983052, 'manage_documents': 983088}, 'advanced': {65540: 'Manage this printer', 8: 'Print', 131072: 'Read permissions', 262144: 'Change permissions', 524288: 'Take ownership', 'manage_printer': 65540, 'print': 8, 'read_permissions': 131072, 'change_permissions': 262144, 'take_ownership': 524288}}, 'service': {'basic': {983551: 'Full Control', 131215: 'Read & Write', 131469: 'Read', 131074: 'Write', 'full_control': 983551, 'read_write': 131215, 'read': 131469, 'write': 131074}, 'advanced': {1: 'Query Config', 2: 'Change Config', 4: 'Query Status', 8: 'Enumerate Dependents', 16: 'Start', 32: 'Stop', 64: 'Pause/Resume', 128: 'Interrogate', 256: 'User-Defined Control', 131072: 'Read Permissions', 262144: 'Change Permissions', 524288: 'Change Owner', 'query_config': 1, 'change_config': 2, 'query_status': 4, 'enum_dependents': 8, 'start': 16, 'stop': 32, 'pause_resume': 64, 'interrogate': 128, 'user_defined': 256, 'read_permissions': 131072, 'change_permissions': 262144, 'change_owner': 524288}}}
        ace_prop = {'file': {0: 'This folder only', 1: 'This folder and files', 2: 'This folder and subfolders', 3: 'This folder, subfolders and files', 9: 'Files only', 10: 'Subfolders only', 11: 'Subfolders and files only', 'this_folder_only': 0, 'this_folder_files': 1, 'this_folder_subfolders': 2, 'this_folder_subfolders_files': 3, 'files_only': 9, 'subfolders_only': 10, 'subfolders_files': 11}, 'registry': {0: 'This key only', 2: 'This key and subkeys', 10: 'Subkeys only', 'this_key_only': 0, 'this_key_subkeys': 2, 'subkeys_only': 10}, 'registry32': {0: 'This key only', 2: 'This key and subkeys', 10: 'Subkeys only', 'this_key_only': 0, 'this_key_subkeys': 2, 'subkeys_only': 10}}
        ace_type = {'grant': win32security.ACCESS_ALLOWED_ACE_TYPE, 'deny': win32security.ACCESS_DENIED_ACE_TYPE, win32security.ACCESS_ALLOWED_ACE_TYPE: 'grant', win32security.ACCESS_DENIED_ACE_TYPE: 'deny'}
        element = {'dacl': win32security.DACL_SECURITY_INFORMATION, 'group': win32security.GROUP_SECURITY_INFORMATION, 'owner': win32security.OWNER_SECURITY_INFORMATION}
        inheritance = {'protected': win32security.PROTECTED_DACL_SECURITY_INFORMATION, 'unprotected': win32security.UNPROTECTED_DACL_SECURITY_INFORMATION}
        obj_type = {'file': win32security.SE_FILE_OBJECT, 'service': win32security.SE_SERVICE, 'printer': win32security.SE_PRINTER, 'registry': win32security.SE_REGISTRY_KEY, 'registry32': win32security.SE_REGISTRY_WOW64_32KEY, 'share': win32security.SE_LMSHARE}
    return Flags() if instantiated else Flags

def dacl(obj_name=None, obj_type='file'):
    if False:
        print('Hello World!')
    "\n    Helper function for instantiating a Dacl class.\n\n    Args:\n\n        obj_name (str):\n            The full path to the object. If None, a blank DACL will be created.\n            Default is None.\n\n        obj_type (str):\n            The type of object. Default is 'File'\n\n    Returns:\n        object: An instantiated Dacl object\n    "
    if not HAS_WIN32:
        return

    class Dacl(flags(False)):
        """
        DACL Object
        """

        def __init__(self, obj_name=None, obj_type='file'):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Either load the DACL from the passed object or create an empty DACL.\n            If `obj_name` is not passed, an empty DACL is created.\n\n            Args:\n\n                obj_name (str):\n                    The full path to the object. If None, a blank DACL will be\n                    created\n\n                obj_type (Optional[str]):\n                    The type of object.\n\n            Returns:\n                obj: A DACL object\n\n            Usage:\n\n            .. code-block:: python\n\n                # Create an Empty DACL\n                dacl = Dacl(obj_type=obj_type)\n\n                # Load the DACL of the named object\n                dacl = Dacl(obj_name, obj_type)\n            '
            if obj_type.lower() not in self.obj_type:
                raise SaltInvocationError(f'Invalid "obj_type" passed: {obj_type}')
            self.dacl_type = obj_type.lower()
            if obj_name is None:
                self.dacl = win32security.ACL()
            else:
                if 'registry' in self.dacl_type:
                    obj_name = self.get_reg_name(obj_name)
                try:
                    sd = win32security.GetNamedSecurityInfo(obj_name, self.obj_type[self.dacl_type], self.element['dacl'])
                except pywintypes.error as exc:
                    if 'The system cannot find' in exc.strerror:
                        msg = f'System cannot find {obj_name}'
                        log.exception(msg)
                        raise CommandExecutionError(msg)
                    raise
                self.dacl = sd.GetSecurityDescriptorDacl()
                if self.dacl is None:
                    self.dacl = win32security.ACL()

        def get_reg_name(self, obj_name):
            if False:
                print('Hello World!')
            "\n            Take the obj_name and convert the hive to a valid registry hive.\n\n            Args:\n\n                obj_name (str):\n                    The full path to the registry key including the hive, eg:\n                    ``HKLM\\SOFTWARE\\salt``. Valid options for the hive are:\n\n                    - HKEY_LOCAL_MACHINE\n                    - MACHINE\n                    - HKLM\n                    - HKEY_USERS\n                    - USERS\n                    - HKU\n                    - HKEY_CURRENT_USER\n                    - CURRENT_USER\n                    - HKCU\n                    - HKEY_CLASSES_ROOT\n                    - CLASSES_ROOT\n                    - HKCR\n\n            Returns:\n                str:\n                    The full path to the registry key in the format expected by\n                    the Windows API\n\n            Usage:\n\n            .. code-block:: python\n\n                import salt.utils.win_dacl\n                dacl = salt.utils.win_dacl.Dacl()\n                valid_key = dacl.get_reg_name('HKLM\\SOFTWARE\\salt')\n\n                # Returns: MACHINE\\SOFTWARE\\salt\n            "
            hives = {'HKEY_LOCAL_MACHINE': 'MACHINE', 'MACHINE': 'MACHINE', 'HKLM': 'MACHINE', 'HKEY_USERS': 'USERS', 'USERS': 'USERS', 'HKU': 'USERS', 'HKEY_CURRENT_USER': 'CURRENT_USER', 'CURRENT_USER': 'CURRENT_USER', 'HKCU': 'CURRENT_USER', 'HKEY_CLASSES_ROOT': 'CLASSES_ROOT', 'CLASSES_ROOT': 'CLASSES_ROOT', 'HKCR': 'CLASSES_ROOT'}
            reg = obj_name.split('\\')
            passed_hive = reg.pop(0)
            try:
                valid_hive = hives[passed_hive.upper()]
            except KeyError:
                log.exception('Invalid Registry Hive: %s', passed_hive)
                raise CommandExecutionError(f'Invalid Registry Hive: {passed_hive}')
            reg.insert(0, valid_hive)
            return '\\\\'.join(reg)

        def add_ace(self, principal, access_mode, permissions, applies_to):
            if False:
                i = 10
                return i + 15
            '\n            Add an ACE to the DACL\n\n            Args:\n\n                principal (str):\n                    The sid of the user/group to for the ACE\n\n                access_mode (str):\n                    Determines the type of ACE to add. Must be either ``grant``\n                    or ``deny``.\n\n                permissions (str, list):\n                    The type of permissions to grant/deny the user. Can be one\n                    of the basic permissions, or a list of advanced permissions.\n\n                applies_to (str):\n                    The objects to which these permissions will apply. Not all\n                    these options apply to all object types.\n\n            Returns:\n                bool: True if successful, otherwise False\n\n            Usage:\n\n            .. code-block:: python\n\n                dacl = Dacl(obj_type=obj_type)\n                dacl.add_ace(sid, access_mode, permission, applies_to)\n                dacl.save(obj_name, protected)\n            '
            sid = get_sid(principal)
            if applies_to not in self.ace_prop[self.dacl_type]:
                raise SaltInvocationError(f"Invalid 'applies_to' for type {self.dacl_type}")
            if self.dacl is None:
                raise SaltInvocationError('You must load the DACL before adding an ACE')
            perm_flag = 0
            if isinstance(permissions, str):
                try:
                    perm_flag = self.ace_perms[self.dacl_type]['basic'][permissions]
                except KeyError as exc:
                    msg = f'Invalid permission specified: {permissions}'
                    log.exception(msg)
                    raise CommandExecutionError(msg, exc)
            else:
                for perm in permissions:
                    try:
                        perm_flag |= self.ace_perms[self.dacl_type]['advanced'][perm]
                    except KeyError as exc:
                        msg = f'Invalid permission specified: {perm}'
                        log.exception(msg)
                        raise CommandExecutionError(msg, exc)
            if access_mode.lower() not in ['grant', 'deny']:
                raise SaltInvocationError(f'Invalid Access Mode: {access_mode}')
            try:
                if access_mode.lower() == 'grant':
                    self.dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, self.ace_prop.get(self.dacl_type, {}).get(applies_to), perm_flag, sid)
                elif access_mode.lower() == 'deny':
                    self.dacl.AddAccessDeniedAceEx(win32security.ACL_REVISION_DS, self.ace_prop.get(self.dacl_type, {}).get(applies_to), perm_flag, sid)
                else:
                    log.exception('Invalid access mode: %s', access_mode)
                    raise SaltInvocationError(f'Invalid access mode: {access_mode}')
            except Exception as exc:
                return (False, f'Error: {exc}')
            return True

        def order_acl(self):
            if False:
                while True:
                    i = 10
            '\n            Put the ACEs in the ACL in the proper order. This is necessary\n            because the add_ace function puts ACEs at the end of the list\n            without regard for order. This will cause the following Windows\n            Security dialog to appear when viewing the security for the object:\n\n            ``The permissions on Directory are incorrectly ordered, which may\n            cause some entries to be ineffective.``\n\n            .. note:: Run this function after adding all your ACEs.\n\n            Proper Orders is as follows:\n\n                1. Implicit Deny\n                2. Inherited Deny\n                3. Implicit Deny Object\n                4. Inherited Deny Object\n                5. Implicit Allow\n                6. Inherited Allow\n                7. Implicit Allow Object\n                8. Inherited Allow Object\n\n            Usage:\n\n            .. code-block:: python\n\n                dacl = Dacl(obj_type=obj_type)\n                dacl.add_ace(sid, access_mode, applies_to, permission)\n                dacl.order_acl()\n                dacl.save(obj_name, protected)\n            '
            new_dacl = Dacl()
            deny_dacl = Dacl()
            deny_obj_dacl = Dacl()
            allow_dacl = Dacl()
            allow_obj_dacl = Dacl()
            for i in range(0, self.dacl.GetAceCount()):
                ace = self.dacl.GetAce(i)
                if ace[0][1] & win32security.INHERITED_ACE == 0:
                    if ace[0][0] == win32security.ACCESS_DENIED_ACE_TYPE:
                        deny_dacl.dacl.AddAccessDeniedAceEx(win32security.ACL_REVISION_DS, ace[0][1], ace[1], ace[2])
                    elif ace[0][0] == win32security.ACCESS_DENIED_OBJECT_ACE_TYPE:
                        deny_obj_dacl.dacl.AddAccessDeniedAceEx(win32security.ACL_REVISION_DS, ace[0][1], ace[1], ace[2])
                    elif ace[0][0] == win32security.ACCESS_ALLOWED_ACE_TYPE:
                        allow_dacl.dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, ace[0][1], ace[1], ace[2])
                    elif ace[0][0] == win32security.ACCESS_ALLOWED_OBJECT_ACE_TYPE:
                        allow_obj_dacl.dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, ace[0][1], ace[1], ace[2])
            for i in range(0, self.dacl.GetAceCount()):
                ace = self.dacl.GetAce(i)
                if ace[0][1] & win32security.INHERITED_ACE == win32security.INHERITED_ACE:
                    ace_prop = ace[0][1] ^ win32security.INHERITED_ACE
                    if ace[0][0] == win32security.ACCESS_DENIED_ACE_TYPE:
                        deny_dacl.dacl.AddAccessDeniedAceEx(win32security.ACL_REVISION_DS, ace_prop, ace[1], ace[2])
                    elif ace[0][0] == win32security.ACCESS_DENIED_OBJECT_ACE_TYPE:
                        deny_obj_dacl.dacl.AddAccessDeniedAceEx(win32security.ACL_REVISION_DS, ace_prop, ace[1], ace[2])
                    elif ace[0][0] == win32security.ACCESS_ALLOWED_ACE_TYPE:
                        allow_dacl.dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, ace_prop, ace[1], ace[2])
                    elif ace[0][0] == win32security.ACCESS_ALLOWED_OBJECT_ACE_TYPE:
                        allow_obj_dacl.dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, ace_prop, ace[1], ace[2])
            for i in range(0, deny_dacl.dacl.GetAceCount()):
                ace = deny_dacl.dacl.GetAce(i)
                new_dacl.dacl.AddAccessDeniedAceEx(win32security.ACL_REVISION_DS, ace[0][1], ace[1], ace[2])
            for i in range(0, deny_obj_dacl.dacl.GetAceCount()):
                ace = deny_obj_dacl.dacl.GetAce(i)
                new_dacl.dacl.AddAccessDeniedAceEx(win32security.ACL_REVISION_DS, ace[0][1] ^ win32security.INHERITED_ACE, ace[1], ace[2])
            for i in range(0, allow_dacl.dacl.GetAceCount()):
                ace = allow_dacl.dacl.GetAce(i)
                new_dacl.dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, ace[0][1], ace[1], ace[2])
            for i in range(0, allow_obj_dacl.dacl.GetAceCount()):
                ace = allow_obj_dacl.dacl.GetAce(i)
                new_dacl.dacl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, ace[0][1] ^ win32security.INHERITED_ACE, ace[1], ace[2])
            self.dacl = new_dacl.dacl

        def get_ace(self, principal):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Get the ACE for a specific principal.\n\n            Args:\n\n                principal (str):\n                    The name of the user or group for which to get the ace. Can\n                    also be a SID.\n\n            Returns:\n                dict: A dictionary containing the ACEs found for the principal\n\n            Usage:\n\n            .. code-block:: python\n\n                dacl = Dacl(obj_type=obj_type)\n                dacl.get_ace()\n            '
            principal = get_name(principal)
            aces = self.list_aces()
            ret = {}
            for inheritance in aces:
                if principal in aces[inheritance]:
                    ret[inheritance] = {principal: aces[inheritance][principal]}
            return ret

        def list_aces(self):
            if False:
                i = 10
                return i + 15
            "\n            List all Entries in the dacl.\n\n            Returns:\n                dict: A dictionary containing the ACEs for the object\n\n            Usage:\n\n            .. code-block:: python\n\n                dacl = Dacl('C:\\Temp')\n                dacl.list_aces()\n            "
            ret = {'Inherited': {}, 'Not Inherited': {}}
            for i in range(0, self.dacl.GetAceCount()):
                ace = self.dacl.GetAce(i)
                (user, a_type, a_prop, a_perms, inheritance) = self._ace_to_dict(ace)
                if user in ret[inheritance]:
                    ret[inheritance][user][a_type] = {'applies to': a_prop, 'permissions': a_perms}
                else:
                    ret[inheritance][user] = {a_type: {'applies to': a_prop, 'permissions': a_perms}}
            return ret

        def _ace_to_dict(self, ace):
            if False:
                i = 10
                return i + 15
            '\n            Helper function for creating the ACE return dictionary\n            '
            sid = win32security.ConvertSidToStringSid(ace[2])
            try:
                principal = get_name(sid)
            except CommandExecutionError:
                principal = sid
            ace_type = self.ace_type[ace[0][0]]
            inherited = ace[0][1] & win32security.INHERITED_ACE == 16
            container_only = ace[0][1] & win32security.NO_PROPAGATE_INHERIT_ACE == 4
            ace_prop = 'NA'
            if self.dacl_type in ['file', 'registry', 'registry32']:
                ace_prop = ace[0][1]
                if inherited:
                    ace_prop = ace[0][1] ^ win32security.INHERITED_ACE
                if container_only:
                    ace_prop = ace[0][1] ^ win32security.NO_PROPAGATE_INHERIT_ACE
                try:
                    ace_prop = self.ace_prop[self.dacl_type][ace_prop]
                except KeyError:
                    ace_prop = 'Unknown propagation'
            obj_type = 'registry' if self.dacl_type == 'registry32' else self.dacl_type
            ace_perms = self.ace_perms[obj_type]['basic'].get(ace[1], [])
            if not ace_perms:
                ace_perms = []
                for perm in self.ace_perms[obj_type]['advanced']:
                    if isinstance(perm, str):
                        continue
                    if ace[1] & perm == perm:
                        ace_perms.append(self.ace_perms[obj_type]['advanced'][perm])
                ace_perms.sort()
            if not ace_perms:
                ace_perms = [f'Undefined Permission: {ace[1]}']
            return (principal, ace_type, ace_prop, ace_perms, 'Inherited' if inherited else 'Not Inherited')

        def rm_ace(self, principal, ace_type='all'):
            if False:
                print('Hello World!')
            "\n            Remove a specific ACE from the DACL.\n\n            Args:\n\n                principal (str):\n                    The user whose ACE to remove. Can be the user name or a SID.\n\n                ace_type (str):\n                    The type of ACE to remove. If not specified, all ACEs will\n                    be removed. Default is 'all'. Valid options are:\n\n                    - 'grant'\n                    - 'deny'\n                    - 'all'\n\n            Returns:\n                list: List of removed aces\n\n            Usage:\n\n            .. code-block:: python\n\n                dacl = Dacl(obj_name='C:\\temp', obj_type='file')\n                dacl.rm_ace('Users')\n                dacl.save(obj_name='C:\\temp')\n            "
            sid = get_sid(principal)
            ace_type = ace_type.lower()
            offset = 0
            ret = []
            for i in range(0, self.dacl.GetAceCount()):
                ace = self.dacl.GetAce(i - offset)
                inherited = ace[0][1] & win32security.INHERITED_ACE == 16
                if ace[2] == sid and (not inherited):
                    if ace_type == 'all' or self.ace_type[ace[0][0]] == ace_type:
                        self.dacl.DeleteAce(i - offset)
                        ret.append(self._ace_to_dict(ace))
                        offset += 1
            if not ret:
                ret = [f'ACE not found for {principal}']
            return ret

        def rm_all_aces(self, ace_type='all'):
            if False:
                i = 10
                return i + 15
            "\n            Removes all ACEs from the DACL.\n\n            Args:\n\n                ace_type (str):\n                    The type of ACE to remove. If not specified, all ACEs will\n                    be removed. Default is 'all'. Valid options are:\n\n                    - 'grant'\n                    - 'deny'\n                    - 'all'\n\n            Returns:\n                list: List of removed aces\n\n            Usage:\n\n            .. code-block:: python\n\n                dacl = Dacl(obj_name='C:\\temp', obj_type='file')\n                dacl.rm_all_aces()\n                dacl.save(obj_name='C:\\temp')\n            "
            offset = 0
            ret = []
            ace_type = ace_type.lower()
            for i in range(0, self.dacl.GetAceCount()):
                ace = self.dacl.GetAce(i - offset)
                inherited = ace[0][1] & win32security.INHERITED_ACE == 16
                if not inherited:
                    if ace_type == 'all' or self.ace_type[ace[0][0]] == ace_type:
                        self.dacl.DeleteAce(i - offset)
                        ret.append(self._ace_to_dict(ace))
                        offset += 1
            return ret

        def save(self, obj_name, protected=None):
            if False:
                return 10
            "\n            Save the DACL\n\n            Args:\n\n                obj_name (str):\n                    The object for which to set permissions. This can be the\n                    path to a file or folder, a registry key, printer, etc. For\n                    more information about how to format the name see:\n\n                    https://msdn.microsoft.com/en-us/library/windows/desktop/aa379593(v=vs.85).aspx\n\n                protected (Optional[bool]):\n                    True will disable inheritance for the object. False will\n                    enable inheritance. None will make no change. Default is\n                    ``None``.\n\n            Returns:\n                bool: True if successful, Otherwise raises an exception\n\n            Usage:\n\n            .. code-block:: python\n\n                dacl = Dacl(obj_type='file')\n                dacl.save('C:\\Temp', True)\n            "
            sec_info = self.element['dacl']
            if protected is not None:
                if protected:
                    sec_info = sec_info | self.inheritance['protected']
                else:
                    sec_info = sec_info | self.inheritance['unprotected']
            if self.dacl_type in ['registry', 'registry32']:
                obj_name = self.get_reg_name(obj_name)
            try:
                win32security.SetNamedSecurityInfo(obj_name, self.obj_type[self.dacl_type], sec_info, None, None, self.dacl, None)
            except pywintypes.error as exc:
                raise CommandExecutionError(f'Failed to set permissions: {obj_name}', exc.strerror)
            return True
    return Dacl(obj_name, obj_type)

def get_sid(principal):
    if False:
        return 10
    "\n    Converts a username to a sid, or verifies a sid. Required for working with\n    the DACL.\n\n    Args:\n\n        principal(str):\n            The principal to lookup the sid. Can be a sid or a username.\n\n    Returns:\n        PySID Object: A sid\n\n    Usage:\n\n    .. code-block:: python\n\n        # Get a user's sid\n        salt.utils.win_dacl.get_sid('jsnuffy')\n\n        # Verify that the sid is valid\n        salt.utils.win_dacl.get_sid('S-1-5-32-544')\n    "
    if principal is None:
        principal = 'NULL SID'
    try:
        sid = salt.utils.win_functions.get_sid_from_name(principal)
    except CommandExecutionError:
        sid = principal
    try:
        sid = win32security.ConvertStringSidToSid(sid)
    except pywintypes.error:
        log.exception('Invalid user/group or sid: %s', principal)
        raise CommandExecutionError(f'Invalid user/group or sid: {principal}')
    except TypeError:
        raise CommandExecutionError
    return sid

def get_sid_string(principal):
    if False:
        while True:
            i = 10
    "\n    Converts a PySID object to a string SID.\n\n    Args:\n\n        principal(str):\n            The principal to lookup the sid. Must be a PySID object.\n\n    Returns:\n        str: A string sid\n\n    Usage:\n\n    .. code-block:: python\n\n        # Get a PySID object\n        py_sid = salt.utils.win_dacl.get_sid('jsnuffy')\n\n        # Get the string version of the SID\n        salt.utils.win_dacl.get_sid_string(py_sid)\n    "
    if principal is None:
        principal = 'NULL SID'
    try:
        return win32security.ConvertSidToStringSid(principal)
    except TypeError:
        principal = get_sid(principal)
    try:
        return win32security.ConvertSidToStringSid(principal)
    except pywintypes.error:
        log.exception('Invalid principal %s', principal)
        raise CommandExecutionError(f'Invalid principal {principal}')

def get_name(principal):
    if False:
        while True:
            i = 10
    "\n    Gets the name from the specified principal.\n\n    Args:\n\n        principal (str):\n            Find the Normalized name based on this. Can be a PySID object, a SID\n            string, or a user name in any capitalization.\n\n            .. note::\n                Searching based on the user name can be slow on hosts connected\n                to large Active Directory domains.\n\n    Returns:\n        str: The name that corresponds to the passed principal\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.get_name('S-1-5-32-544')\n        salt.utils.win_dacl.get_name('adminisTrators')\n    "
    if isinstance(principal, pywintypes.SIDType):
        sid_obj = principal
    else:
        if principal is None:
            principal = 'S-1-0-0'
        try:
            sid_obj = win32security.ConvertStringSidToSid(principal)
        except pywintypes.error:
            try:
                sid_obj = win32security.LookupAccountName(None, principal)[0]
            except pywintypes.error:
                sid_obj = principal
    str_sid = get_sid_string(sid_obj)
    try:
        name = win32security.LookupAccountSid(None, sid_obj)[0]
        if str_sid.startswith('S-1-5-80'):
            name = f'NT Service\\{name}'
        return name
    except (pywintypes.error, TypeError) as exc:
        if not str_sid.startswith('S-1-15-3'):
            message = f'Error resolving "{principal}"'
            if type(exc) == pywintypes.error:
                win_error = win32api.FormatMessage(exc.winerror).rstrip('\n')
                message = f'{message}: {win_error}'
            log.exception(message)
            raise CommandExecutionError(message, exc)

def get_owner(obj_name, obj_type='file'):
    if False:
        i = 10
        return i + 15
    "\n    Gets the owner of the passed object\n\n    Args:\n\n        obj_name (str):\n            The path for which to obtain owner information. The format of this\n            parameter is different depending on the ``obj_type``\n\n        obj_type (str):\n            The type of object to query. This value changes the format of the\n            ``obj_name`` parameter as follows:\n\n            - file: indicates a file or directory\n                - a relative path, such as ``FileName.txt`` or ``..\\FileName``\n                - an absolute path, such as ``C:\\DirName\\FileName.txt``\n                - A UNC name, such as ``\\\\ServerName\\ShareName\\FileName.txt``\n            - service: indicates the name of a Windows service\n            - printer: indicates the name of a printer\n            - registry: indicates a registry key\n                - Uses the following literal strings to denote the hive:\n                    - HKEY_LOCAL_MACHINE\n                    - MACHINE\n                    - HKLM\n                    - HKEY_USERS\n                    - USERS\n                    - HKU\n                    - HKEY_CURRENT_USER\n                    - CURRENT_USER\n                    - HKCU\n                    - HKEY_CLASSES_ROOT\n                    - CLASSES_ROOT\n                    - HKCR\n                - Should be in the format of ``HIVE\\Path\\To\\Key``. For example,\n                    ``HKLM\\SOFTWARE\\Windows``\n            - registry32: indicates a registry key under WOW64. Formatting is\n                the same as it is for ``registry``\n            - share: indicates a network share\n\n    Returns:\n        str: The owner (group or user)\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.get_owner('c:\\\\file')\n    "
    try:
        obj_type_flag = flags().obj_type[obj_type.lower()]
    except KeyError:
        raise SaltInvocationError(f'Invalid "obj_type" passed: {obj_type}')
    if obj_type in ['registry', 'registry32']:
        obj_name = dacl().get_reg_name(obj_name)
    try:
        security_descriptor = win32security.GetNamedSecurityInfo(obj_name, obj_type_flag, win32security.OWNER_SECURITY_INFORMATION)
        owner_sid = security_descriptor.GetSecurityDescriptorOwner()
    except MemoryError:
        owner_sid = 'S-1-0-0'
    except pywintypes.error as exc:
        if exc.winerror == 1 or exc.winerror == 50:
            owner_sid = 'S-1-0-0'
        else:
            log.exception('Failed to get the owner: %s', obj_name)
            raise CommandExecutionError(f'Failed to get owner: {obj_name}', exc.strerror)
    return get_name(owner_sid)

def get_primary_group(obj_name, obj_type='file'):
    if False:
        print('Hello World!')
    "\n    Gets the primary group of the passed object\n\n    Args:\n\n        obj_name (str):\n            The path for which to obtain primary group information\n\n        obj_type (str):\n            The type of object to query. This value changes the format of the\n            ``obj_name`` parameter as follows:\n\n            - file: indicates a file or directory\n                - a relative path, such as ``FileName.txt`` or ``..\\FileName``\n                - an absolute path, such as ``C:\\DirName\\FileName.txt``\n                - A UNC name, such as ``\\\\ServerName\\ShareName\\FileName.txt``\n            - service: indicates the name of a Windows service\n            - printer: indicates the name of a printer\n            - registry: indicates a registry key\n                - Uses the following literal strings to denote the hive:\n                    - HKEY_LOCAL_MACHINE\n                    - MACHINE\n                    - HKLM\n                    - HKEY_USERS\n                    - USERS\n                    - HKU\n                    - HKEY_CURRENT_USER\n                    - CURRENT_USER\n                    - HKCU\n                    - HKEY_CLASSES_ROOT\n                    - CLASSES_ROOT\n                    - HKCR\n                - Should be in the format of ``HIVE\\Path\\To\\Key``. For example,\n                    ``HKLM\\SOFTWARE\\Windows``\n            - registry32: indicates a registry key under WOW64. Formatting is\n                the same as it is for ``registry``\n            - share: indicates a network share\n\n    Returns:\n        str: The primary group for the object\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.get_primary_group('c:\\\\file')\n    "
    try:
        obj_type_flag = flags().obj_type[obj_type.lower()]
    except KeyError:
        raise SaltInvocationError(f'Invalid "obj_type" passed: {obj_type}')
    if 'registry' in obj_type.lower():
        obj_name = dacl().get_reg_name(obj_name)
        log.debug('Name converted to: %s', obj_name)
    try:
        security_descriptor = win32security.GetNamedSecurityInfo(obj_name, obj_type_flag, win32security.GROUP_SECURITY_INFORMATION)
        primary_group_gid = security_descriptor.GetSecurityDescriptorGroup()
    except MemoryError:
        primary_group_gid = 'S-1-0-0'
    except pywintypes.error as exc:
        if exc.winerror == 1 or exc.winerror == 50:
            primary_group_gid = 'S-1-0-0'
        else:
            log.exception('Failed to get the primary group: %s', obj_name)
            raise CommandExecutionError(f'Failed to get primary group: {obj_name}', exc.strerror)
    return get_name(win32security.ConvertSidToStringSid(primary_group_gid))

def set_owner(obj_name, principal, obj_type='file'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the owner of an object. This can be a file, folder, registry key,\n    printer, service, etc...\n\n    Args:\n\n        obj_name (str):\n            The object for which to set owner. This can be the path to a file or\n            folder, a registry key, printer, etc. For more information about how\n            to format the name see:\n\n            https://msdn.microsoft.com/en-us/library/windows/desktop/aa379593(v=vs.85).aspx\n\n        principal (str):\n            The name of the user or group to make owner of the object. Can also\n            pass a SID.\n\n        obj_type (Optional[str]):\n            The type of object for which to set the owner. Default is ``file``\n\n    Returns:\n        bool: True if successful, raises an error otherwise\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.set_owner('C:\\MyDirectory', 'jsnuffy', 'file')\n    "
    sid = get_sid(principal)
    obj_flags = flags()
    if obj_type.lower() not in obj_flags.obj_type:
        raise SaltInvocationError(f'Invalid "obj_type" passed: {obj_type}')
    if 'registry' in obj_type.lower():
        obj_name = dacl().get_reg_name(obj_name)
    new_privs = set()
    luid = win32security.LookupPrivilegeValue('', 'SeTakeOwnershipPrivilege')
    new_privs.add((luid, win32con.SE_PRIVILEGE_ENABLED))
    luid = win32security.LookupPrivilegeValue('', 'SeRestorePrivilege')
    new_privs.add((luid, win32con.SE_PRIVILEGE_ENABLED))
    p_handle = win32api.GetCurrentProcess()
    t_handle = win32security.OpenProcessToken(p_handle, win32security.TOKEN_ALL_ACCESS | win32con.TOKEN_ADJUST_PRIVILEGES)
    win32security.AdjustTokenPrivileges(t_handle, 0, new_privs)
    try:
        win32security.SetNamedSecurityInfo(obj_name, obj_flags.obj_type[obj_type.lower()], obj_flags.element['owner'], sid, None, None, None)
    except pywintypes.error as exc:
        log.exception('Failed to make %s the owner: %s', principal, exc)
        raise CommandExecutionError(f'Failed to set owner: {obj_name}', exc.strerror)
    return True

def set_primary_group(obj_name, principal, obj_type='file'):
    if False:
        while True:
            i = 10
    "\n    Set the primary group of an object. This can be a file, folder, registry\n    key, printer, service, etc...\n\n    Args:\n\n        obj_name (str):\n            The object for which to set primary group. This can be the path to a\n            file or folder, a registry key, printer, etc. For more information\n            about how to format the name see:\n\n            https://msdn.microsoft.com/en-us/library/windows/desktop/aa379593(v=vs.85).aspx\n\n        principal (str):\n            The name of the group to make primary for the object. Can also pass\n            a SID.\n\n        obj_type (Optional[str]):\n            The type of object for which to set the primary group.\n\n    Returns:\n        bool: True if successful, raises an error otherwise\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.set_primary_group('C:\\MyDirectory', 'Administrators', 'file')\n    "
    if principal is None:
        principal = 'None'
    gid = get_sid(principal)
    obj_flags = flags()
    if obj_type.lower() not in obj_flags.obj_type:
        raise SaltInvocationError(f'Invalid "obj_type" passed: {obj_type}')
    if 'registry' in obj_type.lower():
        obj_name = dacl().get_reg_name(obj_name)
    new_privs = set()
    luid = win32security.LookupPrivilegeValue('', 'SeTakeOwnershipPrivilege')
    new_privs.add((luid, win32con.SE_PRIVILEGE_ENABLED))
    luid = win32security.LookupPrivilegeValue('', 'SeRestorePrivilege')
    new_privs.add((luid, win32con.SE_PRIVILEGE_ENABLED))
    p_handle = win32api.GetCurrentProcess()
    t_handle = win32security.OpenProcessToken(p_handle, win32security.TOKEN_ALL_ACCESS | win32con.TOKEN_ADJUST_PRIVILEGES)
    win32security.AdjustTokenPrivileges(t_handle, 0, new_privs)
    try:
        win32security.SetNamedSecurityInfo(obj_name, obj_flags.obj_type[obj_type.lower()], obj_flags.element['group'], None, gid, None, None)
    except pywintypes.error as exc:
        log.exception('Failed to make %s the primary group: %s', principal, exc)
        raise CommandExecutionError(f'Failed to set primary group: {obj_name}', exc.strerror)
    return True

def set_permissions(obj_name, principal, permissions, access_mode='grant', applies_to=None, obj_type='file', reset_perms=False, protected=None):
    if False:
        print('Hello World!')
    "\n    Set the permissions of an object. This can be a file, folder, registry key,\n    printer, service, etc...\n\n    Args:\n\n        obj_name (str):\n            The object for which to set permissions. This can be the path to a\n            file or folder, a registry key, printer, etc. For more information\n            about how to format the name see:\n\n            https://msdn.microsoft.com/en-us/library/windows/desktop/aa379593(v=vs.85).aspx\n\n        principal (str):\n            The name of the user or group for which to set permissions. Can also\n            pass a SID.\n\n        permissions (str, list):\n            The type of permissions to grant/deny the user. Can be one of the\n            basic permissions, or a list of advanced permissions.\n\n        access_mode (Optional[str]):\n            Whether to grant or deny user the access. Valid options are:\n\n            - grant (default): Grants the user access\n            - deny: Denies the user access\n\n        applies_to (Optional[str]):\n            The objects to which these permissions will apply. Not all these\n            options apply to all object types. Defaults to\n            'this_folder_subfolders_files'\n\n        obj_type (Optional[str]):\n            The type of object for which to set permissions. Default is 'file'\n\n        reset_perms (Optional[bool]):\n            True will overwrite the permissions on the specified object. False\n            will append the permissions. Default is False\n\n        protected (Optional[bool]):\n            True will disable inheritance for the object. False will enable\n            inheritance. None will make no change. Default is None.\n\n    Returns:\n        bool: True if successful, raises an error otherwise\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.set_permissions(\n            'C:\\Temp', 'jsnuffy', 'full_control', 'grant')\n    "
    if applies_to is None:
        if 'registry' in obj_type.lower():
            applies_to = 'this_key_subkeys'
        elif obj_type.lower() == 'file':
            applies_to = 'this_folder_subfolders_files'
    if reset_perms:
        obj_dacl = dacl(obj_type=obj_type)
    else:
        obj_dacl = dacl(obj_name, obj_type)
        obj_dacl.rm_ace(principal, access_mode)
    obj_dacl.add_ace(principal, access_mode, permissions, applies_to)
    obj_dacl.order_acl()
    obj_dacl.save(obj_name, protected)
    return True

def rm_permissions(obj_name, principal, ace_type='all', obj_type='file'):
    if False:
        i = 10
        return i + 15
    "\n    Remove a user's ACE from an object. This can be a file, folder, registry\n    key, printer, service, etc...\n\n    Args:\n\n        obj_name (str):\n            The object from which to remove the ace. This can be the\n            path to a file or folder, a registry key, printer, etc. For more\n            information about how to format the name see:\n\n            https://msdn.microsoft.com/en-us/library/windows/desktop/aa379593(v=vs.85).aspx\n\n        principal (str):\n            The name of the user or group for which to set permissions. Can also\n            pass a SID.\n\n        ace_type (Optional[str]):\n            The type of ace to remove. There are two types of ACEs, 'grant' and\n            'deny'. 'all' will remove all ACEs for the user. Default is 'all'\n\n        obj_type (Optional[str]):\n            The type of object for which to set permissions. Default is 'file'\n\n    Returns:\n        bool: True if successful, raises an error otherwise\n\n    Usage:\n\n    .. code-block:: python\n\n        # Remove jsnuffy's grant ACE from C:\\Temp\n        salt.utils.win_dacl.rm_permissions('C:\\\\Temp', 'jsnuffy', 'grant')\n\n        # Remove all ACEs for jsnuffy from C:\\Temp\n        salt.utils.win_dacl.rm_permissions('C:\\\\Temp', 'jsnuffy')\n    "
    obj_dacl = dacl(obj_name, obj_type)
    obj_dacl.rm_ace(principal, ace_type)
    obj_dacl.save(obj_name)
    return True

def get_permissions(obj_name, principal=None, obj_type='file'):
    if False:
        while True:
            i = 10
    "\n    Get the permissions for the passed object\n\n    Args:\n\n        obj_name (str):\n            The name of or path to the object.\n\n        principal (Optional[str]):\n            The name of the user or group for which to get permissions. Can also\n            pass a SID. If None, all ACEs defined on the object will be\n            returned. Default is None\n\n        obj_type (Optional[str]):\n            The type of object for which to get permissions.\n\n    Returns:\n        dict: A dictionary representing the object permissions\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.get_permissions('C:\\Temp')\n    "
    obj_dacl = dacl(obj_name=obj_name, obj_type=obj_type)
    if principal is None:
        return obj_dacl.list_aces()
    return obj_dacl.get_ace(principal)

def has_permission(obj_name, principal, permission, access_mode='grant', obj_type='file', exact=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check if the object has a specific permission\n\n    Args:\n\n        obj_name (str):\n            The name of or path to the object.\n\n        principal (str):\n            The name of the user or group for which to get permissions. Can also\n            pass a SID.\n\n        permission (str):\n            The permission to verify. Valid options depend on the obj_type.\n\n        access_mode (Optional[str]):\n            The access mode to check. Is the user granted or denied the\n            permission. Default is 'grant'. Valid options are:\n\n            - grant\n            - deny\n\n        obj_type (Optional[str]):\n            The type of object for which to check permissions. Default is 'file'\n\n        exact (Optional[bool]):\n            True for an exact match, otherwise check to see if the permission is\n            included in the ACE. Default is True\n\n    Returns:\n        bool: True if the object has the permission, otherwise False\n\n    Usage:\n\n    .. code-block:: python\n\n        # Does Joe have read permissions to C:\\Temp\n        salt.utils.win_dacl.has_permission('C:\\\\Temp', 'joe', 'read', 'grant', exact=False)\n\n        # Does Joe have Full Control of C:\\Temp\n        salt.utils.win_dacl.has_permission('C:\\\\Temp', 'joe', 'full_control', 'grant')\n    "
    if access_mode.lower() not in ['grant', 'deny']:
        raise SaltInvocationError(f'Invalid "access_mode" passed: {access_mode}')
    access_mode = access_mode.lower()
    obj_dacl = dacl(obj_name, obj_type)
    obj_type = obj_type.lower()
    sid = get_sid(principal)
    chk_flag = obj_dacl.ace_perms[obj_type]['basic'].get(permission.lower(), obj_dacl.ace_perms[obj_type]['advanced'].get(permission.lower(), False))
    if not chk_flag:
        raise SaltInvocationError(f'Invalid "permission" passed: {permission}')
    cur_flag = None
    for i in range(0, obj_dacl.dacl.GetAceCount()):
        ace = obj_dacl.dacl.GetAce(i)
        if ace[2] == sid and obj_dacl.ace_type[ace[0][0]] == access_mode:
            cur_flag = ace[1]
    if not cur_flag:
        return False
    if exact:
        return cur_flag == chk_flag
    return cur_flag & chk_flag == chk_flag

def has_permissions(obj_name, principal, permissions, access_mode='grant', obj_type='file', exact=True):
    if False:
        return 10
    "\n    Check if the object has the passed permissions. Can be all them or the exact\n    permissions passed and nothing more.\n\n    Args:\n\n        obj_name (str):\n            The name of or path to the object.\n\n        principal (str):\n            The name of the user or group for which to get permissions. Can also\n            pass a SID.\n\n        permissions (list):\n            The list of permissions to verify\n\n        access_mode (Optional[str]):\n            The access mode to check. Is the user granted or denied the\n            permission. Default is 'grant'. Valid options are:\n\n            - grant\n            - deny\n\n        obj_type (Optional[str]):\n            The type of object for which to check permissions. Default is 'file'\n\n        exact (Optional[bool]):\n            ``True`` checks if the permissions are exactly those passed in\n            permissions. ``False`` checks to see if the permissions are included\n            in the ACE. Default is ``True``\n\n    Returns:\n        bool: True if the object has the permission, otherwise False\n\n    Usage:\n\n    .. code-block:: python\n\n        # Does Joe have read and write permissions to C:\\Temp\n        salt.utils.win_dacl.has_permission('C:\\\\Temp', 'joe', ['read', 'write'], 'grant', exact=False)\n\n        # Does Joe have Full Control of C:\\Temp\n        salt.utils.win_dacl.has_permissions('C:\\\\Temp', 'joe', 'full_control', 'grant')\n    "
    if isinstance(permissions, str):
        return has_permission(obj_name=obj_name, obj_type=obj_type, permission=permissions, access_mode=access_mode, principal=principal, exact=exact)
    if access_mode.lower() not in ['grant', 'deny']:
        raise SaltInvocationError(f'Invalid "access_mode" passed: {access_mode}')
    access_mode = access_mode.lower()
    obj_dacl = dacl(obj_name, obj_type)
    obj_type = obj_type.lower()
    sid = get_sid(principal)
    chk_flag = 0
    for permission in permissions:
        chk_flag |= obj_dacl.ace_perms[obj_type]['basic'].get(permission.lower(), obj_dacl.ace_perms[obj_type]['advanced'].get(permission.lower(), False))
        if not chk_flag:
            raise SaltInvocationError(f'Invalid "permission" passed: {permission}')
    cur_flag = None
    for i in range(0, obj_dacl.dacl.GetAceCount()):
        ace = obj_dacl.dacl.GetAce(i)
        if ace[2] == sid and obj_dacl.ace_type[ace[0][0]] == access_mode:
            cur_flag = ace[1]
    if not cur_flag:
        return False
    if exact:
        return cur_flag == chk_flag
    return cur_flag & chk_flag == chk_flag

def set_inheritance(obj_name, enabled, obj_type='file', clear=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Enable or disable an objects inheritance.\n\n    Args:\n\n        obj_name (str):\n            The name of the object\n\n        enabled (bool):\n            True to enable inheritance, False to disable\n\n        obj_type (Optional[str]):\n            The type of object. Only three objects allow inheritance. Valid\n            objects are:\n\n            - file (default): This is a file or directory\n            - registry\n            - registry32 (for WOW64)\n\n        clear (Optional[bool]):\n            True to clear existing ACEs, False to keep existing ACEs.\n            Default is False\n\n    Returns:\n        bool: True if successful, otherwise an Error\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.set_inheritance('C:\\Temp', False)\n    "
    if obj_type not in ['file', 'registry', 'registry32']:
        raise SaltInvocationError(f'obj_type called with incorrect parameter: {obj_name}')
    if clear:
        obj_dacl = dacl(obj_type=obj_type)
    else:
        obj_dacl = dacl(obj_name, obj_type)
    return obj_dacl.save(obj_name, not enabled)

def get_inheritance(obj_name, obj_type='file'):
    if False:
        print('Hello World!')
    "\n    Get an object's inheritance.\n\n    Args:\n\n        obj_name (str):\n            The name of the object\n\n        obj_type (Optional[str]):\n            The type of object. Only three object types allow inheritance. Valid\n            objects are:\n\n            - file (default): This is a file or directory\n            - registry\n            - registry32 (for WOW64)\n\n            The following should return False as there is no inheritance:\n\n            - service\n            - printer\n            - share\n\n    Returns:\n        bool: True if enabled, otherwise False\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.get_inheritance('HKLM\\SOFTWARE\\salt', 'registry')\n    "
    obj_dacl = dacl(obj_name=obj_name, obj_type=obj_type)
    inherited = win32security.INHERITED_ACE
    for i in range(0, obj_dacl.dacl.GetAceCount()):
        ace = obj_dacl.dacl.GetAce(i)
        if ace[0][1] & inherited == inherited:
            return True
    return False

def copy_security(source, target, obj_type='file', copy_owner=True, copy_group=True, copy_dacl=True, copy_sacl=True):
    if False:
        i = 10
        return i + 15
    "\n    Copy the security descriptor of the Source to the Target. You can specify a\n    specific portion of the security descriptor to copy using one of the\n    `copy_*` parameters.\n\n    .. note::\n        At least one `copy_*` parameter must be ``True``\n\n    .. note::\n        The user account running this command must have the following\n        privileges:\n\n        - SeTakeOwnershipPrivilege\n        - SeRestorePrivilege\n        - SeSecurityPrivilege\n\n    Args:\n\n        source (str):\n            The full path to the source. This is where the security info will be\n            copied from\n\n        target (str):\n            The full path to the target. This is where the security info will be\n            applied\n\n        obj_type (str): file\n            The type of object to query. This value changes the format of the\n            ``obj_name`` parameter as follows:\n            - file: indicates a file or directory\n                - a relative path, such as ``FileName.txt`` or ``..\\FileName``\n                - an absolute path, such as ``C:\\DirName\\FileName.txt``\n                - A UNC name, such as ``\\\\ServerName\\ShareName\\FileName.txt``\n            - service: indicates the name of a Windows service\n            - printer: indicates the name of a printer\n            - registry: indicates a registry key\n                - Uses the following literal strings to denote the hive:\n                    - HKEY_LOCAL_MACHINE\n                    - MACHINE\n                    - HKLM\n                    - HKEY_USERS\n                    - USERS\n                    - HKU\n                    - HKEY_CURRENT_USER\n                    - CURRENT_USER\n                    - HKCU\n                    - HKEY_CLASSES_ROOT\n                    - CLASSES_ROOT\n                    - HKCR\n                - Should be in the format of ``HIVE\\Path\\To\\Key``. For example,\n                    ``HKLM\\SOFTWARE\\Windows``\n            - registry32: indicates a registry key under WOW64. Formatting is\n                the same as it is for ``registry``\n            - share: indicates a network share\n\n        copy_owner (bool): True\n            ``True`` copies owner information. Default is ``True``\n\n        copy_group (bool): True\n            ``True`` copies group information. Default is ``True``\n\n        copy_dacl (bool): True\n            ``True`` copies the DACL. Default is ``True``\n\n        copy_sacl (bool): True\n            ``True`` copies the SACL. Default is ``True``\n\n    Returns:\n        bool: ``True`` if successful\n\n    Raises:\n        SaltInvocationError: When parameters are invalid\n        CommandExecutionError: On failure to set security\n\n    Usage:\n\n    .. code-block:: python\n\n        salt.utils.win_dacl.copy_security(\n            source='C:\\\\temp\\\\source_file.txt',\n            target='C:\\\\temp\\\\target_file.txt',\n            obj_type='file')\n\n        salt.utils.win_dacl.copy_security(\n            source='HKLM\\\\SOFTWARE\\\\salt\\\\test_source',\n            target='HKLM\\\\SOFTWARE\\\\salt\\\\test_target',\n            obj_type='registry',\n            copy_owner=False)\n    "
    obj_dacl = dacl(obj_type=obj_type)
    if 'registry' in obj_type.lower():
        source = obj_dacl.get_reg_name(source)
        log.info('Source converted to: %s', source)
        target = obj_dacl.get_reg_name(target)
        log.info('Target converted to: %s', target)
    try:
        obj_type_flag = flags().obj_type[obj_type.lower()]
    except KeyError:
        raise SaltInvocationError(f'Invalid "obj_type" passed: {obj_type}')
    security_flags = 0
    if copy_owner:
        security_flags |= win32security.OWNER_SECURITY_INFORMATION
    if copy_group:
        security_flags |= win32security.GROUP_SECURITY_INFORMATION
    if copy_dacl:
        security_flags |= win32security.DACL_SECURITY_INFORMATION
    if copy_sacl:
        security_flags |= win32security.SACL_SECURITY_INFORMATION
    if not security_flags:
        raise SaltInvocationError('One of copy_owner, copy_group, copy_dacl, or copy_sacl must be True')
    new_privs = set()
    luid = win32security.LookupPrivilegeValue('', 'SeTakeOwnershipPrivilege')
    new_privs.add((luid, win32con.SE_PRIVILEGE_ENABLED))
    luid = win32security.LookupPrivilegeValue('', 'SeRestorePrivilege')
    new_privs.add((luid, win32con.SE_PRIVILEGE_ENABLED))
    luid = win32security.LookupPrivilegeValue('', 'SeSecurityPrivilege')
    new_privs.add((luid, win32con.SE_PRIVILEGE_ENABLED))
    p_handle = win32api.GetCurrentProcess()
    t_handle = win32security.OpenProcessToken(p_handle, win32security.TOKEN_ALL_ACCESS | win32con.TOKEN_ADJUST_PRIVILEGES)
    win32security.AdjustTokenPrivileges(t_handle, 0, new_privs)
    sec = win32security.GetNamedSecurityInfo(source, obj_type_flag, security_flags)
    sd_sid = sec.GetSecurityDescriptorOwner()
    sd_gid = sec.GetSecurityDescriptorGroup()
    sd_dacl = sec.GetSecurityDescriptorDacl()
    sd_sacl = sec.GetSecurityDescriptorSacl()
    try:
        win32security.SetNamedSecurityInfo(target, obj_type_flag, security_flags, sd_sid, sd_gid, sd_dacl, sd_sacl)
    except pywintypes.error as exc:
        raise CommandExecutionError(f'Failed to set security info: {exc.strerror}')
    return True

def _check_perms(obj_name, obj_type, new_perms, access_mode, ret, test_mode=False):
    if False:
        while True:
            i = 10
    "\n    Helper function used by ``check_perms`` for checking and setting Grant and\n    Deny permissions.\n\n    Args:\n\n        obj_name (str):\n            The name or full path to the object\n\n        obj_type (Optional[str]):\n            The type of object for which to check permissions. Default is 'file'\n\n        new_perms (dict):\n            A dictionary containing the user/group and the basic permissions to\n            check/grant, ie: ``{'user': {'perms': 'basic_permission'}}``.\n\n        access_mode (str):\n            The access mode to set. Either ``grant`` or ``deny``\n\n        ret (dict):\n            A dictionary to append changes to and return. If not passed, will\n            create a new dictionary to return.\n\n        test_mode (bool):\n            ``True`` will only return the changes that would be made. ``False``\n            will make the changes as well as return the changes that would be\n            made.\n\n    Returns:\n        dict: A dictionary of return data as expected by the state system\n    "
    access_mode = access_mode.lower()
    perms_label = f'{access_mode}_perms'
    cur_perms = get_permissions(obj_name=obj_name, obj_type=obj_type)
    changes = {}
    for user in new_perms:
        applies_to_text = ''
        try:
            user_name = get_name(principal=user)
        except CommandExecutionError:
            ret['comment'].append('{} Perms: User "{}" missing from Target System'.format(access_mode.capitalize(), user))
            continue
        if user_name not in cur_perms['Not Inherited']:
            changes.setdefault(user, {})
            changes[user]['permissions'] = new_perms[user]['perms']
            if 'applies_to' in new_perms[user]:
                changes[user]['applies_to'] = new_perms[user]['applies_to']
        elif not has_permissions(obj_name=obj_name, principal=user_name, permissions=new_perms[user]['perms'], access_mode=access_mode, obj_type=obj_type, exact=True):
            changes.setdefault(user, {})
            changes[user]['permissions'] = new_perms[user]['perms']
            if 'applies_to' in new_perms[user]:
                applies_to = new_perms[user]['applies_to']
                at_flag = flags().ace_prop[obj_type][applies_to]
                applies_to_text = flags().ace_prop[obj_type][at_flag]
                if access_mode in cur_perms['Not Inherited'][user_name]:
                    if not cur_perms['Not Inherited'][user_name][access_mode]['applies to'] == applies_to_text:
                        changes.setdefault(user, {})
                        changes[user]['applies_to'] = applies_to
    if changes:
        ret['changes'].setdefault(perms_label, {})
        for user in changes:
            user_name = get_name(principal=user)
            if test_mode is True:
                ret['changes'][perms_label].setdefault(user, {})
                ret['changes'][perms_label][user] = changes[user]
            elif not test_mode:
                try:
                    set_permissions(obj_name=obj_name, principal=user_name, permissions=changes[user]['permissions'], access_mode=access_mode, applies_to=changes[user].get('applies_to'), obj_type=obj_type)
                    ret['changes'].setdefault(perms_label, {}).setdefault(user, {})
                    ret['changes'][perms_label][user] = changes[user]
                except CommandExecutionError as exc:
                    ret['result'] = False
                    ret['comment'].append('Failed to change {} permissions for "{}" to {}\nError: {}'.format(access_mode, user, changes[user], exc.strerror))
    return ret

def check_perms(obj_name, obj_type='file', ret=None, owner=None, grant_perms=None, deny_perms=None, inheritance=True, reset=False, test_mode=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check owner and permissions for the passed directory. This function checks\n    the permissions and sets them, returning the changes made.\n\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        obj_name (str):\n            The name or full path to the object\n\n        obj_type (Optional[str]):\n            The type of object for which to check permissions. Default is 'file'\n\n        ret (dict):\n            A dictionary to append changes to and return. If not passed, will\n            create a new dictionary to return.\n\n        owner (str):\n            The owner to set for the directory.\n\n        grant_perms (dict):\n            A dictionary containing the user/group and the basic permissions to\n            check/grant, ie: ``{'user': {'perms': 'basic_permission'}}``.\n            Default is ``None``.\n\n        deny_perms (dict):\n            A dictionary containing the user/group and permissions to\n            check/deny. Default is ``None``.\n\n        inheritance (bool):\n            ``True`` will enable inheritance from the parent object. ``False``\n            will disable inheritance. Default is ``True``.\n\n        reset (bool):\n            ``True`` will clear the DACL and set only the permissions defined\n             in ``grant_perms`` and ``deny_perms``. ``False`` append permissions\n             to the existing DACL. Default is ``False``. This does NOT affect\n            inherited permissions.\n\n        test_mode (bool):\n            ``True`` will only return the changes that would be made. ``False``\n            will make the changes as well as return the changes that would be\n            made.\n\n    Returns:\n        dict: A dictionary of changes that have been made\n\n    Usage:\n\n    .. code-block:: bash\n\n        # You have to use __utils__ in order for __opts__ to be available\n\n        # To see changes to ``C:\\Temp`` if the 'Users' group is given 'read & execute' permissions.\n        __utils__['dacl.check_perms'](obj_name='C:\\Temp',\n                                      obj_type='file',\n                                      owner='Administrators',\n                                      grant_perms={\n                                          'Users': {\n                                              'perms': 'read_execute'\n                                          }\n                                      })\n\n        # Specify advanced attributes with a list\n        __utils__['dacl.check_perms'](obj_name='C:\\Temp',\n                                      obj_type='file',\n                                      owner='Administrators',\n                                      grant_perms={\n                                          'jsnuffy': {\n                                              'perms': [\n                                                  'read_attributes',\n                                                  'read_ea'\n                                              ],\n                                              'applies_to': 'files_only'\n                                          }\n                                      })\n    "
    if obj_type.lower() not in flags().obj_type:
        raise SaltInvocationError(f'Invalid "obj_type" passed: {obj_type}')
    obj_type = obj_type.lower()
    if not ret:
        ret = {'name': obj_name, 'changes': {}, 'comment': [], 'result': True}
        orig_comment = ''
    else:
        orig_comment = ret['comment']
        ret['comment'] = []
    if owner:
        owner = get_name(principal=owner)
        current_owner = get_owner(obj_name=obj_name, obj_type=obj_type)
        if owner != current_owner:
            if test_mode is True:
                ret['changes']['owner'] = owner
            else:
                try:
                    set_owner(obj_name=obj_name, principal=owner, obj_type=obj_type)
                    log.debug('Owner set to %s', owner)
                    ret['changes']['owner'] = owner
                except CommandExecutionError:
                    ret['result'] = False
                    ret['comment'].append(f'Failed to change owner to "{owner}"')
    if inheritance is not None:
        if not inheritance == get_inheritance(obj_name=obj_name, obj_type=obj_type):
            if test_mode is True:
                ret['changes']['inheritance'] = inheritance
            else:
                try:
                    set_inheritance(obj_name=obj_name, enabled=inheritance, obj_type=obj_type)
                    log.debug('%s inheritance', 'Enabling' if inheritance else 'Disabling')
                    ret['changes']['inheritance'] = inheritance
                except CommandExecutionError:
                    ret['result'] = False
                    ret['comment'].append('Failed to set inheritance for "{}" to {}'.format(obj_name, inheritance))
    if reset:
        log.debug('Resetting permissions for %s', obj_name)
        cur_perms = get_permissions(obj_name=obj_name, obj_type=obj_type)
        for user_name in cur_perms['Not Inherited']:
            if user_name not in {get_name(k) for k in grant_perms or {}}:
                if 'grant' in cur_perms['Not Inherited'][user_name]:
                    ret['changes'].setdefault('remove_perms', {})
                    if test_mode is True:
                        ret['changes']['remove_perms'].update({user_name: cur_perms['Not Inherited'][user_name]})
                    else:
                        rm_permissions(obj_name=obj_name, principal=user_name, ace_type='grant', obj_type=obj_type)
                        ret['changes']['remove_perms'].update({user_name: cur_perms['Not Inherited'][user_name]})
            if user_name not in {get_name(k) for k in deny_perms or {}}:
                if 'deny' in cur_perms['Not Inherited'][user_name]:
                    ret['changes'].setdefault('remove_perms', {})
                    if test_mode is True:
                        ret['changes']['remove_perms'].update({user_name: cur_perms['Not Inherited'][user_name]})
                    else:
                        rm_permissions(obj_name=obj_name, principal=user_name, ace_type='deny', obj_type=obj_type)
                        ret['changes']['remove_perms'].update({user_name: cur_perms['Not Inherited'][user_name]})
    log.debug('Getting current permissions for %s', obj_name)
    if deny_perms is not None:
        ret = _check_perms(obj_name=obj_name, obj_type=obj_type, new_perms=deny_perms, access_mode='deny', ret=ret, test_mode=test_mode)
    if grant_perms is not None:
        ret = _check_perms(obj_name=obj_name, obj_type=obj_type, new_perms=grant_perms, access_mode='grant', ret=ret, test_mode=test_mode)
    if reset and (not test_mode):
        log.debug('Resetting permissions for %s', obj_name)
        cur_perms = get_permissions(obj_name=obj_name, obj_type=obj_type)
        for user_name in cur_perms['Not Inherited']:
            if user_name not in {get_name(k) for k in grant_perms or {}}:
                if 'grant' in cur_perms['Not Inherited'][user_name]:
                    rm_permissions(obj_name=obj_name, principal=user_name, ace_type='grant', obj_type=obj_type)
            if user_name not in {get_name(k) for k in deny_perms or {}}:
                if 'deny' in cur_perms['Not Inherited'][user_name]:
                    rm_permissions(obj_name=obj_name, principal=user_name, ace_type='deny', obj_type=obj_type)
    if isinstance(orig_comment, str):
        if orig_comment:
            ret['comment'].insert(0, orig_comment)
    elif orig_comment:
        ret['comment'] = orig_comment.extend(ret['comment'])
    ret['comment'] = '\n'.join(ret['comment'])
    if test_mode and ret['changes']:
        ret['result'] = None
    return ret

def _set_perms(obj_dacl, obj_type, new_perms, cur_perms, access_mode):
    if False:
        print('Hello World!')
    obj_type = obj_type.lower()
    ret = {}
    for user in new_perms:
        try:
            user_name = get_name(user)
        except CommandExecutionError:
            log.debug('%s Perms: User "%s" missing from Target System', access_mode.capitalize(), user)
            continue
        applies_to = None
        if obj_type in ['file', 'registry', 'registry32']:
            if 'applies_to' not in new_perms[user]:
                if user_name in cur_perms['Not Inherited'] and 'deny' in cur_perms['Not Inherited'][user_name]:
                    for flag in flags().ace_prop[obj_type]:
                        if flags().ace_prop[obj_type][flag] == cur_perms['Not Inherited'][user_name]['deny']['applies to']:
                            at_flag = flag
                            for flag1 in flags().ace_prop[obj_type]:
                                if flags().ace_prop[obj_type][flag1] == at_flag:
                                    applies_to = flag1
                if not applies_to:
                    if obj_type == 'file':
                        applies_to = 'this_folder_subfolders_files'
                    elif 'registry' in obj_type:
                        applies_to = 'this_key_subkeys'
            else:
                applies_to = new_perms[user]['applies_to']
        if obj_dacl.add_ace(principal=user, access_mode=access_mode, permissions=new_perms[user]['perms'], applies_to=applies_to):
            ret[user] = new_perms[user]
    return ret

def set_perms(obj_name, obj_type='file', grant_perms=None, deny_perms=None, inheritance=True, reset=False):
    if False:
        while True:
            i = 10
    '\n    Set permissions for the given path\n\n    .. versionadded:: 2019.2.0\n\n    Args:\n\n        obj_name (str):\n            The name or full path to the object\n\n        obj_type (Optional[str]):\n            The type of object for which to check permissions. Default is \'file\'\n\n        grant_perms (dict):\n            A dictionary containing the user/group and the basic permissions to\n            grant, ie: ``{\'user\': {\'perms\': \'basic_permission\'}}``. You can also\n            set the ``applies_to`` setting here. The default for ``applise_to``\n            is ``this_folder_subfolders_files``. Specify another ``applies_to``\n            setting like this:\n\n            .. code-block:: yaml\n\n                {\'user\': {\'perms\': \'full_control\', \'applies_to\': \'this_folder\'}}\n\n            To set advanced permissions use a list for the ``perms`` parameter,\n            ie:\n\n            .. code-block:: yaml\n\n                {\'user\': {\'perms\': [\'read_attributes\', \'read_ea\'], \'applies_to\': \'this_folder\'}}\n\n            To see a list of available attributes and applies to settings see\n            the documentation for salt.utils.win_dacl.\n\n            A value of ``None`` will make no changes to the ``grant`` portion of\n            the DACL. Default is ``None``.\n\n        deny_perms (dict):\n            A dictionary containing the user/group and permissions to deny along\n            with the ``applies_to`` setting. Use the same format used for the\n            ``grant_perms`` parameter. Remember, deny permissions supersede\n            grant permissions.\n\n            A value of ``None`` will make no changes to the ``deny`` portion of\n            the DACL. Default is ``None``.\n\n        inheritance (bool):\n            If ``True`` the object will inherit permissions from the parent, if\n            ``False``, inheritance will be disabled. Inheritance setting will\n            not apply to parent directories if they must be created. Default is\n            ``False``.\n\n        reset (bool):\n            If ``True`` the existing DCL will be cleared and replaced with the\n            settings defined in this function. If ``False``, new entries will be\n            appended to the existing DACL. Default is ``False``.\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: If unsuccessful\n\n    Usage:\n\n    .. code-block:: bash\n\n        import salt.utils.win_dacl\n\n        # To grant the \'Users\' group \'read & execute\' permissions.\n        salt.utils.win_dacl.set_perms(obj_name=\'C:\\Temp\',\n                                      obj_type=\'file\',\n                                      grant_perms={\n                                          \'Users\': {\n                                              \'perms\': \'read_execute\'\n                                          }\n                                      })\n\n        # Specify advanced attributes with a list\n        salt.utils.win_dacl.set_perms(obj_name=\'C:\\Temp\',\n                                      obj_type=\'file\',\n                                      grant_perms={\n                                          \'jsnuffy\': {\n                                              \'perms\': [\n                                                  \'read_attributes\',\n                                                  \'read_ea\'\n                                              ],\n                                              \'applies_to\': \'this_folder_only\'\n                                          }\n                                      }"\n    '
    ret = {}
    if reset:
        obj_dacl = dacl(obj_type=obj_type)
        cur_perms = {'Inherited': {}, 'Not Inherited': {}}
    else:
        obj_dacl = dacl(obj_name, obj_type=obj_type)
        cur_perms = get_permissions(obj_name=obj_name, obj_type=obj_type)
    if deny_perms is not None:
        ret['deny'] = _set_perms(obj_dacl=obj_dacl, obj_type=obj_type, new_perms=deny_perms, cur_perms=cur_perms, access_mode='deny')
    if grant_perms is not None:
        ret['grant'] = _set_perms(obj_dacl=obj_dacl, obj_type=obj_type, new_perms=grant_perms, cur_perms=cur_perms, access_mode='grant')
    obj_dacl.order_acl()
    if obj_dacl.save(obj_name, not inheritance):
        return ret
    return {}