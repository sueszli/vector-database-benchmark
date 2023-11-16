import os
import ntsecuritycon
import pythoncom
import win32api
import win32com.server.policy
import win32con
import win32security
from ntsecuritycon import FILE_ALL_ACCESS, FILE_APPEND_DATA, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_READ_ATTRIBUTES, FILE_READ_DATA, FILE_READ_EA, FILE_WRITE_ATTRIBUTES, FILE_WRITE_DATA, FILE_WRITE_EA, PSPCB_SI_INITDIALOG, READ_CONTROL, SI_ACCESS_CONTAINER, SI_ACCESS_GENERAL, SI_ACCESS_PROPERTY, SI_ACCESS_SPECIFIC, SI_ADVANCED, SI_CONTAINER, SI_EDIT_ALL, SI_EDIT_AUDITS, SI_EDIT_PROPERTIES, SI_PAGE_ADVPERM, SI_PAGE_AUDIT, SI_PAGE_OWNER, SI_PAGE_PERM, SI_PAGE_TITLE, SI_RESET, STANDARD_RIGHTS_EXECUTE, STANDARD_RIGHTS_READ, STANDARD_RIGHTS_WRITE, SYNCHRONIZE, WRITE_DAC, WRITE_OWNER
from pythoncom import IID_NULL
from win32com.authorization import authorization
from win32com.shell.shellcon import PSPCB_CREATE, PSPCB_RELEASE
from win32security import CONTAINER_INHERIT_ACE, INHERIT_ONLY_ACE, OBJECT_INHERIT_ACE

class SecurityInformation(win32com.server.policy.DesignatedWrapPolicy):
    _com_interfaces_ = [authorization.IID_ISecurityInformation]
    _public_methods_ = ['GetObjectInformation', 'GetSecurity', 'SetSecurity', 'GetAccessRights', 'GetInheritTypes', 'MapGeneric', 'PropertySheetPageCallback']

    def __init__(self, FileName):
        if False:
            print('Hello World!')
        self.FileName = FileName
        self._wrap_(self)

    def GetObjectInformation(self):
        if False:
            for i in range(10):
                print('nop')
        'Identifies object whose security will be modified, and determines options available\n        to the end user'
        flags = SI_ADVANCED | SI_EDIT_ALL | SI_PAGE_TITLE | SI_RESET
        if os.path.isdir(self.FileName):
            flags |= SI_CONTAINER
        hinstance = 0
        servername = ''
        objectname = os.path.split(self.FileName)[1]
        pagetitle = 'Python ACL Editor'
        if os.path.isdir(self.FileName):
            pagetitle += ' (dir)'
        else:
            pagetitle += ' (file)'
        objecttype = IID_NULL
        return (flags, hinstance, servername, objectname, pagetitle, objecttype)

    def GetSecurity(self, requestedinfo, bdefault):
        if False:
            for i in range(10):
                print('nop')
        'Requests the existing permissions for object'
        if bdefault:
            return win32security.SECURITY_DESCRIPTOR()
        else:
            return win32security.GetNamedSecurityInfo(self.FileName, win32security.SE_FILE_OBJECT, requestedinfo)

    def SetSecurity(self, requestedinfo, sd):
        if False:
            for i in range(10):
                print('nop')
        'Applies permissions to the object'
        owner = sd.GetSecurityDescriptorOwner()
        group = sd.GetSecurityDescriptorGroup()
        dacl = sd.GetSecurityDescriptorDacl()
        sacl = sd.GetSecurityDescriptorSacl()
        win32security.SetNamedSecurityInfo(self.FileName, win32security.SE_FILE_OBJECT, requestedinfo, owner, group, dacl, sacl)

    def GetAccessRights(self, objecttype, flags):
        if False:
            i = 10
            return i + 15
        'Returns a tuple of (AccessRights, DefaultAccess), where AccessRights is a sequence of tuples representing\n        SI_ACCESS structs, containing (guid, access mask, Name, flags). DefaultAccess indicates which of the\n        AccessRights will be used initially when a new ACE is added (zero based).\n        Flags can contain SI_ACCESS_SPECIFIC,SI_ACCESS_GENERAL,SI_ACCESS_CONTAINER,SI_ACCESS_PROPERTY,\n              CONTAINER_INHERIT_ACE,INHERIT_ONLY_ACE,OBJECT_INHERIT_ACE\n        '
        if objecttype is not None and objecttype != IID_NULL:
            raise NotImplementedError('Object type is not supported')
        if os.path.isdir(self.FileName):
            file_append_data_desc = 'Create subfolders'
            file_write_data_desc = 'Create Files'
        else:
            file_append_data_desc = 'Append data'
            file_write_data_desc = 'Write data'
        accessrights = [(IID_NULL, FILE_GENERIC_READ, 'Generic read', SI_ACCESS_GENERAL | SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE), (IID_NULL, FILE_GENERIC_WRITE, 'Generic write', SI_ACCESS_GENERAL | SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE), (IID_NULL, win32con.DELETE, 'Delete', SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE), (IID_NULL, WRITE_OWNER, 'Change owner', SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE), (IID_NULL, READ_CONTROL, 'Read Permissions', SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE), (IID_NULL, WRITE_DAC, 'Change permissions', SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE), (IID_NULL, FILE_APPEND_DATA, file_append_data_desc, SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE), (IID_NULL, FILE_WRITE_DATA, file_write_data_desc, SI_ACCESS_SPECIFIC | OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE)]
        return (accessrights, 0)

    def MapGeneric(self, guid, aceflags, mask):
        if False:
            while True:
                i = 10
        'Converts generic access rights to specific rights.  This implementation uses standard file system rights,\n        but you can map them any way that suits your application.\n        '
        return win32security.MapGenericMask(mask, (FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_GENERIC_EXECUTE, FILE_ALL_ACCESS))

    def GetInheritTypes(self):
        if False:
            while True:
                i = 10
        'Specifies which types of ACE inheritance are supported.\n        Returns a sequence of tuples representing SI_INHERIT_TYPE structs, containing\n        (object type guid, inheritance flags, display name).  Guid is usually only used with\n        Directory Service objects.\n        '
        return ((IID_NULL, 0, 'Only current object'), (IID_NULL, OBJECT_INHERIT_ACE, 'Files inherit permissions'), (IID_NULL, CONTAINER_INHERIT_ACE, 'Sub Folders inherit permissions'), (IID_NULL, CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE, 'Files and subfolders'))

    def PropertySheetPageCallback(self, hwnd, msg, pagetype):
        if False:
            i = 10
            return i + 15
        'Invoked each time a property sheet page is created or destroyed.'
        return None

    def EditSecurity(self, owner_hwnd=0):
        if False:
            i = 10
            return i + 15
        'Creates an ACL editor dialog based on parameters returned by interface methods'
        isi = pythoncom.WrapObject(self, authorization.IID_ISecurityInformation, pythoncom.IID_IUnknown)
        authorization.EditSecurity(owner_hwnd, isi)
temp_dir = win32api.GetTempPath()
dir_name = win32api.GetTempFileName(temp_dir, 'isi')[0]
print(dir_name)
os.remove(dir_name)
os.mkdir(dir_name)
si = SecurityInformation(dir_name)
si.EditSecurity()
fname = win32api.GetTempFileName(dir_name, 'isi')[0]
si = SecurityInformation(fname)
si.EditSecurity()