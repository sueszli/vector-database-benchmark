"""
Implements a permissions editor for services.
Service can be specified as plain name for local machine,
or as a remote service of the form \\\\machinename\\service
"""
import os
import ntsecuritycon
import pythoncom
import win32api
import win32com.server.policy
import win32con
import win32security
import win32service
from win32com.authorization import authorization
SERVICE_GENERIC_EXECUTE = win32service.SERVICE_START | win32service.SERVICE_STOP | win32service.SERVICE_PAUSE_CONTINUE | win32service.SERVICE_USER_DEFINED_CONTROL
SERVICE_GENERIC_READ = win32service.SERVICE_QUERY_CONFIG | win32service.SERVICE_QUERY_STATUS | win32service.SERVICE_INTERROGATE | win32service.SERVICE_ENUMERATE_DEPENDENTS
SERVICE_GENERIC_WRITE = win32service.SERVICE_CHANGE_CONFIG
from ntsecuritycon import PSPCB_SI_INITDIALOG, READ_CONTROL, SI_ACCESS_CONTAINER, SI_ACCESS_GENERAL, SI_ACCESS_PROPERTY, SI_ACCESS_SPECIFIC, SI_ADVANCED, SI_CONTAINER, SI_EDIT_ALL, SI_EDIT_AUDITS, SI_EDIT_PROPERTIES, SI_PAGE_ADVPERM, SI_PAGE_AUDIT, SI_PAGE_OWNER, SI_PAGE_PERM, SI_PAGE_TITLE, SI_RESET, STANDARD_RIGHTS_EXECUTE, STANDARD_RIGHTS_READ, STANDARD_RIGHTS_WRITE, WRITE_DAC, WRITE_OWNER
from pythoncom import IID_NULL
from win32com.shell.shellcon import PSPCB_CREATE, PSPCB_RELEASE
from win32security import CONTAINER_INHERIT_ACE, INHERIT_ONLY_ACE, OBJECT_INHERIT_ACE

class ServiceSecurity(win32com.server.policy.DesignatedWrapPolicy):
    _com_interfaces_ = [authorization.IID_ISecurityInformation]
    _public_methods_ = ['GetObjectInformation', 'GetSecurity', 'SetSecurity', 'GetAccessRights', 'GetInheritTypes', 'MapGeneric', 'PropertySheetPageCallback']

    def __init__(self, ServiceName):
        if False:
            for i in range(10):
                print('nop')
        self.ServiceName = ServiceName
        self._wrap_(self)

    def GetObjectInformation(self):
        if False:
            print('Hello World!')
        'Identifies object whose security will be modified, and determines options available\n        to the end user'
        flags = SI_ADVANCED | SI_EDIT_ALL | SI_PAGE_TITLE | SI_RESET
        hinstance = 0
        servername = ''
        objectname = os.path.split(self.ServiceName)[1]
        pagetitle = 'Service Permissions for ' + self.ServiceName
        objecttype = IID_NULL
        return (flags, hinstance, servername, objectname, pagetitle, objecttype)

    def GetSecurity(self, requestedinfo, bdefault):
        if False:
            return 10
        'Requests the existing permissions for object'
        if bdefault:
            return win32security.SECURITY_DESCRIPTOR()
        else:
            return win32security.GetNamedSecurityInfo(self.ServiceName, win32security.SE_SERVICE, requestedinfo)

    def SetSecurity(self, requestedinfo, sd):
        if False:
            return 10
        'Applies permissions to the object'
        owner = sd.GetSecurityDescriptorOwner()
        group = sd.GetSecurityDescriptorGroup()
        dacl = sd.GetSecurityDescriptorDacl()
        sacl = sd.GetSecurityDescriptorSacl()
        win32security.SetNamedSecurityInfo(self.ServiceName, win32security.SE_SERVICE, requestedinfo, owner, group, dacl, sacl)

    def GetAccessRights(self, objecttype, flags):
        if False:
            return 10
        'Returns a tuple of (AccessRights, DefaultAccess), where AccessRights is a sequence of tuples representing\n        SI_ACCESS structs, containing (guid, access mask, Name, flags). DefaultAccess indicates which of the\n        AccessRights will be used initially when a new ACE is added (zero based).\n        Flags can contain SI_ACCESS_SPECIFIC,SI_ACCESS_GENERAL,SI_ACCESS_CONTAINER,SI_ACCESS_PROPERTY,\n              CONTAINER_INHERIT_ACE,INHERIT_ONLY_ACE,OBJECT_INHERIT_ACE\n        '
        if objecttype is not None and objecttype != IID_NULL:
            raise NotImplementedError('Object type is not supported')
        accessrights = [(IID_NULL, win32service.SERVICE_ALL_ACCESS, 'Full control', SI_ACCESS_GENERAL), (IID_NULL, SERVICE_GENERIC_READ, 'Generic read', SI_ACCESS_GENERAL), (IID_NULL, SERVICE_GENERIC_WRITE, 'Generic write', SI_ACCESS_GENERAL), (IID_NULL, SERVICE_GENERIC_EXECUTE, 'Start/Stop/Pause service', SI_ACCESS_GENERAL), (IID_NULL, READ_CONTROL, 'Read Permissions', SI_ACCESS_GENERAL), (IID_NULL, WRITE_DAC, 'Change permissions', SI_ACCESS_GENERAL), (IID_NULL, WRITE_OWNER, 'Change owner', SI_ACCESS_GENERAL), (IID_NULL, win32con.DELETE, 'Delete service', SI_ACCESS_GENERAL), (IID_NULL, win32service.SERVICE_START, 'Start service', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_STOP, 'Stop service', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_PAUSE_CONTINUE, 'Pause/unpause service', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_USER_DEFINED_CONTROL, 'Execute user defined operations', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_QUERY_CONFIG, 'Read configuration', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_CHANGE_CONFIG, 'Change configuration', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_ENUMERATE_DEPENDENTS, 'List dependent services', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_QUERY_STATUS, 'Query status', SI_ACCESS_SPECIFIC), (IID_NULL, win32service.SERVICE_INTERROGATE, 'Query status (immediate)', SI_ACCESS_SPECIFIC)]
        return (accessrights, 0)

    def MapGeneric(self, guid, aceflags, mask):
        if False:
            for i in range(10):
                print('nop')
        'Converts generic access rights to specific rights.'
        return win32security.MapGenericMask(mask, (SERVICE_GENERIC_READ, SERVICE_GENERIC_WRITE, SERVICE_GENERIC_EXECUTE, win32service.SERVICE_ALL_ACCESS))

    def GetInheritTypes(self):
        if False:
            i = 10
            return i + 15
        "Specifies which types of ACE inheritance are supported.\n        Services don't use any inheritance\n        "
        return ((IID_NULL, 0, 'Only current object'),)

    def PropertySheetPageCallback(self, hwnd, msg, pagetype):
        if False:
            print('Hello World!')
        'Invoked each time a property sheet page is created or destroyed.'
        return None

    def EditSecurity(self, owner_hwnd=0):
        if False:
            return 10
        'Creates an ACL editor dialog based on parameters returned by interface methods'
        isi = pythoncom.WrapObject(self, authorization.IID_ISecurityInformation, pythoncom.IID_IUnknown)
        authorization.EditSecurity(owner_hwnd, isi)
if __name__ == '__main__':
    scm = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_ENUMERATE_SERVICE)
    svcs = win32service.EnumServicesStatus(scm)
    win32service.CloseServiceHandle(scm)
    si = ServiceSecurity(svcs[0][0])
    si.EditSecurity()