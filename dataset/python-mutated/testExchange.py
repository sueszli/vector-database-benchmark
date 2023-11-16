import os
import pythoncom
from win32com.client import constants, gencache
ammodule = None

def GetDefaultProfileName():
    if False:
        while True:
            i = 10
    import win32api
    import win32con
    try:
        key = win32api.RegOpenKey(win32con.HKEY_CURRENT_USER, 'Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows Messaging Subsystem\\Profiles')
        try:
            return win32api.RegQueryValueEx(key, 'DefaultProfile')[0]
        finally:
            key.Close()
    except win32api.error:
        return None

def DumpFolder(folder, indent=0):
    if False:
        print('Hello World!')
    print(' ' * indent, folder.Name)
    folders = folder.Folders
    folder = folders.GetFirst()
    while folder:
        DumpFolder(folder, indent + 1)
        folder = folders.GetNext()

def DumpFolders(session):
    if False:
        return 10
    try:
        infostores = session.InfoStores
    except AttributeError:
        store = session.DefaultStore
        folder = store.GetRootFolder()
        DumpFolder(folder)
        return
    print(infostores)
    print('There are %d infostores' % infostores.Count)
    for i in range(infostores.Count):
        infostore = infostores[i + 1]
        print('Infostore = ', infostore.Name)
        try:
            folder = infostore.RootFolder
        except pythoncom.com_error as details:
            (hr, msg, exc, arg) = details
            if exc and exc[-1] == -2147221219:
                print('This info store is currently not available')
                continue
        DumpFolder(folder)
PropTagsById = {}
if ammodule:
    for (name, val) in ammodule.constants.__dict__.items():
        PropTagsById[val] = name

def TestAddress(session):
    if False:
        for i in range(10):
            print('nop')
    pass

def TestUser(session):
    if False:
        while True:
            i = 10
    ae = session.CurrentUser
    fields = getattr(ae, 'Fields', [])
    print('User has %d fields' % len(fields))
    for f in range(len(fields)):
        field = fields[f + 1]
        try:
            id = PropTagsById[field.ID]
        except KeyError:
            id = field.ID
        print(f'{field.Name}/{id}={field.Value}')

def test():
    if False:
        while True:
            i = 10
    import win32com.client
    oldcwd = os.getcwd()
    try:
        session = gencache.EnsureDispatch('MAPI.Session')
        try:
            session.Logon(GetDefaultProfileName())
        except pythoncom.com_error as details:
            print('Could not log on to MAPI:', details)
            return
    except pythoncom.error:
        app = gencache.EnsureDispatch('Outlook.Application')
        session = app.Session
    try:
        TestUser(session)
        TestAddress(session)
        DumpFolders(session)
    finally:
        session.Logoff()
        os.chdir(oldcwd)
if __name__ == '__main__':
    from .util import CheckClean
    test()
    CheckClean()