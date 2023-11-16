import os
import sys
import traceback
import win32api
import win32ui
g_iniName = 'Mssccprj.scc'
g_sourceSafe = None

def FindVssProjectInfo(fullfname):
    if False:
        for i in range(10):
            print('nop')
    'Looks up the file system for an INI file describing the project.\n\n    Looking up the tree is for ni style packages.\n\n    Returns (projectName, pathToFileName) where pathToFileName contains\n    the path from the ini file to the actual file.\n    '
    (path, fnameonly) = os.path.split(fullfname)
    origPath = path
    project = ''
    retPaths = [fnameonly]
    while not project:
        iniName = os.path.join(path, g_iniName)
        database = win32api.GetProfileVal('Python', 'Database', '', iniName)
        project = win32api.GetProfileVal('Python', 'Project', '', iniName)
        if project:
            break
        (path, addpath) = os.path.split(path)
        if not addpath:
            break
        retPaths.insert(0, addpath)
    if not project:
        win32ui.MessageBox('%s\r\n\r\nThis directory is not configured for Python/VSS' % origPath)
        return
    return (project, '/'.join(retPaths), database)

def CheckoutFile(fileName):
    if False:
        print('Hello World!')
    global g_sourceSafe
    import pythoncom
    ok = 0
    try:
        import win32com.client
        import win32com.client.gencache
        mod = win32com.client.gencache.EnsureModule('{783CD4E0-9D54-11CF-B8EE-00608CC9A71F}', 0, 5, 0)
        if mod is None:
            win32ui.MessageBox('VSS does not appear to be installed.  The TypeInfo can not be created')
            return ok
        rc = FindVssProjectInfo(fileName)
        if rc is None:
            return
        (project, vssFname, database) = rc
        if g_sourceSafe is None:
            g_sourceSafe = win32com.client.Dispatch('SourceSafe')
            if not database:
                database = pythoncom.Missing
            g_sourceSafe.Open(database, pythoncom.Missing, pythoncom.Missing)
        item = g_sourceSafe.VSSItem(f'$/{project}/{vssFname}')
        item.Checkout(None, fileName)
        ok = 1
    except pythoncom.com_error as exc:
        win32ui.MessageBox(exc.strerror, 'Error checking out file')
    except:
        (typ, val, tb) = sys.exc_info()
        traceback.print_exc()
        win32ui.MessageBox(f'{str(typ)} - {str(val)}', 'Error checking out file')
        tb = None
    return ok