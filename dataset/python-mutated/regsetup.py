class error(Exception):
    pass
import sys

def FileExists(fname):
    if False:
        while True:
            i = 10
    'Check if a file exists.  Returns true or false.'
    import os
    try:
        os.stat(fname)
        return 1
    except OSError as details:
        return 0

def IsPackageDir(path, packageName, knownFileName):
    if False:
        for i in range(10):
            print('nop')
    'Given a path, a ni package name, and possibly a known file name in\n    the root of the package, see if this path is good.\n    '
    import os
    if knownFileName is None:
        knownFileName = '.'
    return FileExists(os.path.join(os.path.join(path, packageName), knownFileName))

def IsDebug():
    if False:
        return 10
    'Return "_d" if we\'re running a debug version.\n\n    This is to be used within DLL names when locating them.\n    '
    import importlib.machinery
    return '_d' if '_d.pyd' in importlib.machinery.EXTENSION_SUFFIXES else ''

def FindPackagePath(packageName, knownFileName, searchPaths):
    if False:
        while True:
            i = 10
    'Find a package.\n\n    Given a ni style package name, check the package is registered.\n\n    First place looked is the registry for an existing entry.  Then\n    the searchPaths are searched.\n    '
    import os
    import regutil
    pathLook = regutil.GetRegisteredNamedPath(packageName)
    if pathLook and IsPackageDir(pathLook, packageName, knownFileName):
        return (pathLook, None)
    for pathLook in searchPaths:
        if IsPackageDir(pathLook, packageName, knownFileName):
            ret = os.path.abspath(pathLook)
            return (ret, ret)
    raise error('The package %s can not be located' % packageName)

def FindHelpPath(helpFile, helpDesc, searchPaths):
    if False:
        while True:
            i = 10
    import os
    import win32api
    import win32con
    try:
        key = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, 'Software\\Microsoft\\Windows\\Help', 0, win32con.KEY_ALL_ACCESS)
        try:
            try:
                path = win32api.RegQueryValueEx(key, helpDesc)[0]
                if FileExists(os.path.join(path, helpFile)):
                    return os.path.abspath(path)
            except win32api.error:
                pass
        finally:
            key.Close()
    except win32api.error:
        pass
    for pathLook in searchPaths:
        if FileExists(os.path.join(pathLook, helpFile)):
            return os.path.abspath(pathLook)
        pathLook = os.path.join(pathLook, 'Help')
        if FileExists(os.path.join(pathLook, helpFile)):
            return os.path.abspath(pathLook)
    raise error('The help file %s can not be located' % helpFile)

def FindAppPath(appName, knownFileName, searchPaths):
    if False:
        for i in range(10):
            print('nop')
    'Find an application.\n\n    First place looked is the registry for an existing entry.  Then\n    the searchPaths are searched.\n    '
    import os
    import regutil
    regPath = regutil.GetRegisteredNamedPath(appName)
    if regPath:
        pathLook = regPath.split(';')[0]
    if regPath and FileExists(os.path.join(pathLook, knownFileName)):
        return None
    for pathLook in searchPaths:
        if FileExists(os.path.join(pathLook, knownFileName)):
            return os.path.abspath(pathLook)
    raise error(f'The file {knownFileName} can not be located for application {appName}')

def FindPythonExe(exeAlias, possibleRealNames, searchPaths):
    if False:
        for i in range(10):
            print('nop')
    "Find an exe.\n\n    Returns the full path to the .exe, and a boolean indicating if the current\n    registered entry is OK.  We don't trust the already registered version even\n    if it exists - it may be wrong (ie, for a different Python version)\n    "
    import os
    import sys
    import regutil
    import win32api
    if possibleRealNames is None:
        possibleRealNames = exeAlias
    found = os.path.join(sys.prefix, possibleRealNames)
    if not FileExists(found):
        if '64 bit' in sys.version:
            found = os.path.join(sys.prefix, 'PCBuild', 'amd64', possibleRealNames)
        else:
            found = os.path.join(sys.prefix, 'PCBuild', possibleRealNames)
    if not FileExists(found):
        found = LocateFileName(possibleRealNames, searchPaths)
    registered_ok = 0
    try:
        registered = win32api.RegQueryValue(regutil.GetRootKey(), regutil.GetAppPathsKey() + '\\' + exeAlias)
        registered_ok = found == registered
    except win32api.error:
        pass
    return (found, registered_ok)

def QuotedFileName(fname):
    if False:
        return 10
    'Given a filename, return a quoted version if necessary'
    import regutil
    try:
        fname.index(' ')
        return '"%s"' % fname
    except ValueError:
        return fname

def LocateFileName(fileNamesString, searchPaths):
    if False:
        i = 10
        return i + 15
    'Locate a file name, anywhere on the search path.\n\n    If the file can not be located, prompt the user to find it for us\n    (using a common OpenFile dialog)\n\n    Raises KeyboardInterrupt if the user cancels.\n    '
    import os
    import regutil
    fileNames = fileNamesString.split(';')
    for path in searchPaths:
        for fileName in fileNames:
            try:
                retPath = os.path.join(path, fileName)
                os.stat(retPath)
                break
            except OSError:
                retPath = None
        if retPath:
            break
    else:
        fileName = fileNames[0]
        try:
            import win32con
            import win32ui
        except ImportError:
            raise error('Need to locate the file %s, but the win32ui module is not available\nPlease run the program again, passing as a parameter the path to this file.' % fileName)
        flags = win32con.OFN_FILEMUSTEXIST
        ext = os.path.splitext(fileName)[1]
        filter = f'Files of requested type (*{ext})|*{ext}||'
        dlg = win32ui.CreateFileDialog(1, None, fileName, flags, filter, None)
        dlg.SetOFNTitle('Locate ' + fileName)
        if dlg.DoModal() != win32con.IDOK:
            raise KeyboardInterrupt('User cancelled the process')
        retPath = dlg.GetPathName()
    return os.path.abspath(retPath)

def LocatePath(fileName, searchPaths):
    if False:
        while True:
            i = 10
    'Like LocateFileName, but returns a directory only.'
    import os
    return os.path.abspath(os.path.split(LocateFileName(fileName, searchPaths))[0])

def LocateOptionalPath(fileName, searchPaths):
    if False:
        for i in range(10):
            print('nop')
    'Like LocatePath, but returns None if the user cancels.'
    try:
        return LocatePath(fileName, searchPaths)
    except KeyboardInterrupt:
        return None

def LocateOptionalFileName(fileName, searchPaths=None):
    if False:
        while True:
            i = 10
    'Like LocateFileName, but returns None if the user cancels.'
    try:
        return LocateFileName(fileName, searchPaths)
    except KeyboardInterrupt:
        return None

def LocatePythonCore(searchPaths):
    if False:
        while True:
            i = 10
    'Locate and validate the core Python directories.  Returns a list\n    of paths that should be used as the core (ie, un-named) portion of\n    the Python path.\n    '
    import os
    import regutil
    currentPath = regutil.GetRegisteredNamedPath(None)
    if currentPath:
        presearchPaths = currentPath.split(';')
    else:
        presearchPaths = [os.path.abspath('.')]
    libPath = None
    for path in presearchPaths:
        if FileExists(os.path.join(path, 'os.py')):
            libPath = path
            break
    if libPath is None and searchPaths is not None:
        libPath = LocatePath('os.py', searchPaths)
    if libPath is None:
        raise error('The core Python library could not be located.')
    corePath = None
    suffix = IsDebug()
    for path in presearchPaths:
        if FileExists(os.path.join(path, 'unicodedata%s.pyd' % suffix)):
            corePath = path
            break
    if corePath is None and searchPaths is not None:
        corePath = LocatePath('unicodedata%s.pyd' % suffix, searchPaths)
    if corePath is None:
        raise error('The core Python path could not be located.')
    installPath = os.path.abspath(os.path.join(libPath, '..'))
    return (installPath, [libPath, corePath])

def FindRegisterPackage(packageName, knownFile, searchPaths, registryAppName=None):
    if False:
        return 10
    'Find and Register a package.\n\n    Assumes the core registry setup correctly.\n\n    In addition, if the location located by the package is already\n    in the **core** path, then an entry is registered, but no path.\n    (no other paths are checked, as the application whose path was used\n    may later be uninstalled.  This should not happen with the core)\n    '
    import regutil
    if not packageName:
        raise error('A package name must be supplied')
    corePaths = regutil.GetRegisteredNamedPath(None).split(';')
    if not searchPaths:
        searchPaths = corePaths
    registryAppName = registryAppName or packageName
    try:
        (pathLook, pathAdd) = FindPackagePath(packageName, knownFile, searchPaths)
        if pathAdd is not None:
            if pathAdd in corePaths:
                pathAdd = ''
            regutil.RegisterNamedPath(registryAppName, pathAdd)
        return pathLook
    except error as details:
        print(f'*** The {packageName} package could not be registered - {details}')
        print('*** Please ensure you have passed the correct paths on the command line.')
        print('*** - For packages, you should pass a path to the packages parent directory,')
        print('*** - and not the package directory itself...')

def FindRegisterApp(appName, knownFiles, searchPaths):
    if False:
        i = 10
        return i + 15
    'Find and Register a package.\n\n    Assumes the core registry setup correctly.\n\n    '
    import regutil
    if isinstance(knownFiles, str):
        knownFiles = [knownFiles]
    paths = []
    try:
        for knownFile in knownFiles:
            pathLook = FindAppPath(appName, knownFile, searchPaths)
            if pathLook:
                paths.append(pathLook)
    except error as details:
        print('*** ', details)
        return
    regutil.RegisterNamedPath(appName, ';'.join(paths))

def FindRegisterPythonExe(exeAlias, searchPaths, actualFileNames=None):
    if False:
        for i in range(10):
            print('nop')
    'Find and Register a Python exe (not necessarily *the* python.exe)\n\n    Assumes the core registry setup correctly.\n    '
    import regutil
    (fname, ok) = FindPythonExe(exeAlias, actualFileNames, searchPaths)
    if not ok:
        regutil.RegisterPythonExe(fname, exeAlias)
    return fname

def FindRegisterHelpFile(helpFile, searchPaths, helpDesc=None):
    if False:
        return 10
    import regutil
    try:
        pathLook = FindHelpPath(helpFile, helpDesc, searchPaths)
    except error as details:
        print('*** ', details)
        return
    regutil.RegisterHelpFile(helpFile, pathLook, helpDesc)

def SetupCore(searchPaths):
    if False:
        return 10
    'Setup the core Python information in the registry.\n\n    This function makes no assumptions about the current state of sys.path.\n\n    After this function has completed, you should have access to the standard\n    Python library, and the standard Win32 extensions\n    '
    import sys
    for path in searchPaths:
        sys.path.append(path)
    import os
    import regutil
    import win32api
    import win32con
    (installPath, corePaths) = LocatePythonCore(searchPaths)
    print(corePaths)
    regutil.RegisterNamedPath(None, ';'.join(corePaths))
    hKey = win32api.RegCreateKey(regutil.GetRootKey(), regutil.BuildDefaultPythonKey())
    try:
        win32api.RegSetValue(hKey, 'InstallPath', win32con.REG_SZ, installPath)
    finally:
        win32api.RegCloseKey(hKey)
    win32paths = os.path.abspath(os.path.split(win32api.__file__)[0]) + ';' + os.path.abspath(os.path.split(LocateFileName('win32con.py;win32con.pyc', sys.path))[0])
    check = os.path.join(sys.prefix, 'PCBuild')
    if '64 bit' in sys.version:
        check = os.path.join(check, 'amd64')
    if os.path.isdir(check):
        regutil.RegisterNamedPath('PCBuild', check)

def RegisterShellInfo(searchPaths):
    if False:
        return 10
    'Registers key parts of the Python installation with the Windows Shell.\n\n    Assumes a valid, minimal Python installation exists\n    (ie, SetupCore() has been previously successfully run)\n    '
    import regutil
    import win32con
    suffix = IsDebug()
    exePath = FindRegisterPythonExe('Python%s.exe' % suffix, searchPaths)
    regutil.SetRegistryDefaultValue('.py', 'Python.File', win32con.HKEY_CLASSES_ROOT)
    regutil.RegisterShellCommand('Open', QuotedFileName(exePath) + ' "%1" %*', '&Run')
    regutil.SetRegistryDefaultValue('Python.File\\DefaultIcon', '%s,0' % exePath, win32con.HKEY_CLASSES_ROOT)
    FindRegisterHelpFile('Python.hlp', searchPaths, 'Main Python Documentation')
    FindRegisterHelpFile('ActivePython.chm', searchPaths, 'Main Python Documentation')
usage = 'regsetup.py - Setup/maintain the registry for Python apps.\n\nRun without options, (but possibly search paths) to repair a totally broken\npython registry setup.  This should allow other options to work.\n\nUsage:   %s [options ...] paths ...\n-p packageName  -- Find and register a package.  Looks in the paths for\n                   a sub-directory with the name of the package, and\n                   adds a path entry for the package.\n-a appName      -- Unconditionally add an application name to the path.\n                   A new path entry is create with the app name, and the\n                   paths specified are added to the registry.\n-c              -- Add the specified paths to the core Pythonpath.\n                   If a path appears on the core path, and a package also\n                   needs that same path, the package will not bother\n                   registering it.  Therefore, By adding paths to the\n                   core path, you can avoid packages re-registering the same path.\n-m filename     -- Find and register the specific file name as a module.\n                   Do not include a path on the filename!\n--shell         -- Register everything with the Win95/NT shell.\n--upackage name -- Unregister the package\n--uapp name     -- Unregister the app (identical to --upackage)\n--umodule name  -- Unregister the module\n\n--description   -- Print a description of the usage.\n--examples      -- Print examples of usage.\n' % sys.argv[0]
description = 'If no options are processed, the program attempts to validate and set\nthe standard Python path to the point where the standard library is\navailable.  This can be handy if you move Python to a new drive/sub-directory,\nin which case most of the options would fail (as they need at least string.py,\nos.py etc to function.)\nRunning without options should repair Python well enough to run with\nthe other options.\n\npaths are search paths that the program will use to seek out a file.\nFor example, when registering the core Python, you may wish to\nprovide paths to non-standard places to look for the Python help files,\nlibrary files, etc.\n\nSee also the "regcheck.py" utility which will check and dump the contents\nof the registry.\n'
examples = '\nExamples:\n"regsetup c:\\wierd\\spot\\1 c:\\wierd\\spot\\2"\nAttempts to setup the core Python.  Looks in some standard places,\nas well as the 2 wierd spots to locate the core Python files (eg, Python.exe,\npythonXX.dll, the standard library and Win32 Extensions.\n\n"regsetup -a myappname . .\\subdir"\nRegisters a new Pythonpath entry named myappname, with "C:\\I\\AM\\HERE" and\n"C:\\I\\AM\\HERE\\subdir" added to the path (ie, all args are converted to\nabsolute paths)\n\n"regsetup -c c:\\my\\python\\files"\nUnconditionally add "c:\\my\\python\\files" to the \'core\' Python path.\n\n"regsetup -m some.pyd \\windows\\system"\nRegister the module some.pyd in \\windows\\system as a registered\nmodule.  This will allow some.pyd to be imported, even though the\nwindows system directory is not (usually!) on the Python Path.\n\n"regsetup --umodule some"\nUnregister the module "some".  This means normal import rules then apply\nfor that module.\n'
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['/?', '-?', '-help', '-h']:
        print(usage)
    elif len(sys.argv) == 1 or not sys.argv[1][0] in ['/', '-']:
        searchPath = sys.path[:]
        for arg in sys.argv[1:]:
            searchPath.append(arg)
        searchPath.append('..\\Build')
        searchPath.append('..\\Lib')
        searchPath.append('..')
        searchPath.append('..\\..')
        searchPath.append('..\\..\\lib')
        searchPath.append('..\\build')
        if '64 bit' in sys.version:
            searchPath.append('..\\..\\pcbuild\\amd64')
        else:
            searchPath.append('..\\..\\pcbuild')
        print('Attempting to setup/repair the Python core')
        SetupCore(searchPath)
        RegisterShellInfo(searchPath)
        FindRegisterHelpFile('PyWin32.chm', searchPath, 'Pythonwin Reference')
        print('Registration complete - checking the registry...')
        import regcheck
        regcheck.CheckRegistry()
    else:
        searchPaths = []
        import getopt
        (opts, args) = getopt.getopt(sys.argv[1:], 'p:a:m:c', ['shell', 'upackage=', 'uapp=', 'umodule=', 'description', 'examples'])
        for arg in args:
            searchPaths.append(arg)
        for (o, a) in opts:
            if o == '--description':
                print(description)
            if o == '--examples':
                print(examples)
            if o == '--shell':
                print('Registering the Python core.')
                RegisterShellInfo(searchPaths)
            if o == '-p':
                print('Registering package', a)
                FindRegisterPackage(a, None, searchPaths)
            if o in ['--upackage', '--uapp']:
                import regutil
                print('Unregistering application/package', a)
                regutil.UnregisterNamedPath(a)
            if o == '-a':
                import regutil
                path = ';'.join(searchPaths)
                print('Registering application', a, 'to path', path)
                regutil.RegisterNamedPath(a, path)
            if o == '-c':
                if not len(searchPaths):
                    raise error('-c option must provide at least one additional path')
                import regutil
                currentPaths = regutil.GetRegisteredNamedPath(None).split(';')
                oldLen = len(currentPaths)
                for newPath in searchPaths:
                    if newPath not in currentPaths:
                        currentPaths.append(newPath)
                if len(currentPaths) != oldLen:
                    print('Registering %d new core paths' % (len(currentPaths) - oldLen))
                    regutil.RegisterNamedPath(None, ';'.join(currentPaths))
                else:
                    print('All specified paths are already registered.')