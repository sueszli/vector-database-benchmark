import os, sys, glob, shutil, time
import winreg as winreg
import tempfile
tee_f = open(os.path.join(tempfile.gettempdir(), 'pywin32_postinstall.log'), 'w')

class Tee:

    def __init__(self, file):
        if False:
            print('Hello World!')
        self.f = file

    def write(self, what):
        if False:
            print('Hello World!')
        if self.f is not None:
            try:
                self.f.write(what.replace('\n', '\r\n'))
            except IOError:
                pass
        tee_f.write(what)

    def flush(self):
        if False:
            print('Hello World!')
        if self.f is not None:
            try:
                self.f.flush()
            except IOError:
                pass
        tee_f.flush()
if sys.stdout is None:
    sys.stdout = sys.stderr
sys.stderr = Tee(sys.stderr)
sys.stdout = Tee(sys.stdout)
com_modules = [('win32com.servers.interp', 'Interpreter'), ('win32com.servers.dictionary', 'DictionaryPolicy'), ('win32com.axscript.client.pyscript', 'PyScript')]
silent = 0
verbose = 1
ver_string = '%d.%d' % (sys.version_info[0], sys.version_info[1])
root_key_name = 'Software\\Python\\PythonCore\\' + ver_string
try:
    file_created
    is_bdist_wininst = True
except NameError:
    is_bdist_wininst = False

    def file_created(file):
        if False:
            print('Hello World!')
        pass

    def directory_created(directory):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_root_hkey():
        if False:
            while True:
                i = 10
        try:
            winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_key_name, 0, winreg.KEY_CREATE_SUB_KEY)
            return winreg.HKEY_LOCAL_MACHINE
        except OSError as details:
            return winreg.HKEY_CURRENT_USER
try:
    create_shortcut
except NameError:

    def create_shortcut(path, description, filename, arguments='', workdir='', iconpath='', iconindex=0):
        if False:
            while True:
                i = 10
        import pythoncom
        from win32com.shell import shell, shellcon
        ilink = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
        ilink.SetPath(path)
        ilink.SetDescription(description)
        if arguments:
            ilink.SetArguments(arguments)
        if workdir:
            ilink.SetWorkingDirectory(workdir)
        if iconpath or iconindex:
            ilink.SetIconLocation(iconpath, iconindex)
        ipf = ilink.QueryInterface(pythoncom.IID_IPersistFile)
        ipf.Save(filename, 0)

    def get_special_folder_path(path_name):
        if False:
            print('Hello World!')
        import pythoncom
        from win32com.shell import shell, shellcon
        for maybe in '\n            CSIDL_COMMON_STARTMENU CSIDL_STARTMENU CSIDL_COMMON_APPDATA\n            CSIDL_LOCAL_APPDATA CSIDL_APPDATA CSIDL_COMMON_DESKTOPDIRECTORY\n            CSIDL_DESKTOPDIRECTORY CSIDL_COMMON_STARTUP CSIDL_STARTUP\n            CSIDL_COMMON_PROGRAMS CSIDL_PROGRAMS CSIDL_PROGRAM_FILES_COMMON\n            CSIDL_PROGRAM_FILES CSIDL_FONTS'.split():
            if maybe == path_name:
                csidl = getattr(shellcon, maybe)
                return shell.SHGetSpecialFolderPath(0, csidl, False)
        raise ValueError('%s is an unknown path ID' % (path_name,))

def CopyTo(desc, src, dest):
    if False:
        print('Hello World!')
    import win32api, win32con
    while 1:
        try:
            win32api.CopyFile(src, dest, 0)
            return
        except win32api.error as details:
            if details.winerror == 5:
                raise
            if silent:
                raise
            tb = None
            full_desc = "Error %s\n\nIf you have any Python applications running, please close them now\nand select 'Retry'\n\n%s" % (desc, details.strerror)
            rc = win32api.MessageBox(0, full_desc, 'Installation Error', win32con.MB_ABORTRETRYIGNORE)
            if rc == win32con.IDABORT:
                raise
            elif rc == win32con.IDIGNORE:
                return

def LoadSystemModule(lib_dir, modname):
    if False:
        return 10
    import imp
    for suffix_item in imp.get_suffixes():
        if suffix_item[0] == '_d.pyd':
            suffix = '_d'
            break
    else:
        suffix = ''
    filename = '%s%d%d%s.dll' % (modname, sys.version_info[0], sys.version_info[1], suffix)
    filename = os.path.join(lib_dir, 'pywin32_system32', filename)
    mod = imp.load_dynamic(modname, filename)

def SetPyKeyVal(key_name, value_name, value):
    if False:
        while True:
            i = 10
    root_hkey = get_root_hkey()
    root_key = winreg.OpenKey(root_hkey, root_key_name)
    try:
        my_key = winreg.CreateKey(root_key, key_name)
        try:
            winreg.SetValueEx(my_key, value_name, 0, winreg.REG_SZ, value)
        finally:
            my_key.Close()
    finally:
        root_key.Close()
    if verbose:
        print('-> %s\\%s[%s]=%r' % (root_key_name, key_name, value_name, value))

def RegisterCOMObjects(register=1):
    if False:
        print('Hello World!')
    import win32com.server.register
    if register:
        func = win32com.server.register.RegisterClasses
    else:
        func = win32com.server.register.UnregisterClasses
    flags = {}
    if not verbose:
        flags['quiet'] = 1
    for (module, klass_name) in com_modules:
        __import__(module)
        mod = sys.modules[module]
        flags['finalize_register'] = getattr(mod, 'DllRegisterServer', None)
        flags['finalize_unregister'] = getattr(mod, 'DllUnregisterServer', None)
        klass = getattr(mod, klass_name)
        func(klass, **flags)

def RegisterPythonwin(register=True):
    if False:
        return 10
    " Add (or remove) Pythonwin to context menu for python scripts.\n        ??? Should probably also add Edit command for pys files also.\n        Also need to remove these keys on uninstall, but there's no function\n            like file_created to add registry entries to uninstall log ???\n    "
    import os, distutils.sysconfig
    lib_dir = distutils.sysconfig.get_python_lib(plat_specific=1)
    classes_root = get_root_hkey()
    pythonwin_exe = os.path.join(lib_dir, 'Pythonwin', 'Pythonwin.exe')
    pythonwin_edit_command = pythonwin_exe + ' /edit "%1"'
    keys_vals = [('Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\Pythonwin.exe', '', pythonwin_exe), ('Software\\Classes\\Python.File\\shell\\Edit with Pythonwin', 'command', pythonwin_edit_command), ('Software\\Classes\\Python.NoConFile\\shell\\Edit with Pythonwin', 'command', pythonwin_edit_command)]
    try:
        if register:
            for (key, sub_key, val) in keys_vals:
                hkey = winreg.CreateKey(classes_root, key)
                if sub_key:
                    hkey = winreg.CreateKey(hkey, sub_key)
                winreg.SetValueEx(hkey, None, 0, winreg.REG_SZ, val)
                hkey.Close()
        else:
            for (key, sub_key, val) in keys_vals:
                try:
                    winreg.DeleteKey(classes_root, key)
                except OSError as why:
                    winerror = getattr(why, 'winerror', why.errno)
                    if winerror != 2:
                        raise
    finally:
        from win32com.shell import shell, shellcon
        shell.SHChangeNotify(shellcon.SHCNE_ASSOCCHANGED, shellcon.SHCNF_IDLIST, None, None)

def get_shortcuts_folder():
    if False:
        for i in range(10):
            print('nop')
    if get_root_hkey() == winreg.HKEY_LOCAL_MACHINE:
        try:
            fldr = get_special_folder_path('CSIDL_COMMON_PROGRAMS')
        except OSError:
            fldr = get_special_folder_path('CSIDL_PROGRAMS')
    else:
        fldr = get_special_folder_path('CSIDL_PROGRAMS')
    try:
        install_group = winreg.QueryValue(get_root_hkey(), root_key_name + '\\InstallPath\\InstallGroup')
    except OSError:
        vi = sys.version_info
        install_group = 'Python %d.%d' % (vi[0], vi[1])
    return os.path.join(fldr, install_group)

def get_system_dir():
    if False:
        return 10
    import win32api
    try:
        import pythoncom
        import win32process
        from win32com.shell import shell, shellcon
        try:
            if win32process.IsWow64Process():
                return shell.SHGetSpecialFolderPath(0, shellcon.CSIDL_SYSTEMX86)
            return shell.SHGetSpecialFolderPath(0, shellcon.CSIDL_SYSTEM)
        except (pythoncom.com_error, win32process.error):
            return win32api.GetSystemDirectory()
    except ImportError:
        return win32api.GetSystemDirectory()

def fixup_dbi():
    if False:
        return 10
    import win32api, win32con
    pyd_name = os.path.join(os.path.dirname(win32api.__file__), 'dbi.pyd')
    pyd_d_name = os.path.join(os.path.dirname(win32api.__file__), 'dbi_d.pyd')
    py_name = os.path.join(os.path.dirname(win32con.__file__), 'dbi.py')
    for this_pyd in (pyd_name, pyd_d_name):
        this_dest = this_pyd + '.old'
        if os.path.isfile(this_pyd) and os.path.isfile(py_name):
            try:
                if os.path.isfile(this_dest):
                    print("Old dbi '%s' already exists - deleting '%s'" % (this_dest, this_pyd))
                    os.remove(this_pyd)
                else:
                    os.rename(this_pyd, this_dest)
                    print("renamed '%s'->'%s.old'" % (this_pyd, this_pyd))
                    file_created(this_pyd + '.old')
            except os.error as exc:
                print("FAILED to rename '%s': %s" % (this_pyd, exc))

def install():
    if False:
        while True:
            i = 10
    import distutils.sysconfig
    import traceback
    lib_dir = distutils.sysconfig.get_python_lib(plat_specific=1)
    if os.path.isfile(os.path.join(sys.prefix, 'pywin32.pth')):
        os.unlink(os.path.join(sys.prefix, 'pywin32.pth'))
    for name in 'win32 win32\\lib Pythonwin'.split():
        sys.path.append(os.path.join(lib_dir, name))
    for name in 'pythoncom pywintypes'.split():
        keyname = 'Software\\Python\\PythonCore\\' + sys.winver + '\\Modules\\' + name
        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                winreg.DeleteKey(root, keyname + '\\Debug')
            except WindowsError:
                pass
            try:
                winreg.DeleteKey(root, keyname)
            except WindowsError:
                pass
    LoadSystemModule(lib_dir, 'pywintypes')
    LoadSystemModule(lib_dir, 'pythoncom')
    import win32api
    files = glob.glob(os.path.join(lib_dir, 'pywin32_system32\\*.*'))
    if not files:
        raise RuntimeError('No system files to copy!!')
    for dest_dir in [get_system_dir(), sys.prefix]:
        worked = 0
        try:
            for fname in files:
                base = os.path.basename(fname)
                dst = os.path.join(dest_dir, base)
                CopyTo('installing %s' % base, fname, dst)
                if verbose:
                    print('Copied %s to %s' % (base, dst))
                file_created(dst)
                worked = 1
                if dest_dir != sys.prefix:
                    bad_fname = os.path.join(sys.prefix, base)
                    if os.path.exists(bad_fname):
                        os.unlink(bad_fname)
            if worked:
                break
        except win32api.error as details:
            if details.winerror == 5:
                if os.path.exists(dst):
                    msg = "The file '%s' exists, but can not be replaced due to insufficient permissions.  You must reinstall this software as an Administrator" % dst
                    print(msg)
                    raise RuntimeError(msg)
                continue
            raise
    else:
        raise RuntimeError("You don't have enough permissions to install the system files")
    pywin_dir = os.path.join(lib_dir, 'Pythonwin', 'pywin')
    for fname in glob.glob(os.path.join(pywin_dir, '*.cfg')):
        file_created(fname[:-1] + 'c')
    try:
        try:
            RegisterCOMObjects()
        except win32api.error as details:
            if details.winerror != 5:
                raise
            print('You do not have the permissions to install COM objects.')
            print('The sample COM objects were not registered.')
    except:
        print('FAILED to register the Python COM objects')
        traceback.print_exc()
    winreg.CreateKey(get_root_hkey(), root_key_name)
    chm_file = os.path.join(lib_dir, 'PyWin32.chm')
    if os.path.isfile(chm_file):
        SetPyKeyVal('Help', None, None)
        SetPyKeyVal('Help\\Pythonwin Reference', None, chm_file)
    else:
        print('NOTE: PyWin32.chm can not be located, so has not been registered')
    fixup_dbi()
    try:
        RegisterPythonwin()
    except:
        print('Failed to register pythonwin as editor')
        traceback.print_exc()
    else:
        if verbose:
            print('Pythonwin has been registered in context menu')
    make_dir = os.path.join(lib_dir, 'win32com', 'gen_py')
    if not os.path.isdir(make_dir):
        if verbose:
            print('Creating directory', make_dir)
        directory_created(make_dir)
        os.mkdir(make_dir)
    try:
        fldr = get_shortcuts_folder()
        if os.path.isdir(fldr):
            dst = os.path.join(fldr, 'PythonWin.lnk')
            create_shortcut(os.path.join(lib_dir, 'Pythonwin\\Pythonwin.exe'), 'The Pythonwin IDE', dst, '', sys.prefix)
            file_created(dst)
            if verbose:
                print('Shortcut for Pythonwin created')
            dst = os.path.join(fldr, 'Python for Windows Documentation.lnk')
            doc = 'Documentation for the PyWin32 extensions'
            create_shortcut(chm_file, doc, dst)
            file_created(dst)
            if verbose:
                print('Shortcut to documentation created')
        elif verbose:
            print("Can't install shortcuts - %r is not a folder" % (fldr,))
    except Exception as details:
        print(details)
    try:
        import win32com.client
    except ImportError:
        pass
    print('The pywin32 extensions were successfully installed.')

def uninstall():
    if False:
        print('Hello World!')
    import distutils.sysconfig
    lib_dir = distutils.sysconfig.get_python_lib(plat_specific=1)
    LoadSystemModule(lib_dir, 'pywintypes')
    LoadSystemModule(lib_dir, 'pythoncom')
    try:
        RegisterCOMObjects(False)
    except Exception as why:
        print('Failed to unregister COM objects:', why)
    try:
        RegisterPythonwin(False)
    except Exception as why:
        print('Failed to unregister Pythonwin:', why)
    else:
        if verbose:
            print('Unregistered Pythonwin')
    try:
        gen_dir = os.path.join(lib_dir, 'win32com', 'gen_py')
        if os.path.isdir(gen_dir):
            shutil.rmtree(gen_dir)
            if verbose:
                print('Removed directory', gen_dir)
        pywin_dir = os.path.join(lib_dir, 'Pythonwin', 'pywin')
        for fname in glob.glob(os.path.join(pywin_dir, '*.cfc')):
            os.remove(fname)
        try:
            os.remove(os.path.join(lib_dir, 'win32', 'dbi.pyd.old'))
        except os.error:
            pass
        try:
            os.remove(os.path.join(lib_dir, 'win32', 'dbi_d.pyd.old'))
        except os.error:
            pass
    except Exception as why:
        print('Failed to remove misc files:', why)
    try:
        fldr = get_shortcuts_folder()
        for link in ('PythonWin.lnk', 'Python for Windows Documentation.lnk'):
            fqlink = os.path.join(fldr, link)
            if os.path.isfile(fqlink):
                os.remove(fqlink)
                if verbose:
                    print('Removed', link)
    except Exception as why:
        print('Failed to remove shortcuts:', why)
    files = glob.glob(os.path.join(lib_dir, 'pywin32_system32\\*.*'))
    try:
        for dest_dir in [get_system_dir(), sys.prefix]:
            worked = 0
            for fname in files:
                base = os.path.basename(fname)
                dst = os.path.join(dest_dir, base)
                if os.path.isfile(dst):
                    try:
                        os.remove(dst)
                        worked = 1
                        if verbose:
                            print('Removed file %s' % dst)
                    except Exception:
                        print('FAILED to remove', dst)
            if worked:
                break
    except Exception as why:
        print('FAILED to remove system files:', why)

def usage():
    if False:
        while True:
            i = 10
    msg = '%s: A post-install script for the pywin32 extensions.\n\nTypical usage:\n\n> python pywin32_postinstall.py -install\n\nIf you installed pywin32 via a .exe installer, this should be run\nautomatically after installation, but if it fails you can run it again.\n\nIf you installed pywin32 via PIP, you almost certainly need to run this to\nsetup the environment correctly.\n\nExecute with script with a \'-install\' parameter, to ensure the environment\nis setup correctly.\n\nOptions:\n  -install  : Configure the Python environment correctly for pywin32.\n  -remove   : Try and remove everything that was installed or copied.\n  -wait pid : Wait for the specified process to terminate before starting.\n  -silent   : Don\'t display the "Abort/Retry/Ignore" dialog for files in use.\n  -quiet    : Don\'t display progress messages.\n'
    print(msg.strip() % os.path.basename(sys.argv[0]))
if __name__ == '__main__':
    if len(sys.argv) == 1:
        usage()
        sys.exit(1)
    arg_index = 1
    while arg_index < len(sys.argv):
        arg = sys.argv[arg_index]
        if arg == '-wait':
            arg_index += 1
            pid = int(sys.argv[arg_index])
            try:
                os.waitpid(pid, 0)
            except AttributeError:
                time.sleep(3)
            except os.error:
                pass
        elif arg == '-install':
            install()
        elif arg == '-silent':
            silent = 1
        elif arg == '-quiet':
            verbose = 0
        elif arg == '-remove':
            if not is_bdist_wininst:
                uninstall()
        else:
            print('Unknown option:', arg)
            usage()
            sys.exit(0)
        arg_index += 1