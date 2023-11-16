"""Create Spyder start menu and desktop entries"""
from __future__ import print_function
import os
import sys
import os.path as osp
import struct
import winreg
EWS = 'Edit with Spyder'
KEY_C = 'Software\\Classes\\%s'
KEY_C0 = KEY_C % 'Python.%sFile\\shell\\%s'
KEY_C1 = KEY_C0 + '\\command'
ver_string = '%d.%d' % (sys.version_info[0], sys.version_info[1])
root_key_name = 'Software\\Python\\PythonCore\\' + ver_string
try:
    file_created
    is_bdist_wininst = True
except NameError:
    is_bdist_wininst = False

    def file_created(file):
        if False:
            for i in range(10):
                print('nop')
        pass

    def directory_created(directory):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_root_hkey():
        if False:
            return 10
        try:
            winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_key_name, 0, winreg.KEY_CREATE_SUB_KEY)
            return winreg.HKEY_LOCAL_MACHINE
        except OSError:
            return winreg.HKEY_CURRENT_USER
try:
    create_shortcut
except NameError:

    def create_shortcut(path, description, filename, arguments='', workdir='', iconpath='', iconindex=0):
        if False:
            while True:
                i = 10
        try:
            import pythoncom
        except ImportError:
            print('pywin32 is required to run this script manually', file=sys.stderr)
            sys.exit(1)
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
            for i in range(10):
                print('nop')
        try:
            import pythoncom
        except ImportError:
            print('pywin32 is required to run this script manually', file=sys.stderr)
            sys.exit(1)
        from win32com.shell import shell, shellcon
        path_names = ['CSIDL_COMMON_STARTMENU', 'CSIDL_STARTMENU', 'CSIDL_COMMON_APPDATA', 'CSIDL_LOCAL_APPDATA', 'CSIDL_APPDATA', 'CSIDL_COMMON_DESKTOPDIRECTORY', 'CSIDL_DESKTOPDIRECTORY', 'CSIDL_COMMON_STARTUP', 'CSIDL_STARTUP', 'CSIDL_COMMON_PROGRAMS', 'CSIDL_PROGRAMS', 'CSIDL_PROGRAM_FILES_COMMON', 'CSIDL_PROGRAM_FILES', 'CSIDL_FONTS']
        for maybe in path_names:
            if maybe == path_name:
                csidl = getattr(shellcon, maybe)
                return shell.SHGetSpecialFolderPath(0, csidl, False)
        raise ValueError('%s is an unknown path ID' % (path_name,))

def install():
    if False:
        print('Hello World!')
    'Function executed when running the script with the -install switch'
    start_menu = osp.join(get_special_folder_path('CSIDL_PROGRAMS'), 'Spyder (Py%i.%i %i bit)' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8))
    if not osp.isdir(start_menu):
        os.mkdir(start_menu)
        directory_created(start_menu)
    python = osp.abspath(osp.join(sys.prefix, 'python.exe'))
    pythonw = osp.abspath(osp.join(sys.prefix, 'pythonw.exe'))
    script = osp.abspath(osp.join(sys.prefix, 'scripts', 'spyder'))
    if not osp.exists(script):
        script = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), 'spyder'))
    workdir = '%HOMEDRIVE%%HOMEPATH%'
    lib_dir = sysconfig.get_path('platlib')
    ico_dir = osp.join(lib_dir, 'spyder', 'windows')
    if not osp.isdir(ico_dir):
        ico_dir = osp.dirname(osp.abspath(__file__))
    desc = 'The Scientific Python Development Environment'
    fname = osp.join(start_menu, 'Spyder (full).lnk')
    create_shortcut(python, desc, fname, '"%s"' % script, workdir, osp.join(ico_dir, 'spyder.ico'))
    file_created(fname)
    fname = osp.join(start_menu, 'Spyder-Reset all settings.lnk')
    create_shortcut(python, 'Reset Spyder settings to defaults', fname, '"%s" --reset' % script, workdir)
    file_created(fname)
    current = True
    root = winreg.HKEY_CURRENT_USER if current else winreg.HKEY_LOCAL_MACHINE
    winreg.SetValueEx(winreg.CreateKey(root, KEY_C1 % ('', EWS)), '', 0, winreg.REG_SZ, '"%s" "%s\\Scripts\\spyder" "%%1"' % (pythonw, sys.prefix))
    winreg.SetValueEx(winreg.CreateKey(root, KEY_C1 % ('NoCon', EWS)), '', 0, winreg.REG_SZ, '"%s" "%s\\Scripts\\spyder" "%%1"' % (pythonw, sys.prefix))
    desktop_folder = get_special_folder_path('CSIDL_DESKTOPDIRECTORY')
    fname = osp.join(desktop_folder, 'Spyder.lnk')
    desc = 'The Scientific Python Development Environment'
    create_shortcut(pythonw, desc, fname, '"%s"' % script, workdir, osp.join(ico_dir, 'spyder.ico'))
    file_created(fname)

def remove():
    if False:
        i = 10
        return i + 15
    'Function executed when running the script with the -remove switch'
    current = True
    root = winreg.HKEY_CURRENT_USER if current else winreg.HKEY_LOCAL_MACHINE
    for key in (KEY_C1 % ('', EWS), KEY_C1 % ('NoCon', EWS), KEY_C0 % ('', EWS), KEY_C0 % ('NoCon', EWS)):
        try:
            winreg.DeleteKey(root, key)
        except WindowsError:
            pass
        else:
            if not is_bdist_wininst:
                print('Successfully removed Spyder shortcuts from Windows Explorer context menu.', file=sys.stdout)
    if not is_bdist_wininst:
        desktop_folder = get_special_folder_path('CSIDL_DESKTOPDIRECTORY')
        fname = osp.join(desktop_folder, 'Spyder.lnk')
        if osp.isfile(fname):
            try:
                os.remove(fname)
            except OSError:
                print('Failed to remove %s; you may be able to remove it manually.' % fname, file=sys.stderr)
            else:
                print('Successfully removed Spyder shortcuts from your desktop.', file=sys.stdout)
        start_menu = osp.join(get_special_folder_path('CSIDL_PROGRAMS'), 'Spyder (Py%i.%i %i bit)' % (sys.version_info[0], sys.version_info[1], struct.calcsize('P') * 8))
        if osp.isdir(start_menu):
            for fname in os.listdir(start_menu):
                try:
                    os.remove(osp.join(start_menu, fname))
                except OSError:
                    print('Failed to remove %s; you may be able to remove it manually.' % fname, file=sys.stderr)
                else:
                    print('Successfully removed Spyder shortcuts from your  start menu.', file=sys.stdout)
            try:
                os.rmdir(start_menu)
            except OSError:
                print('Failed to remove %s; you may be able to remove it manually.' % fname, file=sys.stderr)
            else:
                print('Successfully removed Spyder shortcut folder from your  start menu.', file=sys.stdout)
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-install':
            try:
                install()
            except OSError:
                print('Failed to create Start Menu items.', file=sys.stderr)
        elif sys.argv[1] == '-remove':
            remove()
        else:
            print('Unknown command line option %s' % sys.argv[1], file=sys.stderr)
    else:
        print('You need to pass either -install or -remove as options to this script', file=sys.stderr)