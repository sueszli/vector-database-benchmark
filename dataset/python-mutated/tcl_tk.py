import locale
import os
from PyInstaller import compat
from PyInstaller import isolated
from PyInstaller import log as logging
from PyInstaller.building.datastruct import Tree
from PyInstaller.depend import bindepend
logger = logging.getLogger(__name__)
TK_ROOTNAME = 'tk'
TCL_ROOTNAME = 'tcl'

@isolated.decorate
def _get_tcl_tk_info():
    if False:
        i = 10
        return i + 15
    '\n    Isolated-subprocess helper to retrieve the basic Tcl/Tk information:\n     - tcl_dir = path to the Tcl library/data directory.\n     - tcl_version = Tcl version\n     - tk_version = Tk version\n     - tcl_theaded = boolean indicating whether Tcl/Tk is built with multi-threading support.\n    '
    try:
        import tkinter
        from _tkinter import TCL_VERSION, TK_VERSION
    except ImportError:
        return (None, None, None, False)
    try:
        tcl = tkinter.Tcl()
    except tkinter.TclError:
        return (None, None, None, False)
    tcl_dir = tcl.eval('info library')
    try:
        tcl.getvar('tcl_platform(threaded)')
        tcl_threaded = True
    except tkinter.TclError:
        tcl_threaded = False
    return (tcl_dir, TCL_VERSION, TK_VERSION, tcl_threaded)
(tcl_dir, tcl_version, tk_version, tcl_threaded) = _get_tcl_tk_info()

def _warn_if_activetcl_or_teapot_installed(tcl_root, tcltree):
    if False:
        i = 10
        return i + 15
    '\n    If the current Tcl installation is a Teapot-distributed version of ActiveTcl *and* the current platform is macOS,\n    log a non-fatal warning that the resulting executable will (probably) fail to run on non-host systems.\n\n    PyInstaller does *not* freeze all ActiveTcl dependencies -- including Teapot, which is typically ignorable. Since\n    Teapot is *not* ignorable in this case, this function warns of impending failure.\n\n    See Also\n    -------\n    https://github.com/pyinstaller/pyinstaller/issues/621\n    '
    import macholib.util
    if macholib.util.in_system_path(tcl_root):
        return
    try:
        init_resource = [r[1] for r in tcltree if r[1].endswith('init.tcl')][0]
    except IndexError:
        return
    mentions_activetcl = False
    mentions_teapot = False
    with open(init_resource, 'r', encoding=locale.getpreferredencoding()) as init_file:
        for line in init_file.readlines():
            line = line.strip().lower()
            if line.startswith('#'):
                continue
            if 'activetcl' in line:
                mentions_activetcl = True
            if 'teapot' in line:
                mentions_teapot = True
            if mentions_activetcl and mentions_teapot:
                break
    if mentions_activetcl and mentions_teapot:
        logger.warning('\nYou appear to be using an ActiveTcl build of Tcl/Tk, which PyInstaller has\ndifficulty freezing. To fix this, comment out all references to "teapot" in:\n\n     %s\n\nSee https://github.com/pyinstaller/pyinstaller/issues/621 for more information.\n            ' % init_resource)

def find_tcl_tk_shared_libs(tkinter_ext_file):
    if False:
        print('Hello World!')
    '\n    Find Tcl and Tk shared libraries against which the _tkinter module is linked.\n\n    Returns\n    -------\n    list\n        list containing two tuples, one for Tcl and one for Tk library, where each tuple contains the library name and\n        its full path, i.e., [(tcl_lib, tcl_libpath), (tk_lib, tk_libpath)]. If a library is not found, the\n        corresponding tuple elements are set to None.\n    '
    tcl_lib = None
    tcl_libpath = None
    tk_lib = None
    tk_libpath = None
    for (_, lib_path) in bindepend.get_imports(tkinter_ext_file):
        if lib_path is None:
            continue
        lib_name = os.path.basename(lib_path)
        lib_name_lower = lib_name.lower()
        if 'tcl' in lib_name_lower:
            tcl_lib = lib_name
            tcl_libpath = lib_path
        elif 'tk' in lib_name_lower:
            tk_lib = lib_name
            tk_libpath = lib_path
    return [(tcl_lib, tcl_libpath), (tk_lib, tk_libpath)]

def _find_tcl_tk(tkinter_ext_file):
    if False:
        i = 10
        return i + 15
    '\n    Get a platform-specific 2-tuple of the absolute paths of the top-level external data directories for both\n    Tcl and Tk, respectively.\n\n    Returns\n    -------\n    list\n        2-tuple that contains the values of `${TCL_LIBRARY}` and `${TK_LIBRARY}`, respectively.\n    '
    if compat.is_darwin:
        libs = find_tcl_tk_shared_libs(tkinter_ext_file)
        path_to_tcl = libs[0][1]
        if path_to_tcl is None:
            return (None, None)
        if 'Library/Frameworks/Tcl.framework' in path_to_tcl:
            return (None, None)
    else:
        pass
    tk_dir = os.path.join(os.path.dirname(tcl_dir), f'tk{tk_version}')
    return (tcl_dir, tk_dir)

def _collect_tcl_modules(tcl_root):
    if False:
        print('Hello World!')
    '\n    Get a list of TOC-style 3-tuples describing Tcl modules. The modules directory is separate from the library/data\n    one, and is located at $tcl_root/../tclX, where X is the major Tcl version.\n\n    Returns\n    -------\n    Tree\n        Such list, if the modules directory exists.\n    '
    tcl_major_version = tcl_version.split('.')[0]
    modules_dirname = f'tcl{tcl_major_version}'
    modules_path = os.path.join(tcl_root, '..', modules_dirname)
    if not os.path.isdir(modules_path):
        logger.warning('Tcl modules directory %s does not exist.', modules_path)
        return []
    return Tree(modules_path, prefix=modules_dirname)

def collect_tcl_tk_files(tkinter_ext_file):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a list of TOC-style 3-tuples describing all external Tcl/Tk data files.\n\n    Returns\n    -------\n    Tree\n        Such list.\n    '
    (tcl_root, tk_root) = _find_tcl_tk(tkinter_ext_file)
    if compat.is_darwin and (not tcl_root) and (not tk_root):
        logger.info("Not collecting Tcl/Tk data - either python is using macOS' system Tcl/Tk framework, or Tcl/Tk data directories could not be found.")
        return []
    if not tcl_root:
        logger.error('Tcl/Tk improperly installed on this system.')
        return []
    if not os.path.isdir(tcl_root):
        logger.error('Tcl data directory "%s" not found.', tcl_root)
        return []
    if not os.path.isdir(tk_root):
        logger.error('Tk data directory "%s" not found.', tk_root)
        return []
    tcltree = Tree(tcl_root, prefix=TCL_ROOTNAME, excludes=['demos', '*.lib', 'tclConfig.sh'])
    tktree = Tree(tk_root, prefix=TK_ROOTNAME, excludes=['demos', '*.lib', 'tkConfig.sh'])
    if compat.is_darwin:
        _warn_if_activetcl_or_teapot_installed(tcl_root, tcltree)
    tclmodulestree = _collect_tcl_modules(tcl_root)
    return tcltree + tktree + tclmodulestree