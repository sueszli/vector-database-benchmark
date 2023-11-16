""" Details see below in class definition.
"""
import os
import sys
from nuitka.Options import isStandaloneMode, shallCreateAppBundle
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.PythonVersions import getSystemPrefixPath, getTkInterVersion
from nuitka.utils.FileOperations import listDllFilesFromDirectory, relpath
from nuitka.utils.Utils import isMacOS, isWin32Windows

def _isTkInterModule(module):
    if False:
        i = 10
        return i + 15
    full_name = module.getFullName()
    return full_name in ('Tkinter', 'tkinter', 'PySimpleGUI', 'PySimpleGUI27')

class NuitkaPluginTkinter(NuitkaPluginBase):
    """This class represents the main logic of the TkInter plugin.

     This is a plug-in to make programs work well in standalone mode which are using tkinter.
     These programs require the presence of certain libraries written in the TCL language.
     On Windows platforms, and even on Linux, the existence of these libraries cannot be
     assumed. We therefore

     1. Copy the TCL libraries as sub-folders to the program's dist folder
     2. Redirect the program's tkinter requests to these library copies. This is
        done by setting appropriate variables in the os.environ dictionary.
        Tkinter will use these variable value to locate the library locations.

     Each time before the program issues an import to a tkinter module, we make
     sure, that the TCL environment variables are correctly set.

    Notes:
         You can enforce using a specific TCL folder by using TCL_LIBRARY
         and a Tk folder by using TK_LIBRARY, but that ought to normally
         not be necessary.
    """
    plugin_name = 'tk-inter'
    plugin_desc = "Required by Python's Tk modules."
    plugin_gui_toolkit = True
    binding_name = 'tkinter'

    def __init__(self, tcl_library_dir, tk_library_dir):
        if False:
            for i in range(10):
                print('nop')
        self.tcl_library_dir = tcl_library_dir
        self.tk_library_dir = tk_library_dir
        self.files_copied = False
        self.tk_inter_version = getTkInterVersion()
        if self.tk_inter_version is None:
            self.sysexit('Error, it seems tk-inter is not installed.')
        assert self.tk_inter_version in ('8.5', '8.6'), self.tk_inter_version
        return None

    @classmethod
    def isRelevant(cls):
        if False:
            i = 10
            return i + 15
        'This method is called one time only to check, whether the plugin might make sense at all.\n\n        Returns:\n            True if this is a standalone, else False.\n        '
        return isStandaloneMode()

    @staticmethod
    def createPreModuleLoadCode(module):
        if False:
            print('Hello World!')
        'This method is called with a module that will be imported.\n\n        Notes:\n            If the word "tkinter" occurs in its full name, we know that the correct\n            setting of the TCL environment must be ensured before this happens.\n\n        Args:\n            module: the module object\n        Returns:\n            Code to insert and None (tuple)\n        '
        if _isTkInterModule(module):
            code = '\nimport os\nos.environ["TCL_LIBRARY"] = os.path.join(__nuitka_binary_dir, "tcl")\nos.environ["TK_LIBRARY"] = os.path.join(__nuitka_binary_dir, "tk")'
            return (code, 'Need to make sure we set environment variables for TCL.')

    @classmethod
    def addPluginCommandLineOptions(cls, group):
        if False:
            while True:
                i = 10
        group.add_option('--tk-library-dir', action='store', dest='tk_library_dir', default=None, help='The Tk library dir. Nuitka is supposed to automatically detect it, but you can\noverride it here. Default is automatic detection.')
        group.add_option('--tcl-library-dir', action='store', dest='tcl_library_dir', default=None, help='The Tcl library dir. See comments for Tk library dir.')

    @staticmethod
    def _getTkinterDnDPlatformDirectory():
        if False:
            for i in range(10):
                print('nop')
        import platform
        if platform.system() == 'Darwin':
            return 'osx64'
        elif platform.system() == 'Linux':
            return 'linux64'
        elif platform.system() == 'Windows':
            return 'win64'
        else:
            return None

    def _considerDataFilesTkinterDnD(self, module):
        if False:
            return 10
        platform_rep = self._getTkinterDnDPlatformDirectory()
        if platform_rep is None:
            return
        yield self.makeIncludedPackageDataFiles(package_name='tkinterdnd2', package_directory=module.getCompileTimeDirectory(), pattern=os.path.join('tkdnd', platform_rep, '**'), reason="Tcl needed for 'tkinterdnd2' usage", tags='tcl')

    def _getTclCandidatePaths(self):
        if False:
            i = 10
            return i + 15
        yield os.environ.get('TCL_LIBRARY')
        for sys_prefix_path in (sys.prefix, getSystemPrefixPath()):
            yield os.path.join(sys_prefix_path, 'tcl', 'tcl%s' % self.tk_inter_version)
            yield os.path.join(sys_prefix_path, 'lib', 'tcl%s' % self.tk_inter_version)
            yield os.path.join(sys_prefix_path, 'Library', 'lib', 'tcl%s' % self.tk_inter_version)
        if not isWin32Windows():
            yield ('/usr/share/tcltk/tcl%s' % self.tk_inter_version)
            yield ('/usr/share/tcl%s' % self.tk_inter_version)
            yield ('/usr/lib64/tcl/tcl%s' % self.tk_inter_version)
            yield ('/usr/lib/tcl%s' % self.tk_inter_version)

    def _getTkCandidatePaths(self):
        if False:
            while True:
                i = 10
        yield os.environ.get('TK_LIBRARY')
        for sys_prefix_path in (sys.prefix, getSystemPrefixPath()):
            yield os.path.join(sys_prefix_path, 'tcl', 'tk%s' % self.tk_inter_version)
            yield os.path.join(sys_prefix_path, 'lib', 'tk%s' % self.tk_inter_version)
            yield os.path.join(sys_prefix_path, 'Library', 'lib', 'tk%s' % self.tk_inter_version)
        if not isWin32Windows():
            yield ('/usr/share/tcltk/tk%s' % self.tk_inter_version)
            yield ('/usr/share/tk%s' % self.tk_inter_version)
            yield ('/usr/lib64/tcl/tk%s' % self.tk_inter_version)
            yield ('/usr/lib/tk%s' % self.tk_inter_version)

    def considerDataFiles(self, module):
        if False:
            return 10
        "Provide TCL libraries to the dist folder.\n\n        Notes:\n            We will provide the copy the TCL/TK directories to the program's root directory,\n            that might be shiftable with some work.\n\n        Args:\n            module: the module in question, maybe ours\n\n        Yields:\n            IncludedDataFile objects.\n        "
        if module.getFullName() == 'tkinterdnd2.TkinterDnD':
            yield self._considerDataFilesTkinterDnD(module)
            return
        if not _isTkInterModule(module) or self.files_copied:
            return
        tcl_library_dir = self.tcl_library_dir
        if tcl_library_dir is None:
            for tcl_library_dir in self._getTclCandidatePaths():
                if tcl_library_dir is not None and os.path.exists(os.path.join(tcl_library_dir, 'init.tcl')):
                    break
        if tcl_library_dir is None or not os.path.exists(tcl_library_dir):
            self.sysexit("Could not find Tcl, you might need to use '--tcl-library-dir' and if that works, report a bug so it can be added to Nuitka.")
        tk_library_dir = self.tk_library_dir
        if tk_library_dir is None:
            for tk_library_dir in self._getTkCandidatePaths():
                if tk_library_dir is not None and os.path.exists(os.path.join(tk_library_dir, 'dialog.tcl')):
                    break
        if tk_library_dir is None or not os.path.exists(tk_library_dir):
            self.sysexit("Could not find Tk, you might need to use '--tk-library-dir' and if that works, report a bug.")
        yield self.makeIncludedDataDirectory(source_path=tk_library_dir, dest_path='tk', reason='Tk needed for tkinter usage', ignore_dirs=('demos',), tags='tk')
        yield self.makeIncludedDataDirectory(source_path=tcl_library_dir, ignore_dirs=('opt0.4', 'http1.0') if isMacOS() and shallCreateAppBundle() else (), dest_path='tcl', reason='Tcl needed for tkinter usage', tags='tcl')
        if isWin32Windows():
            yield self.makeIncludedDataDirectory(source_path=os.path.join(tcl_library_dir, '..', 'tcl8'), dest_path='tcl8', reason='Tcl needed for tkinter usage', tags='tcl')
        self.files_copied = True

    def getExtraDlls(self, module):
        if False:
            print('Hello World!')
        if module.getFullName() == 'tkinterdnd2.TkinterDnD':
            platform_rep = self._getTkinterDnDPlatformDirectory()
            if platform_rep is None:
                return
            module_directory = module.getCompileTimeDirectory()
            for (filename, _dll_filename) in listDllFilesFromDirectory(os.path.join(module_directory, 'tkdnd', platform_rep)):
                dest_path = relpath(filename, module_directory)
                yield self.makeDllEntryPoint(source_path=filename, dest_path=os.path.join('tkinterdnd2', dest_path), module_name='tkinterdnd2', package_name='tkinterdnd2', reason='tkinterdnd2 package DLL')

    def onModuleCompleteSet(self, module_set):
        if False:
            i = 10
            return i + 15
        if str is bytes:
            plugin_binding_name = 'Tkinter'
        else:
            plugin_binding_name = 'tkinter'
        self.onModuleCompleteSetGUI(module_set=module_set, plugin_binding_name=plugin_binding_name)

class NuitkaPluginDetectorTkinter(NuitkaPluginBase):
    """Used only if plugin is not activated.

    Notes:
        We are given the chance to issue a warning if we think we may be required.
    """
    detector_for = NuitkaPluginTkinter

    @classmethod
    def isRelevant(cls):
        if False:
            while True:
                i = 10
        'This method is called one time only to check, whether the plugin might make sense at all.\n\n        Returns:\n            True if this is a standalone compilation on Windows, else False.\n        '
        return isStandaloneMode()

    def checkModuleSourceCode(self, module_name, source_code):
        if False:
            print('Hello World!')
        'This method checks the source code\n\n        Notes:\n            We only use it to check whether this is the main module, and whether\n            it contains the keyword "tkinter".\n            We assume that the main program determines whether tkinter is used.\n            References by dependent or imported modules are assumed irrelevant.\n\n        Args:\n            module_name: the name of the module\n            source_code: the module\'s source code\n\n        Returns:\n            None\n        '
        if module_name == '__main__':
            for line in source_code.splitlines():
                if '#' in line:
                    line = line[:line.find('#')]
                if 'tkinter' in line or 'Tkinter' in line:
                    self.warnUnusedPlugin('Tkinter needs TCL included.')
                    break