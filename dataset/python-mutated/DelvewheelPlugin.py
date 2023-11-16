""" Support for delvewheel, details in below class definitions.

"""
import os
import re
from nuitka import Options
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.PythonFlavors import isAnacondaPython
from nuitka.utils.FileOperations import listDllFilesFromDirectory

class NuitkaPluginDelvewheel(NuitkaPluginBase):
    """This class represents the main logic of the delvewheel plugin.

    This is a plugin to ensure that delvewheel DLLs are loading properly in
    standalone mode. This needed to include the correct DLLs to the correct
    place.
    """
    plugin_name = 'delvewheel'
    plugin_desc = "Required for 'support' of delvewheel using packages in standalone mode."

    def __init__(self):
        if False:
            print('Hello World!')
        self.dll_directories = {}
        self.dll_directory = None

    @staticmethod
    def isAlwaysEnabled():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isRelevant():
        if False:
            print('Hello World!')
        return Options.isStandaloneMode()

    def _add_dll_directory(self, arg):
        if False:
            print('Hello World!')
        self.dll_directory = arg

    def onModuleSourceCode(self, module_name, source_filename, source_code):
        if False:
            i = 10
            return i + 15
        if '_delvewheel_' not in source_code:
            return None
        match = re.search('(def _delvewheel_(?:init_)?patch_(.*?)\\(\\):\\n.*?_delvewheel_(?:init_)?patch_\\2\\(\\))', source_code, re.S)
        if not match:
            return None
        delvewheel_version = match.group(2).replace('_', '.')
        code = match.group(1)
        code = code.replace('os.add_dll_directory', 'add_dll_directory')
        code = code.replace('sys.version_info[:2] >= (3, 8)', 'True')
        code = code.replace('sys.version_info[:2] >= (3, 10)', 'True')
        self.dll_directory = None
        exec_globals = {'__file__': self.locateModule(module_name) + '\\__init__.py', 'add_dll_directory': self._add_dll_directory}
        self.dll_directory = None
        exec(code, exec_globals)
        if not isAnacondaPython():
            assert self.dll_directory is not None, module_name
        if self.dll_directory is not None:
            self.dll_directory = os.path.normpath(self.dll_directory)
            if os.path.basename(self.dll_directory) in ('site-packages', 'dist-packages', 'vendor-packages'):
                self.dll_directory = None
        self.dll_directories[module_name] = self.dll_directory
        if self.dll_directories[module_name]:
            self.info("Detected usage of 'delvewheel' version '%s' in module '%s'." % (delvewheel_version, module_name.asString()))

    def getExtraDlls(self, module):
        if False:
            i = 10
            return i + 15
        full_name = module.getFullName()
        dll_directory = self.dll_directories.get(full_name)
        if dll_directory is not None:
            for (dll_filename, dll_basename) in listDllFilesFromDirectory(dll_directory):
                yield self.makeDllEntryPoint(source_path=dll_filename, dest_path=os.path.join(os.path.basename(dll_directory), dll_basename), module_name=full_name, package_name=full_name, reason="needed by '%s'" % full_name.asString())