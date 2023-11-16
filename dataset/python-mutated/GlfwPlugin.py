""" Support for glfw, details in below class definitions.

"""
import os
import re
from nuitka import Options
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.FileOperations import getFileContents
from nuitka.utils.ModuleNames import ModuleName
from nuitka.utils.Utils import isLinux, isMacOS, isWin32Windows

class NuitkaPluginGlfw(NuitkaPluginBase):
    """This class represents the main logic of the glfw plugin.

    This is a plugin to ensure that glfw platform specific backends are loading
    properly. This need to include the correct DLL and make sure it's used by
    setting an environment variable.

    """
    plugin_name = 'glfw'
    plugin_desc = "Required for 'OpenGL' (PyOpenGL) and 'glfw' package in standalone mode."

    @staticmethod
    def isAlwaysEnabled():
        if False:
            return 10
        return True

    @classmethod
    def isRelevant(cls):
        if False:
            i = 10
            return i + 15
        'Check whether plugin might be required.\n\n        Returns:\n            True if this is a standalone compilation.\n        '
        return Options.isStandaloneMode()

    def getImplicitImports(self, module):
        if False:
            return 10
        if module.getFullName() == 'OpenGL':
            opengl_infos = self.queryRuntimeInformationSingle(setup_codes='import OpenGL.plugins', value='[(f.name, f.import_path) for f in OpenGL.plugins.FormatHandler.all()]')
            for (_name, import_path) in opengl_infos:
                yield ModuleName(import_path).getPackageName()
            code = getFileContents(module.getCompileTimeFilename())
            for (os_part, plugin_name_part) in re.findall('PlatformPlugin\\(\\s*[\'"](\\w+)[\'"]\\s*,\\s*[\'"]([\\w\\.]+)[\'"]\\s*\\)', code):
                plugin_name_part = ModuleName(plugin_name_part).getPackageName()
                if os_part == 'nt':
                    if isWin32Windows():
                        yield plugin_name_part
                elif os_part.startswith('linux'):
                    if isLinux():
                        yield plugin_name_part
                elif os_part.startswith('darwin'):
                    if isMacOS():
                        yield plugin_name_part
                elif os_part.startswith(('posix', 'osmesa', 'egl', 'x11', 'wayland', 'xwayland', 'glx')):
                    if not isWin32Windows() and (not isMacOS()):
                        yield plugin_name_part
                else:
                    self.sysexit("Undetected OS specific PyOpenGL plugin '%s', please report bug for." % os_part)

    def _getDLLFilename(self):
        if False:
            print('Hello World!')
        glfw_info = self.queryRuntimeInformationMultiple(info_name='glfw_info', setup_codes='import glfw.library', values=(('dll_filename', 'glfw.library.glfw._name'),))
        return glfw_info.dll_filename

    def getExtraDlls(self, module):
        if False:
            while True:
                i = 10
        if module.getFullName() == 'glfw':
            dll_filename = self._getDLLFilename()
            yield self.makeDllEntryPoint(source_path=dll_filename, dest_path=os.path.join('glfw', os.path.basename(dll_filename)), module_name='glfw', package_name='glfw.library', reason="needed by 'glfw'")

    def createPreModuleLoadCode(self, module):
        if False:
            i = 10
            return i + 15
        if module.getFullName() == 'glfw':
            dll_filename = self._getDLLFilename()
            code = '\nimport os\nos.environ["PYGLFW_LIBRARY"] = os.path.join(__nuitka_binary_dir, "glfw", "%s")\n' % os.path.basename(dll_filename)
            return (code, "Setting 'PYGLFW_LIBRARY' environment variable for glfw to find platform DLL.")