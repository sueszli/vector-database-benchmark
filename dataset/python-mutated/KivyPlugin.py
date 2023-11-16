""" Details see below in class definition.
"""
from nuitka.Options import isStandaloneMode
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginKivy(NuitkaPluginBase):
    """This class represents the main logic of the plugin."""
    plugin_name = 'kivy'
    plugin_desc = "Required by 'kivy' package."

    @staticmethod
    def isAlwaysEnabled():
        if False:
            print('Hello World!')
        return True

    @classmethod
    def isRelevant(cls):
        if False:
            for i in range(10):
                print('nop')
        'One time only check: may this plugin be required?\n\n        Returns:\n            True if this is a standalone compilation.\n        '
        return isStandaloneMode()

    def _getKivyInformation(self):
        if False:
            return 10
        setup_codes = '\nimport kivy.core.image\nimport kivy.core.text\n# Prevent Window from being created at compile time.\nkivy.core.core_select_lib=(lambda *args, **kwargs: None)\nimport kivy.core.window\n\n# Kivy has packages designed to provide these on Windows\ntry:\n    from kivy_deps.sdl2 import dep_bins as sdl2_dep_bins\nexcept ImportError:\n    sdl2_dep_bins = []\ntry:\n    from kivy_deps.glew import dep_bins as glew_dep_bins\nexcept ImportError:\n    glew_dep_bins = []\n'
        info = self.queryRuntimeInformationMultiple(info_name='kivy_info', setup_codes=setup_codes, values=(('libs_loaded', 'kivy.core.image.libs_loaded'), ('window_impl', 'kivy.core.window.window_impl'), ('label_libs', 'kivy.core.text.label_libs'), ('sdl2_dep_bins', 'sdl2_dep_bins'), ('glew_dep_bins', 'glew_dep_bins')))
        if info is None:
            self.sysexit('Error, it seems Kivy is not installed.')
        return info

    def getImplicitImports(self, module):
        if False:
            for i in range(10):
                print('nop')
        full_name = module.getFullName()
        if full_name == 'kivy.core.image':
            for module_name in self._getKivyInformation().libs_loaded:
                yield full_name.getChildNamed(module_name)
        elif full_name == 'kivy.core.window':
            for (_, module_name, _) in self._getKivyInformation().window_impl:
                yield full_name.getChildNamed(module_name)
        elif full_name == 'kivy.core.text':
            for (_, module_name, _) in self._getKivyInformation().label_libs:
                yield full_name.getChildNamed(module_name)
        elif full_name == 'kivy.core.window.window_sdl2':
            yield 'kivy.core.window._window_sdl2'
        elif full_name == 'kivy.core.window._window_sdl2':
            yield 'kivy.core.window.window_info'
        elif full_name == 'kivy.core.window.window_x11':
            yield 'kivy.core.window.window_info'
        elif full_name == 'kivy.graphics.cgl':
            yield 'kivy.graphics.cgl_backend'
        elif full_name == 'kivy.graphics.cgl_backend':
            yield 'kivy.graphics.cgl_backend.cgl_glew'
        elif full_name == 'kivy.graphics.cgl_backend.cgl_glew':
            yield 'kivy.graphics.cgl_backend.cgl_gl'
        elif full_name == 'kivymd.app':
            yield self.locateModules('kivymd.uix')

    def getExtraDlls(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Copy extra shared libraries or data for this installation.\n\n        Args:\n            module: module object\n        Yields:\n            DLL entry point objects\n        '
        full_name = module.getFullName()
        if full_name == 'kivy':
            kivy_info = self._getKivyInformation()
            kivy_dlls = []
            for dll_folder in kivy_info.sdl2_dep_bins + kivy_info.glew_dep_bins:
                kivy_dlls.extend(self.locateDLLsInDirectory(dll_folder))
            for (full_path, target_filename, _dll_extension) in kivy_dlls:
                yield self.makeDllEntryPoint(source_path=full_path, dest_path=target_filename, module_name=full_name, package_name=full_name, reason="needed by 'kivy'")
            self.reportFileCount(full_name, len(kivy_dlls))