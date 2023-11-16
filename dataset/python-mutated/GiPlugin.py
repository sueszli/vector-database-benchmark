""" Support for gi typelib files and DLLs
"""
import os
from nuitka.plugins.PluginBase import NuitkaPluginBase, standalone_only

class NuitkaPluginGi(NuitkaPluginBase):
    plugin_name = 'gi'
    plugin_desc = 'Support for GI package typelib dependency.'

    @staticmethod
    def isAlwaysEnabled():
        if False:
            i = 10
            return i + 15
        'Request to be always enabled.'
        return True

    @staticmethod
    @standalone_only
    def createPreModuleLoadCode(module):
        if False:
            return 10
        'Add typelib search path'
        if module.getFullName() == 'gi':
            code = '\nimport os\nif not os.environ.get("GI_TYPELIB_PATH"):\n    os.environ["GI_TYPELIB_PATH"] = os.path.join(__nuitka_binary_dir, "girepository")'
            return (code, 'Set typelib search path')

    @standalone_only
    def considerDataFiles(self, module):
        if False:
            return 10
        'Copy typelib files from the default installation path'
        if module.getFullName() == 'gi':
            gi_typelib_info = self.queryRuntimeInformationMultiple(info_name='gi_info', setup_codes='import gi; from gi.repository import GObject', values=(('introspection_module', "gi.Repository.get_default().get_typelib_path('GObject')"),))
            if gi_typelib_info is not None:
                gi_repository_path = os.path.dirname(gi_typelib_info.introspection_module)
                yield self.makeIncludedDataDirectory(source_path=gi_repository_path, dest_path='girepository', reason='typelib files for gi modules')

    @staticmethod
    def getImplicitImports(module):
        if False:
            return 10
        full_name = module.getFullName()
        if full_name == 'gi.overrides':
            yield 'gi.overrides.Gtk'
            yield 'gi.overrides.Gdk'
            yield 'gi.overrides.GLib'
            yield 'gi.overrides.GObject'
        elif full_name == 'gi._gi':
            yield 'gi._error'
        elif full_name == 'gi._gi_cairo':
            yield 'cairo'

    @standalone_only
    def getExtraDlls(self, module):
        if False:
            while True:
                i = 10

        def tryLocateAndLoad(dll_name):
            if False:
                i = 10
                return i + 15
            dll_path = self.locateDLL(dll_name)
            if dll_path is None:
                dll_path = self.locateDLL('%s' % dll_name)
            if dll_path is None:
                dll_path = self.locateDLL('lib%s' % dll_name)
            if dll_path is not None:
                yield self.makeDllEntryPoint(source_path=dll_path, dest_path=os.path.basename(dll_path), module_name='gi._gi', package_name='gi', reason="needed by 'gi._gi'")
        if module.getFullName() == 'gi._gi':
            for dll_name in ('gtk-3-0', 'soup-2.4-1', 'soup-gnome-2.4-1', 'libsecret-1-0'):
                yield tryLocateAndLoad(dll_name)