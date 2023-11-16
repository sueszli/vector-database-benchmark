""" Standard plug-in to handle pkg_resource special needs.

Nuitka can detect some things that "pkg_resources" may not even be able to during
runtime, but that is done by nodes and optimization. But there are other things,
that need special case, e.g. the registration of the loader class.
"""
import re
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.Utils import withNoDeprecationWarning

class NuitkaPluginResources(NuitkaPluginBase):
    plugin_name = 'pkg-resources'
    plugin_desc = "Workarounds for 'pkg_resources'."

    def __init__(self):
        if False:
            i = 10
            return i + 15
        with withNoDeprecationWarning():
            try:
                import pkg_resources
            except (ImportError, RuntimeError):
                self.pkg_resources = None
            else:
                self.pkg_resources = pkg_resources
        try:
            import importlib_metadata
        except (ImportError, SyntaxError, RuntimeError):
            self.metadata = None
        else:
            self.metadata = importlib_metadata

    @staticmethod
    def isAlwaysEnabled():
        if False:
            while True:
                i = 10
        return True

    def _handleEasyInstallEntryScript(self, dist, group, name):
        if False:
            i = 10
            return i + 15
        module_name = None
        main_name = None
        if self.metadata:
            dist = self.metadata.distribution(dist.partition('==')[0])
            for entry_point in dist.entry_points:
                if entry_point.group == group and entry_point.name == name:
                    module_name = entry_point.module
                    main_name = entry_point.attr
                    break
        if module_name is None and self.pkg_resources:
            with withNoDeprecationWarning():
                entry_point = self.pkg_resources.get_entry_info(dist, group, name)
            module_name = entry_point.module_name
            main_name = entry_point.name
        if module_name is None:
            self.sysexit('Error, failed to resolve easy install entry script, is the installation broken?')
        return "\nimport sys, re\nsys.argv[0] = re.sub(r'(-script\\.pyw?|\\.exe)?$', '', sys.argv[0])\nimport %(module_name)s\nsys.exit(%(module_name)s.%(main_name)s)\n" % {'module_name': module_name, 'main_name': main_name}

    def onModuleSourceCode(self, module_name, source_filename, source_code):
        if False:
            return 10
        if module_name == '__main__':
            match = re.search("\n# EASY-INSTALL-ENTRY-SCRIPT: '(.*?)','(.*?)','(.*?)'", source_code)
            if match is not None:
                self.info('Detected easy install entry script, compile time detecting entry point.')
                return self._handleEasyInstallEntryScript(*match.groups())
        return source_code

    def createPostModuleLoadCode(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Create code to load after a module was successfully imported.\n\n        For pkg_resources we need to register a provider.\n        '
        if module.getFullName() != 'pkg_resources':
            return
        code = "from __future__ import absolute_import\n\nimport os\nfrom pkg_resources import register_loader_type, EggProvider\n\nclass NuitkaProvider(EggProvider):\n    def _has(self, path):\n        return os.path.exists(path)\n\n    def _isdir(self, path):\n        return os.path.isdir(path)\n\n    def _listdir(self, path):\n        return os.listdir(path)\n\n    def get_resource_stream(self, manager, resource_name):\n        return open(self._fn(self.module_path, resource_name), 'rb')\n\n    def _get(self, path):\n        with open(path, 'rb') as stream:\n            return stream.read()\n\nregister_loader_type(__nuitka_loader_type, NuitkaProvider)\n"
        yield (code, 'Registering Nuitka loader with "pkg_resources".')