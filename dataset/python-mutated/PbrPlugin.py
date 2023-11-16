""" Standard plug-in to make pbr module work when compiled.

The pbr module needs to find a version number in compiled mode. The value
itself seems less important than the fact that some value does exist.
"""
from nuitka import Options
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginPbrWorkarounds(NuitkaPluginBase):
    """This is to make pbr module work when compiled with Nuitka."""
    plugin_name = 'pbr-compat'
    plugin_desc = "Required by the 'pbr' package in standalone mode."

    @classmethod
    def isRelevant(cls):
        if False:
            return 10
        return Options.isStandaloneMode()

    @staticmethod
    def isAlwaysEnabled():
        if False:
            return 10
        return True

    @staticmethod
    def createPreModuleLoadCode(module):
        if False:
            print('Hello World!')
        full_name = module.getFullName()
        if full_name == 'pbr.packaging':
            code = 'import os\nversion = os.environ.get(\n        "PBR_VERSION",\n        os.environ.get("OSLO_PACKAGE_VERSION"))\nif not version:\n    os.environ["OSLO_PACKAGE_VERSION"] = "1.0"\n'
            return (code, 'Monkey patching "pbr" version number.')