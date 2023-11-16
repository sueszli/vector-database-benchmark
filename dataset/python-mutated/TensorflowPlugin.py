""" Deprecated tensorflow plugin.
"""
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginTensorflow(NuitkaPluginBase):
    """This plugin is now not doing anything anymore."""
    plugin_name = 'tensorflow'
    plugin_desc = 'Deprecated, was once required by the tensorflow package'

    @classmethod
    def isDeprecated(cls):
        if False:
            i = 10
            return i + 15
        return True