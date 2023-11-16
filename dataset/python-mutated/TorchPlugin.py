""" Deprecated torch plugin.
"""
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginTorch(NuitkaPluginBase):
    """This plugin is now not doing anything anymore."""
    plugin_name = 'torch'
    plugin_desc = 'Deprecated, was once required by the torch package'

    @classmethod
    def isDeprecated(cls):
        if False:
            for i in range(10):
                print('nop')
        return True