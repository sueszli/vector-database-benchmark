""" Deprecated trio plugin.
"""
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginTrio(NuitkaPluginBase):
    plugin_name = 'trio'
    plugin_desc = "Deprecated, was once required by the 'trio' package"

    @classmethod
    def isDeprecated(cls):
        if False:
            while True:
                i = 10
        return True