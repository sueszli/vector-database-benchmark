""" Details see below in class definition.
"""
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginNumpy(NuitkaPluginBase):
    """This plugin is now not doing anything anymore."""
    plugin_name = 'numpy'
    plugin_desc = 'Deprecated, was once required by the numpy package'

    @classmethod
    def isDeprecated(cls):
        if False:
            i = 10
            return i + 15
        return True