""" Details see below in class definition.
"""
from nuitka import Options
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginGevent(NuitkaPluginBase):
    """This class represents the main logic of the plugin."""
    plugin_name = 'gevent'
    plugin_desc = "Required by the 'gevent' package."

    @staticmethod
    def isAlwaysEnabled():
        if False:
            i = 10
            return i + 15
        return True

    @classmethod
    def isRelevant(cls):
        if False:
            i = 10
            return i + 15
        'One time only check: may this plugin be required?\n\n        Returns:\n            True if this is a standalone compilation.\n        '
        return Options.isStandaloneMode()

    @staticmethod
    def createPostModuleLoadCode(module):
        if False:
            return 10
        'Make sure greentlet tree tracking is switched off.'
        full_name = module.getFullName()
        if full_name == 'gevent':
            code = '\\\nimport gevent._config\ngevent._config.config.track_greenlet_tree = False\n'
            return (code, "Disabling 'gevent' greenlet tree tracking.")