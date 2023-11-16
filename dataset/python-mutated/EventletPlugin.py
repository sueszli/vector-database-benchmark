""" Details see below in class definition.
"""
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginEventlet(NuitkaPluginBase):
    """This class represents the main logic of the plugin."""
    plugin_name = 'eventlet'
    plugin_desc = "Support for including 'eventlet' dependencies and its need for 'dns' package monkey patching."

    @staticmethod
    def isAlwaysEnabled():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getImplicitImports(self, module):
        if False:
            return 10
        full_name = module.getFullName()
        if full_name == 'eventlet':
            yield self.locateModules('dns')
            yield 'eventlet.hubs'
        elif full_name == 'eventlet.hubs':
            yield 'eventlet.hubs.epolls'
            yield 'eventlet.hubs.hub'
            yield 'eventlet.hubs.kqueue'
            yield 'eventlet.hubs.poll'
            yield 'eventlet.hubs.pyevent'
            yield 'eventlet.hubs.selects'
            yield 'eventlet.hubs.timer'

    def decideCompilation(self, module_name):
        if False:
            return 10
        if module_name.hasNamespace('dns'):
            return 'bytecode'