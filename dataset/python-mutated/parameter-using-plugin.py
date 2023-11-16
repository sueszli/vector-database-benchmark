""" Parameter using Nuitka plugin.

"""
import os
import sys
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginForTesting(NuitkaPluginBase):
    plugin_name = __name__.split('.')[-1]

    def __init__(self, trace_my_plugin):
        if False:
            print('Hello World!')
        self.check = trace_my_plugin
        self.info("The 'trace' value is set to '%s'" % self.check)

    @classmethod
    def addPluginCommandLineOptions(cls, group):
        if False:
            print('Hello World!')
        group.add_option('--trace-my-plugin', action='store_true', dest='trace_my_plugin', default=False, help='This is show in help output.')

    def onModuleSourceCode(self, module_name, source_filename, source_code):
        if False:
            i = 10
            return i + 15
        if module_name == '__main__' and self.check:
            self.info('')
            self.info(" Calls to 'math' module:")
            for (i, l) in enumerate(source_code.splitlines()):
                if 'math.' in l:
                    self.info(' %i: %s' % (i + 1, l))
            self.info('')
        return source_code