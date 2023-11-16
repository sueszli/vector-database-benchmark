""" Details see below in class definition.
"""
from nuitka.Options import isStandaloneMode
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.plugins.Plugins import getActiveQtPlugin
from nuitka.utils.Utils import getOS, isMacOS, isWin32Windows

class NuitkaPluginPywebview(NuitkaPluginBase):
    """This class represents the main logic of the plugin."""
    plugin_name = 'pywebview'
    plugin_desc = "Required by the 'webview' package (pywebview on PyPI)."

    @staticmethod
    def isAlwaysEnabled():
        if False:
            while True:
                i = 10
        return True

    @classmethod
    def isRelevant(cls):
        if False:
            while True:
                i = 10
        'One time only check: may this plugin be required?\n\n        Returns:\n            True if this is a standalone compilation.\n        '
        return isStandaloneMode()

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if False:
            for i in range(10):
                print('nop')
        if module_name.isBelowNamespace('webview.platforms'):
            if isWin32Windows():
                result = module_name in ('webview.platforms.winforms', 'webview.platforms.edgechromium', 'webview.platforms.edgehtml', 'webview.platforms.mshtml', 'webview.platforms.cef')
                reason = "Platforms package of webview used on '%s'." % getOS()
            elif isMacOS():
                result = module_name == 'webview.platforms.cocoa'
                reason = "Platforms package of webview used on '%s'." % getOS()
            elif getActiveQtPlugin() is not None:
                result = module_name = 'webview.platforms.qt'
                reason = "Platforms package of webview used due to '%s' plugin being active." % getActiveQtPlugin()
            else:
                result = module_name = 'webview.platforms.gtk'
                reason = "Platforms package of webview used on '%s' without Qt plugin enabled." % getOS()
            return (result, reason)