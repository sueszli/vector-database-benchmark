""" Standard plug-in to tell user about needed or useful options for packages.

When certain GUI packages are used, disabling the console may or may not be what
the user wants, or even be required, as e.g. "wx" on macOS will crash unless the
console is disabled. This reads Yaml configuration.
"""
from nuitka.Options import isOnefileMode, isStandaloneMode, mayDisableConsoleWindow, shallCreateAppBundle, shallDisableConsoleWindow
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.Utils import isMacOS
from nuitka.utils.Yaml import getYamlPackageConfiguration

class NuitkaPluginOptionsNanny(NuitkaPluginBase):
    plugin_name = 'options-nanny'
    plugin_desc = 'Inform the user about potential problems as per package configuration files.'

    def __init__(self):
        if False:
            return 10
        self.config = getYamlPackageConfiguration()

    @staticmethod
    def isAlwaysEnabled():
        if False:
            while True:
                i = 10
        return True

    def sysexitIllegalOptionValue(self, full_name, option, value):
        if False:
            for i in range(10):
                print('nop')
        self.sysexit("Illegal value for package '%s' option '%s' ('%s')" % (full_name, option, value))

    def _checkSupportedVersion(self, full_name, support_info, description, condition):
        if False:
            i = 10
            return i + 15
        if support_info == 'ignore':
            return
        if condition != 'True':
            problem_desc = "incomplete support due untrue condition '%s'" % condition
        else:
            problem_desc = 'incomplete support'
        message = "Using module '%s' (version %s) with %s: %s" % (full_name, '.'.join((str(d) for d in self.getPackageVersion(full_name))), problem_desc, description)
        if support_info == 'error':
            self.sysexit(message)
        elif support_info == 'warning':
            self.warning(message)
        elif support_info == 'info':
            self.info(message)
        else:
            self.sysexit("Error, unknown support_info level '%s' for module '%s'" % full_name.asString())

    def _checkConsoleMode(self, full_name, console):
        if False:
            print('Hello World!')
        if console == 'no':
            if shallDisableConsoleWindow() is not True:
                self.sysexit("Error, when using '%s', you have to use '--disable-console' option." % full_name)
        elif console == 'yes':
            pass
        elif console == 'recommend':
            if shallDisableConsoleWindow() is None:
                if isMacOS():
                    downside_message = 'Otherwise high resolution will not be available and a terminal window will open'
                else:
                    downside_message = 'Otherwise a terminal window will open'
                self.info("Note, when using '%s', consider using '--disable-console' option. %s. Howeverfor debugging, terminal output is the easiest way to see informative traceback and error information, so delay this until your program working and remove once you find it non-working, and use '--enable-console' to make it explicit and not see this message." % (full_name, downside_message))
        else:
            self.sysexitIllegalOptionValue(full_name, 'console', console)

    def _checkMacOSBundleMode(self, full_name, macos_bundle):
        if False:
            print('Hello World!')
        if macos_bundle == 'yes':
            if isStandaloneMode() and (not shallCreateAppBundle()):
                self.sysexit("Error, package '%s' requires '--macos-create-app-bundle' to be used or else it cannot work." % full_name)
        elif macos_bundle == 'no':
            pass
        elif macos_bundle == 'recommend':
            pass
        else:
            self.sysexitIllegalOptionValue(full_name, 'macos_bundle', macos_bundle)

    def _checkMacOSBundleOnefileMode(self, full_name, macos_bundle_as_onefile):
        if False:
            for i in range(10):
                print('nop')
        if macos_bundle_as_onefile == 'yes':
            if isStandaloneMode() and shallCreateAppBundle() and (not isOnefileMode()):
                self.sysexit("Error, package '%s' requires '--onefile' to be used on top of '--macos-create-app-bundle' or else it cannot work." % full_name)
        elif macos_bundle_as_onefile == 'no':
            pass
        else:
            self.sysexitIllegalOptionValue(full_name, 'macos_bundle_onefile_mode', macos_bundle_as_onefile)

    def getImplicitImports(self, module):
        if False:
            return 10
        full_name = module.getFullName()
        for options_config in self.config.get(full_name, section='options'):
            for check in options_config.get('checks', ()):
                condition = check.get('when', 'True')
                if self.evaluateCondition(full_name=full_name, condition=condition):
                    self._checkSupportedVersion(full_name=full_name, support_info=check.get('support_info', 'ignore'), description=check.get('description', 'not given'), condition=condition)
                    if mayDisableConsoleWindow():
                        self._checkConsoleMode(full_name=full_name, console=check.get('console', 'yes'))
                    if isMacOS():
                        self._checkMacOSBundleMode(full_name=full_name, macos_bundle=check.get('macos_bundle', 'no'))
                        self._checkMacOSBundleOnefileMode(full_name=full_name, macos_bundle_as_onefile=check.get('macos_bundle_as_onefile', 'no'))
        return ()