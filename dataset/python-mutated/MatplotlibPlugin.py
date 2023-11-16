""" Matplotlib standard plugin module. """
import os
from nuitka.Options import isStandaloneMode
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.plugins.Plugins import getActiveQtPluginBindingName, hasActivePlugin
from nuitka.utils.Execution import NuitkaCalledProcessError
from nuitka.utils.FileOperations import getFileContentByLine
from nuitka.utils.Jinja2 import renderTemplateFromString

class NuitkaPluginMatplotlib(NuitkaPluginBase):
    """This class represents the main logic of the plugin.

    This is a plugin to ensure scripts using numpy, scipy, matplotlib, pandas,
    scikit-learn, etc. work well in standalone mode.

    While there already are relevant entries in the "ImplicitImports.py" plugin,
    this plugin copies any additional binary or data files required by many
    installations.

    """
    plugin_name = 'matplotlib'
    plugin_desc = "Required for 'matplotlib' module."

    @staticmethod
    def isAlwaysEnabled():
        if False:
            return 10
        'Request to be always enabled.'
        return True

    @classmethod
    def isRelevant(cls):
        if False:
            print('Hello World!')
        'Check whether plugin might be required.\n\n        Returns:\n            True if this is a standalone compilation.\n        '
        return isStandaloneMode()

    def _getMatplotlibInfo(self):
        if False:
            for i in range(10):
                print('nop')
        "Determine the filename of matplotlibrc and the default backend, etc.\n\n        Notes:\n            There might exist a local version outside 'matplotlib/mpl-data' which\n            we then must use instead. Determine its name by asking matplotlib.\n        "
        try:
            info = self.queryRuntimeInformationMultiple(info_name='matplotlib_info', setup_codes='\nfrom matplotlib import matplotlib_fname, get_backend, __version__\ntry:\n    from matplotlib import get_data_path\nexcept ImportError:\n    from matplotlib import _get_data_path as get_data_path\nfrom inspect import getsource\n', values=(('matplotlibrc_filename', 'matplotlib_fname()'), ('backend', 'get_backend()'), ('data_path', 'get_data_path()'), ('matplotlib_version', '__version__')))
        except NuitkaCalledProcessError:
            if 'MPLBACKEND' not in os.environ:
                self.sysexit("Error, failed to detect matplotlib backend. Please set 'MPLBACKEND' environment variable during compilation.", mnemonic='https://matplotlib.org/stable/users/installing/environment_variables_faq.html#envvar-MPLBACKEND')
            raise
        if info is None:
            self.sysexit("Error, it seems 'matplotlib' is not installed or broken.")
        return info

    def considerDataFiles(self, module):
        if False:
            for i in range(10):
                print('nop')
        if module.getFullName() != 'matplotlib':
            return
        matplotlib_info = self._getMatplotlibInfo()
        if not os.path.isdir(matplotlib_info.data_path):
            self.sysexit('mpl-data missing, matplotlib installation appears to be broken')
        self.info("Using %s backend '%s'." % ('configuration file or default' if 'MPLBACKEND' not in os.environ else "as per 'MPLBACKEND' environment variable", matplotlib_info.backend))
        yield self.makeIncludedDataDirectory(source_path=matplotlib_info.data_path, dest_path=os.path.join('matplotlib', 'mpl-data'), ignore_dirs=('sample_data',), ignore_filenames=('matplotlibrc',), reason="package data for 'matplotlib", tags='mpl-data')
        new_lines = []
        found = False
        for line in getFileContentByLine(matplotlib_info.matplotlibrc_filename):
            line = line.rstrip()
            if line.startswith('#') and matplotlib_info.matplotlib_version < '3':
                continue
            new_lines.append(line)
            if line.startswith(('backend ', 'backend:')):
                found = True
        if not found and matplotlib_info.matplotlib_version < '4':
            new_lines.append('backend: %s' % matplotlib_info.backend)
        yield self.makeIncludedGeneratedDataFile(data='\n'.join(new_lines), dest_path=os.path.join('matplotlib', 'mpl-data', 'matplotlibrc'), reason='updated matplotlib config file with backend to use')

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if False:
            for i in range(10):
                print('nop')
        if module_name.hasNamespace('mpl_toolkits'):
            return (True, 'Needed by matplotlib')
        if module_name in ('matplotlib.backends.backend_tk', 'matplotlib.backends.backend_tkagg', 'matplotlib.backend.tkagg'):
            if hasActivePlugin('tk-inter'):
                return (True, 'Needed for tkinter matplotlib backend')

    def createPreModuleLoadCode(self, module):
        if False:
            for i in range(10):
                print('nop')
        'Method called when a module is being imported.\n\n        Notes:\n            If full name equals "matplotlib" we insert code to set the\n            environment variable that e.g. Debian versions of matplotlib\n            use.\n\n        Args:\n            module: the module object\n        Returns:\n            Code to insert and descriptive text (tuple), or (None, None).\n        '
        if module.getFullName() == 'matplotlib':
            code = renderTemplateFromString('\nimport os\nos.environ["MATPLOTLIBDATA"] = os.path.join(__nuitka_binary_dir, "matplotlib", "mpl-data")\nos.environ["MATPLOTLIBRC"] = os.path.join(__nuitka_binary_dir, "matplotlib", "mpl-data", "matplotlibrc")\nos.environ["MPLBACKEND"] = "{{matplotlib_info.backend}}"\n{% if qt_binding_name %}\nos.environ["QT_API"] = "{{qt_binding_name}}"\n{% endif %}\n', matplotlib_info=self._getMatplotlibInfo(), qt_binding_name=getActiveQtPluginBindingName())
            return (code, "Setting environment variables for 'matplotlib' to find package configuration.")