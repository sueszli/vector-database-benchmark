""" Plugin to pre-process PMW for inclusion.

"""
import os
import re
from nuitka import Options
from nuitka.__past__ import StringIO
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.utils.FileOperations import getFileContents, listDir
files = ['Dialog', 'TimeFuncs', 'Balloon', 'ButtonBox', 'EntryField', 'Group', 'LabeledWidget', 'MainMenuBar', 'MenuBar', 'MessageBar', 'MessageDialog', 'NoteBook', 'OptionMenu', 'PanedWidget', 'PromptDialog', 'RadioSelect', 'ScrolledCanvas', 'ScrolledField', 'ScrolledFrame', 'ScrolledListBox', 'ScrolledText', 'HistoryText', 'SelectionDialog', 'TextDialog', 'TimeCounter', 'AboutDialog', 'ComboBox', 'ComboBoxDialog', 'Counter', 'CounterDialog']

class NuitkaPluginPmw(NuitkaPluginBase):
    plugin_name = 'pmw-freezer'
    plugin_desc = "Required by the 'Pmw' package."

    def __init__(self, need_blt, need_color):
        if False:
            for i in range(10):
                print('nop')
        self.need_blt = need_blt
        self.need_color = need_color

    @classmethod
    def addPluginCommandLineOptions(cls, group):
        if False:
            i = 10
            return i + 15
        group.add_option('--include-pmw-blt', action='store_true', dest='need_blt', default=False, help="Should 'Pmw.Blt' not be included, Default is to include it.")
        group.add_option('--include-pmw-color', action='store_true', dest='need_color', default=False, help="Should 'Pmw.Color' not be included, Default is to include it.")

    def onModuleSourceCode(self, module_name, source_filename, source_code):
        if False:
            while True:
                i = 10
        if module_name == 'Pmw':
            pmw_path = self.locateModule(module_name=module_name)
            return self._packagePmw(pmw_path)
        return source_code

    def _packagePmw(self, pmw_path):
        if False:
            print('Hello World!')
        self.info('Packaging Pmw into single module fpor freezing.')

        def _hasLoader(dirname):
            if False:
                i = 10
                return i + 15
            if re.search('^Pmw_[0-9]_[0-9](_[0-9])?$', dirname) is not None:
                for suffix in ('.py', '.pyc', '.pyo'):
                    path = os.path.join(pmw_path, dirname, 'lib', 'PmwLoader' + suffix)
                    if os.path.isfile(path):
                        return 1
            return 0
        candidates = []
        for (_fullpath, candidate) in listDir(pmw_path):
            if _hasLoader(candidate):
                candidates.append(candidate)
        candidates.sort()
        candidates.reverse()
        if not candidates:
            self.sysexit('Error, cannot find any Pmw versions.')
        self.info('Found the following Pmw version candidates %s.' % ','.join(candidates))
        candidate = os.path.join(pmw_path, candidates[0], 'lib')
        version = candidates[0][4:].replace('_', '.')
        self.info('Picked version %s.' % version)
        return self._packagePmw2(candidate, version)

    def _packagePmw2(self, srcdir, version):
        if False:
            while True:
                i = 10

        def mungeFile(filename):
            if False:
                i = 10
                return i + 15
            filename = 'Pmw' + filename + '.py'
            text = getFileContents(os.path.join(srcdir, filename))
            text = re.sub('import Pmw\\>', '', text)
            text = re.sub('INITOPT = Pmw.INITOPT', '', text)
            text = re.sub('\\<Pmw\\.', '', text)
            text = '\n' + '#' * 70 + '\n' + '### File: ' + filename + '\n' + text
            return text
        color_code = '\nfrom . import PmwColor\nColor = PmwColor\ndel PmwColor\n'
        blt_code = '\nfrom . import PmwBlt\nBlt = PmwBlt\ndel PmwBlt\n'
        ignore_blt_code = '\n_bltImported = 1\n_bltbusyOK = 0\n'
        extra_code = "\n\n### Loader functions:\n\n_VERSION = '%s'\n\ndef setversion(version):\n    if version != _VERSION:\n        raise ValueError('Dynamic versioning not available')\n\ndef setalphaversions(*alpha_versions):\n    if alpha_versions != ():\n        raise ValueError('Dynamic versioning not available')\n\ndef version(alpha = 0):\n    if alpha:\n        return ()\n    else:\n        return _VERSION\n\ndef installedversions(alpha = 0):\n    if alpha:\n        return ()\n    else:\n        return (_VERSION,)\n\n"
        outfile = StringIO()
        if self.need_color:
            outfile.write(color_code)
        if self.need_blt:
            outfile.write(blt_code)
        outfile.write(extra_code % version)
        text = mungeFile('Base')
        text = re.sub('from . import PmwLogicalFont', '', text)
        text = re.sub('import PmwLogicalFont', '', text)
        text = re.sub('PmwLogicalFont._font_initialise', '_font_initialise', text)
        outfile.write(text)
        if not self.need_blt:
            outfile.write(ignore_blt_code)
        files.append('LogicalFont')
        for filename in files:
            text = mungeFile(filename)
            outfile.write(text)
        return outfile.getvalue()

class NuitkaPluginDetectorPmw(NuitkaPluginBase):
    detector_for = NuitkaPluginPmw

    @classmethod
    def isRelevant(cls):
        if False:
            while True:
                i = 10
        return Options.isStandaloneMode()

    def onModuleDiscovered(self, module):
        if False:
            print('Hello World!')
        if module.getFullName() == 'Pmw':
            self.warnUnusedPlugin('Proper freezing of Pmw package.')