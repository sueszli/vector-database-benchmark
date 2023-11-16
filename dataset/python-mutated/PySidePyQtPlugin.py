""" Standard plug-in to make PyQt and PySide work well in standalone mode.

To run properly, these need the Qt plugins copied along, which have their
own dependencies.
"""
import os
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.Options import isOnefileMode, isStandaloneMode, shallCreateAppBundle
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.plugins.Plugins import getActiveQtPlugin, getOtherGUIBindingNames, getQtBindingNames, getQtPluginNames, hasActiveGuiPluginForBinding
from nuitka.PythonFlavors import isAnacondaPython
from nuitka.PythonVersions import python_version
from nuitka.utils.Distributions import getDistributionFromModuleName, getDistributionInstallerName, getDistributionName
from nuitka.utils.FileOperations import getFileList, listDir
from nuitka.utils.ModuleNames import ModuleName
from nuitka.utils.Utils import getArchitecture, isMacOS, isWin32Windows

class NuitkaPluginQtBindingsPluginBase(NuitkaPluginBase):
    plugin_gui_toolkit = True
    binding_name = None
    warned_about = set()

    def __init__(self, include_qt_plugins, noinclude_qt_plugins, no_qt_translations):
        if False:
            print('Hello World!')
        self.distribution = getDistributionFromModuleName(self.binding_name)
        if self.distribution is None:
            self.sysexit('Error, failed to locate the %s installation.' % self.binding_name)
        self.distribution_name = getDistributionName(self.distribution)
        self.installer_name = getDistributionInstallerName(self.distribution_name)
        self.qt_plugins_dirs = None
        sensible_qt_plugins = self._getSensiblePlugins()
        include_qt_plugins = OrderedSet(sum([value.split(',') for value in include_qt_plugins], []))
        if 'sensible' in include_qt_plugins:
            include_qt_plugins.remove('sensible')
        self.qt_plugins = sensible_qt_plugins
        self.qt_plugins.update(include_qt_plugins)
        for noinclude_qt_plugin in noinclude_qt_plugins:
            self.qt_plugins.discard(noinclude_qt_plugin)
        self.no_qt_translations = no_qt_translations
        self.web_engine_done_binaries = False
        self.web_engine_done_data = False
        self.binding_package_name = ModuleName(self.binding_name)
        if self.qt_plugins == set(['none']):
            self.qt_plugins = set()
        assert self.binding_name in getQtBindingNames(), self.binding_name
        assert self.plugin_name in getQtPluginNames()
        active_qt_plugin_name = getActiveQtPlugin()
        if active_qt_plugin_name is not None:
            self.sysexit("Error, conflicting plugin '%s', you can only have one enabled." % active_qt_plugin_name)

    @classmethod
    def addPluginCommandLineOptions(cls, group):
        if False:
            print('Hello World!')
        group.add_option('--include-qt-plugins', action='append', dest='include_qt_plugins', default=[], help='Which Qt plugins to include. These can be big with dependencies, so\nby default only the "sensible" ones are included, but you can also put\n"all" or list them individually. If you specify something that does\nnot exist, a list of all available will be given.')
        group.add_option('--noinclude-qt-plugins', action='append', dest='noinclude_qt_plugins', default=[], help='Which Qt plugins to not include. This removes things, so you can\nask to include "all" and selectively remove from there, or even\nfrom the default sensible list.')
        group.add_option('--noinclude-qt-translations', action='store_true', dest='no_qt_translations', default=False, help='Include Qt translations with QtWebEngine if used. These can be a lot\nof files that you may not want to be included.')

    def _getQmlTargetDir(self):
        if False:
            return 10
        'Where does the Qt bindings package expect the QML files.'
        return os.path.join(self.binding_name, 'qml')

    def _isUsingMacOSFrameworks(self):
        if False:
            return 10
        'Is this a framework based build, or one that shared more commonality with Linux'
        if isMacOS() and self.binding_name in ('PySide6', 'PySide2'):
            return os.path.exists(os.path.join(self._getQtInformation().data_path, 'lib/QtWebEngineCore.framework'))
        else:
            return False

    def _getWebEngineResourcesTargetDir(self):
        if False:
            for i in range(10):
                print('nop')
        'Where does the Qt bindings package expect the resources files.'
        if isWin32Windows():
            if self.binding_name in ('PySide2', 'PyQt5'):
                return 'resources'
            else:
                return '.'
        elif self.binding_name in ('PySide2', 'PySide6', 'PyQt6'):
            return '.'
        elif self.binding_name == 'PyQt5':
            return 'resources'
        else:
            assert False

    def _getTranslationsTargetDir(self):
        if False:
            for i in range(10):
                print('nop')
        'Where does the Qt bindings package expect the translation files.'
        if isMacOS():
            return os.path.join(self.binding_name, 'Qt', 'translations')
        elif isWin32Windows():
            if self.binding_name in ('PySide2', 'PyQt5'):
                return 'translations'
            elif self.binding_name == 'PyQt6':
                return '.'
            else:
                return os.path.join(self.binding_name, 'translations')
        elif self.binding_name in ('PySide2', 'PySide6', 'PyQt6'):
            return '.'
        elif self.binding_name == 'PyQt5':
            return 'translations'
        else:
            assert False

    @staticmethod
    def _getWebEngineTargetDir():
        if False:
            while True:
                i = 10
        'Where does the Qt bindings package expect the web process executable.'
        return '.'

    def _getSensiblePlugins(self):
        if False:
            i = 10
            return i + 15
        return OrderedSet(tuple((family for family in ('imageformats', 'iconengines', 'mediaservice', 'printsupport', 'platforms', 'platformthemes', 'styles', 'wayland-shell-integration', 'wayland-decoration-client', 'wayland-graphics-integration-client', 'egldeviceintegrations', 'xcbglintegrations', 'tls') if self.hasPluginFamily(family))))

    def getQtPluginsSelected(self):
        if False:
            for i in range(10):
                print('nop')
        return self.qt_plugins

    def hasQtPluginSelected(self, plugin_name):
        if False:
            for i in range(10):
                print('nop')
        selected = self.getQtPluginsSelected()
        return 'all' in selected or plugin_name in selected

    def _getQtInformation(self):
        if False:
            return 10

        def applyBindingName(template):
            if False:
                print('Hello World!')
            return template % {'binding_name': self.binding_name}

        def getLocationQueryCode(path_name):
            if False:
                return 10
            if self.binding_name == 'PyQt6':
                template = '%(binding_name)s.QtCore.QLibraryInfo.path(%(binding_name)s.QtCore.QLibraryInfo.LibraryPath.%(path_name)s)'
            else:
                template = '%(binding_name)s.QtCore.QLibraryInfo.location(%(binding_name)s.QtCore.QLibraryInfo.%(path_name)s)'
            return template % {'binding_name': self.binding_name, 'path_name': path_name}
        setup_codes = applyBindingName('\nimport os\nimport %(binding_name)s.QtCore\n')
        info = self.queryRuntimeInformationMultiple(info_name=applyBindingName('%(binding_name)s_info'), setup_codes=setup_codes, values=(('library_paths', applyBindingName('%(binding_name)s.QtCore.QCoreApplication.libraryPaths()')), ('guess_path1', applyBindingName("os.path.join(os.path.dirname(%(binding_name)s.__file__), 'plugins')")), ('guess_path2', applyBindingName("os.path.join(os.path.dirname(%(binding_name)s.__file__), '..', '..', '..', 'Library', 'plugins')")), ('version', applyBindingName('%(binding_name)s.__version_info__' if 'PySide' in self.binding_name else '%(binding_name)s.QtCore.PYQT_VERSION_STR')), ('nuitka_patch_level', applyBindingName("getattr(%(binding_name)s, '_nuitka_patch_level', 0)")), ('translations_path', getLocationQueryCode('TranslationsPath')), ('library_executables_path', getLocationQueryCode('LibraryExecutablesPath')), ('data_path', getLocationQueryCode('DataPath'))))
        if info is None:
            self.sysexit("Error, it seems '%s' is not installed or broken." % self.binding_name)
        return info

    def _getBindingVersion(self):
        if False:
            while True:
                i = 10
        'Get the version of the binding in tuple digit form, e.g. (6,0,3)'
        return self._getQtInformation().version

    def _getNuitkaPatchLevel(self):
        if False:
            for i in range(10):
                print('nop')
        'Does it include the Nuitka patch, i.e. is a self-built one with it applied.'
        return self._getQtInformation().nuitka_patch_level

    def _getTranslationsPath(self):
        if False:
            i = 10
            return i + 15
        'Get the path to the Qt translations.'
        return self._getQtInformation().translations_path

    def _getWebEngineResourcesPath(self):
        if False:
            return 10
        'Get the path to the Qt web engine resources.'
        if self._isUsingMacOSFrameworks():
            return os.path.join(self._getQtInformation().data_path, 'lib/QtWebEngineCore.framework/Resources')
        else:
            return os.path.join(self._getQtInformation().data_path, 'resources')

    def _getWebEngineExecutablePath(self):
        if False:
            return 10
        'Get the path to QtWebEngine binary.'
        return os.path.normpath(self._getQtInformation().library_executables_path)

    def getQtPluginDirs(self):
        if False:
            print('Hello World!')
        if self.qt_plugins_dirs is not None:
            return self.qt_plugins_dirs
        qt_info = self._getQtInformation()
        self.qt_plugins_dirs = qt_info.library_paths
        if not self.qt_plugins_dirs and os.path.exists(qt_info.guess_path1):
            self.qt_plugins_dirs.append(qt_info.guess_path1)
        if not self.qt_plugins_dirs and os.path.exists(qt_info.guess_path2):
            self.qt_plugins_dirs.append(qt_info.guess_path2)
        self.qt_plugins_dirs = [os.path.normpath(dirname) for dirname in self.qt_plugins_dirs]
        self.qt_plugins_dirs = tuple(sorted(set(self.qt_plugins_dirs)))
        if not self.qt_plugins_dirs:
            self.warning("Couldn't detect Qt plugin directories.")
        return self.qt_plugins_dirs

    def _getQtBinDirs(self):
        if False:
            while True:
                i = 10
        for plugin_dir in self.getQtPluginDirs():
            if 'PyQt' in self.binding_name:
                qt_bin_dir = os.path.normpath(os.path.join(plugin_dir, '..', 'bin'))
                if os.path.isdir(qt_bin_dir):
                    yield qt_bin_dir
            else:
                qt_bin_dir = os.path.normpath(os.path.join(plugin_dir, '..'))
                yield qt_bin_dir

    def hasPluginFamily(self, family):
        if False:
            i = 10
            return i + 15
        return any((os.path.isdir(os.path.join(plugin_dir, family)) for plugin_dir in self.getQtPluginDirs()))

    def _getQmlDirectory(self):
        if False:
            for i in range(10):
                print('nop')
        for plugin_dir in self.getQtPluginDirs():
            qml_plugin_dir = os.path.normpath(os.path.join(plugin_dir, '..', 'qml'))
            if os.path.exists(qml_plugin_dir):
                return qml_plugin_dir
        self.sysexit('Error, no such Qt plugin family: qml')

    def _getQmlFileList(self, dlls):
        if False:
            while True:
                i = 10
        qml_plugin_dir = self._getQmlDirectory()
        datafile_suffixes = ('.qml', '.qmlc', '.qmltypes', '.js', '.jsc', '.json', '.png', '.ttf', '.metainfo', '.mesh', '.frag', 'qmldir')
        if dlls:
            ignore_suffixes = datafile_suffixes
            only_suffixes = ()
        else:
            ignore_suffixes = ()
            only_suffixes = datafile_suffixes
        return getFileList(qml_plugin_dir, ignore_suffixes=ignore_suffixes, only_suffixes=only_suffixes)

    def _findQtPluginDLLs(self):
        if False:
            for i in range(10):
                print('nop')
        for qt_plugins_dir in self.getQtPluginDirs():
            for filename in getFileList(qt_plugins_dir):
                filename_relative = os.path.relpath(filename, start=qt_plugins_dir)
                qt_plugin_name = filename_relative.split(os.path.sep, 1)[0]
                if not self.hasQtPluginSelected(qt_plugin_name):
                    continue
                yield self.makeDllEntryPoint(source_path=filename, dest_path=os.path.join(self.getQtPluginTargetPath(), filename_relative), module_name=self.binding_package_name, package_name=self.binding_package_name, reason='qt plugin')

    def _getChildNamed(self, *child_names):
        if False:
            i = 10
            return i + 15
        for child_name in child_names:
            return ModuleName(self.binding_name).getChildNamed(child_name)

    def getImplicitImports(self, module):
        if False:
            while True:
                i = 10
        full_name = module.getFullName()
        (top_level_package_name, child_name) = full_name.splitPackageName()
        if top_level_package_name != self.binding_name:
            return
        if child_name == 'QtCore' and 'PyQt' in self.binding_name:
            if python_version < 768:
                yield 'atexit'
            yield 'sip'
            yield self._getChildNamed('sip')
        if child_name in ('QtGui', 'QtAssistant', 'QtDBus', 'QtDeclarative', 'QtSql', 'QtDesigner', 'QtHelp', 'QtNetwork', 'QtScript', 'QtQml', 'QtGui', 'QtScriptTools', 'QtSvg', 'QtTest', 'QtWebKit', 'QtOpenGL', 'QtXml', 'QtXmlPatterns', 'QtPrintSupport', 'QtNfc', 'QtWebKitWidgets', 'QtBluetooth', 'QtMultimediaWidgets', 'QtQuick', 'QtWebChannel', 'QtWebSockets', 'QtX11Extras', '_QOpenGLFunctions_2_0', '_QOpenGLFunctions_2_1', '_QOpenGLFunctions_4_1_Core'):
            yield self._getChildNamed('QtCore')
        if child_name in ('QtDeclarative', 'QtWebKit', 'QtXmlPatterns', 'QtQml', 'QtPrintSupport', 'QtWebKitWidgets', 'QtMultimedia', 'QtMultimediaWidgets', 'QtQuick', 'QtQuickWidgets', 'QtWebSockets', 'QtWebEngineWidgets'):
            yield self._getChildNamed('QtNetwork')
        if child_name == 'QtWebEngineWidgets':
            yield self._getChildNamed('QtWebEngineCore')
            yield self._getChildNamed('QtWebChannel')
            yield self._getChildNamed('QtPrintSupport')
        elif child_name == 'QtScriptTools':
            yield self._getChildNamed('QtScript')
        elif child_name in ('QtWidgets', 'QtDeclarative', 'QtDesigner', 'QtHelp', 'QtScriptTools', 'QtSvg', 'QtTest', 'QtWebKit', 'QtPrintSupport', 'QtWebKitWidgets', 'QtMultimedia', 'QtMultimediaWidgets', 'QtOpenGL', 'QtQuick', 'QtQuickWidgets', 'QtSql', '_QOpenGLFunctions_2_0', '_QOpenGLFunctions_2_1', '_QOpenGLFunctions_4_1_Core'):
            yield self._getChildNamed('QtGui')
        if child_name in ('QtDesigner', 'QtHelp', 'QtTest', 'QtPrintSupport', 'QtSvg', 'QtOpenGL', 'QtWebKitWidgets', 'QtMultimediaWidgets', 'QtQuickWidgets', 'QtSql'):
            yield self._getChildNamed('QtWidgets')
        if child_name in ('QtPrintSupport',):
            yield self._getChildNamed('QtSvg')
        if child_name in ('QtWebKitWidgets',):
            yield self._getChildNamed('QtWebKit')
            yield self._getChildNamed('QtPrintSupport')
        if child_name in ('QtMultimediaWidgets',):
            yield self._getChildNamed('QtMultimedia')
        if child_name in ('QtQuick', 'QtQuickWidgets'):
            yield self._getChildNamed('QtQml')
            yield self._getChildNamed('QtOpenGL')
        if child_name in ('QtQuickWidgets', 'QtQml', 'QtQuickControls2'):
            yield self._getChildNamed('QtQuick')
        if child_name == 'Qt':
            yield self._getChildNamed('QtCore')
            yield self._getChildNamed('QtDBus')
            yield self._getChildNamed('QtGui')
            yield self._getChildNamed('QtNetwork')
            yield self._getChildNamed('QtNetworkAuth')
            yield self._getChildNamed('QtSensors')
            yield self._getChildNamed('QtSerialPort')
            yield self._getChildNamed('QtMultimedia')
            yield self._getChildNamed('QtQml')
            yield self._getChildNamed('QtWidgets')
        if child_name == 'QtUiTools':
            yield self._getChildNamed('QtGui')
            yield self._getChildNamed('QtXml')
        if full_name == 'phonon':
            yield self._getChildNamed('QtGui')

    def createPostModuleLoadCode(self, module):
        if False:
            while True:
                i = 10
        "Create code to load after a module was successfully imported.\n\n        For Qt we need to set the library path to the distribution folder\n        we are running from. The code is immediately run after the code\n        and therefore makes sure it's updated properly.\n        "
        if not isStandaloneMode():
            return
        full_name = module.getFullName()
        if full_name == '%s.QtCore' % self.binding_name:
            code = 'from __future__ import absolute_import\n\nfrom %(package_name)s import QCoreApplication\nimport os\n\nqt_plugins_path = %(qt_plugins_path)s\n\nif qt_plugins_path is not None:\n    QCoreApplication.setLibraryPaths(\n        [\n            os.path.join(\n                os.path.dirname(__file__),\n                "..",\n                %(qt_plugins_path)s\n            )\n        ]\n    )\n\nos.environ["QML2_IMPORT_PATH"] = os.path.join(\n    os.path.dirname(__file__),\n    "qml"\n)\n' % {'package_name': full_name, 'qt_plugins_path': repr(None if self.isDefaultQtPluginTargetPath() else self.getQtPluginTargetPath())}
            yield (code, 'Setting Qt library path to distribution folder. We need to avoid loading target\nsystem Qt plugins, which may be from another Qt version.')

    def isQtWebEngineModule(self, full_name):
        if False:
            print('Hello World!')
        return full_name in (self.binding_name + '.QtWebEngine', self.binding_name + '.QtWebEngineCore')

    def createPreModuleLoadCode(self, module):
        if False:
            return 10
        "Method called when a module is being imported.\n\n        Notes:\n            If full name equals to the binding we insert code to include the dist\n            folder in the 'PATH' environment variable (on Windows only).\n\n        Args:\n            module: the module object\n        Returns:\n            Code to insert and descriptive text (tuple), or (None, None).\n        "
        if not isStandaloneMode():
            return
        full_name = module.getFullName()
        if full_name == self.binding_name and isWin32Windows():
            code = 'import os\npath = os.environ.get("PATH", "")\nif not path.startswith(__nuitka_binary_dir):\n    os.environ["PATH"] = __nuitka_binary_dir + ";" + path\n'
            yield (code, "Adding binary folder to runtime 'PATH' environment variable for proper Qt loading.")
        if self.isQtWebEngineModule(full_name) and self._isUsingMacOSFrameworks():
            code = '\nimport os\nos.environ["QTWEBENGINEPROCESS_PATH"] = os.path.join(\n    __nuitka_binary_dir,\n    "%(web_engine_process_path)s"\n)\nos.environ["QTWEBENGINE_LOCALES_PATH"] = os.path.join(\n    __nuitka_binary_dir,\n    "qtwebengine_locales"\n)\nos.environ["QTWEBENGINE_DISABLE_SANDBOX"]="1"\n' % {'web_engine_process_path': '%s/Qt/lib/QtWebEngineCore.framework/Helpers/QtWebEngineProcess.app/Contents/MacOS/QtWebEngineProcess' % self.binding_name}
            yield (code, "Setting WebEngine binary and translation path'.")

    def _handleWebEngineDataFiles(self):
        if False:
            i = 10
            return i + 15
        if self.web_engine_done_data:
            return
        yield self.makeIncludedGeneratedDataFile(data='[Paths]\nPrefix = .\n', dest_path='qt6.conf' if '6' in self.binding_name else 'qt.conf', reason='QtWebEngine needs Qt configuration file')
        if self._isUsingMacOSFrameworks():
            yield self._handleWebEngineDataFilesMacOSFrameworks()
        else:
            yield self._handleWebEngineDataFilesGeneric()
        self.web_engine_done_data = True

    def _handleWebEngineDataFilesMacOSFrameworks(self):
        if False:
            while True:
                i = 10
        if not shallCreateAppBundle():
            self.sysexit('Error, cannot include required Qt WebEngine binaries unless in an application bundle.')
        resources_dir = self._getWebEngineResourcesPath()
        for filename in getFileList(resources_dir):
            filename_relative = os.path.relpath(filename, resources_dir)
            if not isOnefileMode():
                yield self.makeIncludedAppBundleResourceFile(source_path=filename, dest_path=filename_relative, reason='Qt WebEngine resources')
            yield self.makeIncludedDataFile(source_path=filename, dest_path=filename_relative, reason='Qt WebEngine resources')
        used_frameworks = ['QtWebEngineCore', 'QtCore', 'QtQuick', 'QtQml', 'QtQmlModels', 'QtNetwork', 'QtGui', 'QtWebChannel', 'QtPositioning']
        if self.binding_name in ('PySide6', 'PyQt6'):
            used_frameworks += ['QtOpenGL', 'QtDBus']
        for used_framework in used_frameworks:
            yield self.makeIncludedAppBundleFramework(source_path=os.path.join(self._getQtInformation().data_path, 'lib'), framework_name=used_framework, reason='Qt WebEngine dependency')

    def makeIncludedAppBundleFramework(self, source_path, framework_name, reason, tags=''):
        if False:
            return 10
        framework_basename = framework_name + '.framework'
        framework_path = os.path.join(source_path, framework_basename)
        for filename in getFileList(framework_path):
            filename_relative = os.path.relpath(filename, framework_path)
            yield self.makeIncludedDataFile(source_path=filename, dest_path=os.path.join(self.binding_name, 'Qt', 'lib', framework_basename, filename_relative), reason=reason, tags=tags)

    def _handleWebEngineDataFilesGeneric(self):
        if False:
            print('Hello World!')
        resources_dir = self._getWebEngineResourcesPath()
        for filename in getFileList(resources_dir):
            filename_relative = os.path.relpath(filename, resources_dir)
            yield self.makeIncludedDataFile(source_path=filename, dest_path=os.path.join(self._getWebEngineResourcesTargetDir(), filename_relative), reason='Qt resources')
        if not self.no_qt_translations:
            translations_path = self._getTranslationsPath()
            dest_path = self._getTranslationsTargetDir()
            for filename in getFileList(translations_path):
                filename_relative = os.path.relpath(filename, translations_path)
                yield self.makeIncludedDataFile(source_path=filename, dest_path=os.path.join(dest_path, filename_relative), reason='Qt translation', tags='translation')

    def considerDataFiles(self, module):
        if False:
            for i in range(10):
                print('nop')
        full_name = module.getFullName()
        if full_name == self.binding_name and ('qml' in self.getQtPluginsSelected() or 'all' in self.getQtPluginsSelected()):
            qml_plugin_dir = self._getQmlDirectory()
            qml_target_dir = self._getQmlTargetDir()
            self.info("Including Qt plugins 'qml' below '%s'." % qml_target_dir)
            for filename in self._getQmlFileList(dlls=False):
                filename_relative = os.path.relpath(filename, qml_plugin_dir)
                yield self.makeIncludedDataFile(source_path=filename, dest_path=os.path.join(qml_target_dir, filename_relative), reason='Qt QML datafile', tags='qml')
        elif self.isQtWebEngineModule(full_name):
            yield self._handleWebEngineDataFiles()

    def _getExtraBinariesWebEngineGeneric(self, full_name):
        if False:
            return 10
        if self.web_engine_done_binaries:
            return
        self.info('Including QtWebEngine executable.')
        qt_web_engine_dir = self._getWebEngineExecutablePath()
        for (filename, filename_relative) in listDir(qt_web_engine_dir):
            if filename_relative.startswith('QtWebEngineProcess'):
                yield self.makeExeEntryPoint(source_path=filename, dest_path=os.path.normpath(os.path.join(self._getWebEngineTargetDir(), filename_relative)), module_name=full_name, package_name=full_name, reason="needed by '%s'" % full_name.asString())
                break
        else:
            self.sysexit("Error, cannot locate 'QtWebEngineProcess' executable at '%s'." % qt_web_engine_dir)
        self.web_engine_done_binaries = True

    def getQtPluginTargetPath(self):
        if False:
            while True:
                i = 10
        if self.binding_name == 'PyQt6':
            return os.path.join(self.binding_name, 'Qt6', 'plugins')
        else:
            return os.path.join(self.binding_name, 'qt-plugins')

    def isDefaultQtPluginTargetPath(self):
        if False:
            i = 10
            return i + 15
        return self.binding_name == 'PyQt6'

    def getExtraDlls(self, module):
        if False:
            i = 10
            return i + 15
        full_name = module.getFullName()
        if full_name == self.binding_name:
            if not self.getQtPluginDirs():
                self.sysexit("Error, failed to detect '%s' plugin directories." % self.binding_name)
            self.info("Including Qt plugins '%s' below '%s'." % (','.join(sorted((x for x in self.getQtPluginsSelected() if x != 'xml'))), self.getQtPluginTargetPath()))
            for r in self._findQtPluginDLLs():
                yield r
            if isWin32Windows():
                qt_bin_files = sum((getFileList(qt_bin_dir) for qt_bin_dir in self._getQtBinDirs()), [])
                count = 0
                for filename in qt_bin_files:
                    basename = os.path.basename(filename).lower()
                    if basename in ('libeay32.dll', 'ssleay32.dll'):
                        yield self.makeDllEntryPoint(source_path=filename, dest_path=basename, module_name=full_name, package_name=full_name, reason="needed by '%s'" % full_name.asString())
                        count += 1
                self.reportFileCount(full_name, count, section='OpenSSL')
            if 'qml' in self.getQtPluginsSelected() or 'all' in self.getQtPluginsSelected():
                qml_plugin_dir = self._getQmlDirectory()
                qml_target_dir = self._getQmlTargetDir()
                for filename in self._getQmlFileList(dlls=True):
                    filename_relative = os.path.relpath(filename, qml_plugin_dir)
                    yield self.makeDllEntryPoint(source_path=filename, dest_path=os.path.join(qml_target_dir, filename_relative), module_name=full_name, package_name=full_name, reason='Qt QML plugin DLL')
                if isWin32Windows():
                    gl_dlls = ('libegl.dll', 'libglesv2.dll', 'opengl32sw.dll')
                    count = 0
                    for filename in qt_bin_files:
                        basename = os.path.basename(filename).lower()
                        if basename in gl_dlls or basename.startswith('d3dcompiler_'):
                            yield self.makeDllEntryPoint(source_path=filename, dest_path=basename, module_name=full_name, package_name=full_name, reason="needed by OpenGL for '%s'" % full_name.asString())
                    self.reportFileCount(full_name, count, section='OpenGL')
        elif full_name == self.binding_name + '.QtNetwork':
            yield self._getExtraBinariesQtNetwork(full_name=full_name)
        elif self.isQtWebEngineModule(full_name):
            if not self._isUsingMacOSFrameworks():
                yield self._getExtraBinariesWebEngineGeneric(full_name=full_name)

    def _getExtraBinariesQtNetwork(self, full_name):
        if False:
            while True:
                i = 10
        if isWin32Windows():
            if self.binding_name == 'PyQt5':
                arch_name = getArchitecture()
                if arch_name == 'x86':
                    arch_suffix = ''
                elif arch_name == 'x86_64':
                    arch_suffix = '-x64'
                else:
                    self.sysexit('Error, unknown architecture encountered, need to add support for %s.' % arch_name)
                for dll_basename in ('libssl-1_1', 'libcrypto-1_1'):
                    dll_filename = dll_basename + arch_suffix + '.dll'
                    for plugin_dir in self._getQtBinDirs():
                        candidate = os.path.join(plugin_dir, dll_filename)
                        if os.path.exists(candidate):
                            yield self.makeDllEntryPoint(source_path=candidate, dest_path=dll_filename, module_name=full_name, package_name=full_name, reason="needed by '%s'" % full_name.asString())
                            break
        else:
            dll_path = self.locateDLL('crypto')
            if dll_path is not None:
                yield self.makeDllEntryPoint(source_path=dll_path, dest_path=os.path.basename(dll_path), module_name=full_name, package_name=full_name, reason="needed by '%s'" % full_name.asString())
            dll_path = self.locateDLL('ssl')
            if dll_path is not None:
                yield self.makeDllEntryPoint(source_path=dll_path, dest_path=os.path.basename(dll_path), module_name=full_name, package_name=full_name, reason="needed by '%s'" % full_name.asString())

    def removeDllDependencies(self, dll_filename, dll_filenames):
        if False:
            i = 10
            return i + 15
        for value in self.getQtPluginDirs():
            if dll_filename.startswith(value):
                for sub_dll_filename in dll_filenames:
                    for badword in ('libKF5', 'libkfontinst', 'libkorganizer', 'libplasma', 'libakregator', 'libdolphin', 'libnoteshared', 'libknotes', 'libsystemsettings', 'libkerfuffle', 'libkaddressbook', 'libkworkspace', 'libkmail', 'libmilou', 'libtaskmanager', 'libkonsole', 'libgwenview', 'libweather_ion'):
                        if os.path.basename(sub_dll_filename).startswith(badword):
                            yield sub_dll_filename

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if False:
            while True:
                i = 10
        top_package_name = module_name.getTopLevelPackageName()
        if isStandaloneMode():
            if top_package_name in getQtBindingNames() and top_package_name != self.binding_name:
                if top_package_name not in self.warned_about:
                    self.info('Unwanted import of \'%(unwanted)s\' that conflicts with \'%(binding_name)s\' encountered, preventing its inclusion. As a result an "ImportError" might be given at run time. Uninstall the module it for fully compatible behavior with the uncompiled code.' % {'unwanted': top_package_name, 'binding_name': self.binding_name})
                    self.warned_about.add(top_package_name)
                return (False, "Not included due to potentially conflicting Qt versions with selected Qt binding '%s'." % self.binding_name)
            if top_package_name in getOtherGUIBindingNames() and (not hasActiveGuiPluginForBinding(top_package_name)):
                return (False, 'Not included due to its plugin not being active, but a Qt plugin is.')

    def onModuleCompleteSet(self, module_set):
        if False:
            return 10
        self.onModuleCompleteSetGUI(module_set=module_set, plugin_binding_name=self.binding_name)

    def onModuleSourceCode(self, module_name, source_filename, source_code):
        if False:
            i = 10
            return i + 15
        'Third party packages that make binding selections.'
        if module_name.hasNamespace('pyqtgraph'):
            source_code = source_code.replace('{QT_LIB.lower()}', self.binding_name.lower())
            source_code = source_code.replace('QT_LIB.lower()', repr(self.binding_name.lower()))
        return source_code

    def onDataFileTags(self, included_datafile):
        if False:
            return 10
        if included_datafile.dest_path.endswith('.qml') and (not self.hasQtPluginSelected('qml')):
            self.warning("Including QML file %s, but not having Qt qml plugins is unlikely to work. Consider using '--include-qt-plugins=qml' to include the necessary files to use it." % included_datafile.dest_path)

class NuitkaPluginPyQt5QtPluginsPlugin(NuitkaPluginQtBindingsPluginBase):
    """This is for plugins of PyQt5.

    When loads an image, it may use a plug-in, which in turn used DLLs,
    which for standalone mode, can cause issues of not having it.
    """
    plugin_name = 'pyqt5'
    plugin_desc = 'Required by the PyQt5 package.'
    binding_name = 'PyQt5'

    def __init__(self, include_qt_plugins, noinclude_qt_plugins, no_qt_translations):
        if False:
            for i in range(10):
                print('nop')
        NuitkaPluginQtBindingsPluginBase.__init__(self, include_qt_plugins=include_qt_plugins, noinclude_qt_plugins=noinclude_qt_plugins, no_qt_translations=no_qt_translations)
        self.warning('For the obsolete PyQt5 the Nuitka support is incomplete. Threading, callbacks to compiled functions, etc. may not be working.', mnemonic='pyqt5')

    def _getQtInformation(self):
        if False:
            i = 10
            return i + 15
        result = NuitkaPluginQtBindingsPluginBase._getQtInformation(self)
        if isAnacondaPython():
            if 'CONDA_PREFIX' in os.environ:
                conda_prefix = os.environ['CONDA_PREFIX']
            elif 'CONDA_PYTHON_EXE' in os.environ:
                conda_prefix = os.path.dirname(os.environ['CONDA_PYTHON_EXE'])
            if conda_prefix is not None:
                values = result._asdict()

                def updateStaticPath(value):
                    if False:
                        print('Hello World!')
                    path_parts = value.split('/')
                    if '_h_env' in path_parts:
                        return os.path.normpath(os.path.join(conda_prefix, *path_parts[path_parts.index('_h_env') + 1:]))
                    else:
                        return value
                for key in ('translations_path', 'library_executables_path', 'data_path'):
                    values[key] = updateStaticPath(values[key])
                result = result.__class__(**values)
        return result

    @classmethod
    def isRelevant(cls):
        if False:
            i = 10
            return i + 15
        return isStandaloneMode()

class NuitkaPluginDetectorPyQt5QtPluginsPlugin(NuitkaPluginBase):
    detector_for = NuitkaPluginPyQt5QtPluginsPlugin

    @classmethod
    def isRelevant(cls):
        if False:
            print('Hello World!')
        return isStandaloneMode()

    def onModuleDiscovered(self, module):
        if False:
            for i in range(10):
                print('nop')
        full_name = module.getFullName()
        if full_name == NuitkaPluginPyQt5QtPluginsPlugin.binding_name + '.QtCore':
            self.warnUnusedPlugin('Inclusion of Qt plugins.')
        elif full_name == 'PyQt4.QtCore':
            self.warning('Support for PyQt4 has been dropped. Please contact Nuitka commercial if you need it.')

class NuitkaPluginPySide2Plugins(NuitkaPluginQtBindingsPluginBase):
    """This is for plugins of PySide2.

    When Qt loads an image, it may use a plug-in, which in turn used DLLs,
    which for standalone mode, can cause issues of not having it.
    """
    plugin_name = 'pyside2'
    plugin_desc = 'Required by the PySide2 package.'
    binding_name = 'PySide2'

    def __init__(self, include_qt_plugins, noinclude_qt_plugins, no_qt_translations):
        if False:
            while True:
                i = 10
        if self._getNuitkaPatchLevel() < 1:
            self.warning('For the standard PySide2 incomplete workarounds are applied. For full support consider provided information.', mnemonic='pyside2')
            if python_version < 864:
                self.sysexit('The standard PySide2 is not supported before CPython <3.6. For full support: https://nuitka.net/pages/pyside2.html')
        NuitkaPluginQtBindingsPluginBase.__init__(self, include_qt_plugins=include_qt_plugins, noinclude_qt_plugins=noinclude_qt_plugins, no_qt_translations=no_qt_translations)

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if False:
            while True:
                i = 10
        if module_name == self.binding_name and self._getNuitkaPatchLevel() < 1:
            return (True, 'Need to monkey patch PySide2 for abstract methods.')
        return NuitkaPluginQtBindingsPluginBase.onModuleEncounter(self, using_module_name=using_module_name, module_name=module_name, module_filename=module_filename, module_kind=module_kind)

    def createPostModuleLoadCode(self, module):
        if False:
            print('Hello World!')
        "Create code to load after a module was successfully imported.\n\n        For Qt we need to set the library path to the distribution folder\n        we are running from. The code is immediately run after the code\n        and therefore makes sure it's updated properly.\n        "
        for result in NuitkaPluginQtBindingsPluginBase.createPostModuleLoadCode(self, module):
            yield result
        if self._getNuitkaPatchLevel() < 1 and module.getFullName() == self.binding_name:
            code = '\n# Make them unique and count them.\nwrapper_count = 0\nimport functools\nimport inspect\n\ndef nuitka_wrap(cls):\n    global wrapper_count\n\n    for attr in cls.__dict__:\n        if attr.startswith("__") and attr.endswith("__"):\n            continue\n\n        value = getattr(cls, attr)\n\n        if type(value).__name__ == "compiled_function":\n            # Only work on overloaded attributes.\n            for base in cls.__bases__:\n                base_value = getattr(base, attr, None)\n\n                if base_value:\n                    module = inspect.getmodule(base_value)\n\n                    # PySide C stuff does this, and we only need to cover that.\n                    if module is None:\n                        break\n            else:\n                continue\n\n            wrapper_count += 1\n            wrapper_name = "_wrapped_function_%s_%d" % (attr, wrapper_count)\n\n            signature = inspect.signature(value)\n\n            # Remove annotations junk that cannot be executed.\n            signature = signature.replace(\n                return_annotation = inspect.Signature.empty,\n                parameters=[\n                    parameter.replace(default=inspect.Signature.empty,annotation=inspect.Signature.empty)\n                    for parameter in\n                    signature.parameters.values()\n                ]\n            )\n\n            v = r\'\'\'\ndef %(wrapper_name)s%(signature)s:\n    return %(wrapper_name)s.func(%(parameters)s)\n            \'\'\' % {\n                    "signature": signature,\n                    "parameters": ",".join(signature.parameters),\n                    "wrapper_name": wrapper_name\n                }\n\n            # TODO: Nuitka does not currently statically optimize this, might change!\n            exec(\n                v,\n                globals(),\n            )\n\n            wrapper = globals()[wrapper_name]\n            wrapper.func = value\n            wrapper.__defaults__ = value.__defaults__\n\n            setattr(cls, attr, wrapper)\n\n    return cls\n\n@classmethod\ndef my_init_subclass(cls, *args):\n    return nuitka_wrap(cls)\n\nimport PySide2.QtCore\nPySide2.QtCore.QAbstractItemModel.__init_subclass__ = my_init_subclass\nPySide2.QtCore.QAbstractTableModel.__init_subclass__ = my_init_subclass\nPySide2.QtCore.QObject.__init_subclass__ = my_init_subclass\n'
            yield (code, 'Monkey patching classes derived from PySide2 base classes to pass PySide2 checks.')

class NuitkaPluginDetectorPySide2Plugins(NuitkaPluginBase):
    detector_for = NuitkaPluginPySide2Plugins

    def onModuleDiscovered(self, module):
        if False:
            print('Hello World!')
        if module.getFullName() == NuitkaPluginPySide2Plugins.binding_name + '.QtCore' and getActiveQtPlugin() is None:
            self.warnUnusedPlugin('Making callbacks work and include Qt plugins.')

class NuitkaPluginPySide6Plugins(NuitkaPluginQtBindingsPluginBase):
    """This is for plugins of PySide6.

    When Qt loads an image, it may use a plug-in, which in turn used DLLs,
    which for standalone mode, can cause issues of not having it.
    """
    plugin_name = 'pyside6'
    plugin_desc = 'Required by the PySide6 package for standalone mode.'
    binding_name = 'PySide6'

    def __init__(self, include_qt_plugins, noinclude_qt_plugins, no_qt_translations):
        if False:
            return 10
        NuitkaPluginQtBindingsPluginBase.__init__(self, include_qt_plugins=include_qt_plugins, noinclude_qt_plugins=noinclude_qt_plugins, no_qt_translations=no_qt_translations)
        if self._getBindingVersion() < (6, 5, 0):
            self.warning("Make sure to use PySide 6.5.0 or higher, otherwise Qt slots won't work in all cases.")
        if self._getBindingVersion() < (6, 1, 2):
            self.warning("Make sure to use PySide 6.1.2 or higher, otherwise Qt callbacks to Python won't work.")

class NuitkaPluginDetectorPySide6Plugins(NuitkaPluginBase):
    detector_for = NuitkaPluginPySide6Plugins

    def onModuleDiscovered(self, module):
        if False:
            i = 10
            return i + 15
        if module.getFullName() == NuitkaPluginPySide6Plugins.binding_name + '.QtCore':
            self.warnUnusedPlugin('Standalone mode support and Qt plugins.')

class NuitkaPluginPyQt6Plugins(NuitkaPluginQtBindingsPluginBase):
    """This is for plugins of PyQt6.

    When Qt loads an image, it may use a plug-in, which in turn used DLLs,
    which for standalone mode, can cause issues of not having it.
    """
    plugin_name = 'pyqt6'
    plugin_desc = 'Required by the PyQt6 package for standalone mode.'
    binding_name = 'PyQt6'

    def __init__(self, include_qt_plugins, noinclude_qt_plugins, no_qt_translations):
        if False:
            for i in range(10):
                print('nop')
        NuitkaPluginQtBindingsPluginBase.__init__(self, include_qt_plugins=include_qt_plugins, noinclude_qt_plugins=noinclude_qt_plugins, no_qt_translations=no_qt_translations)
        self.info('Support for PyQt6 is not perfect, e.g. Qt threading does not work, so prefer PySide6 if you can.')

class NuitkaPluginNoQt(NuitkaPluginBase):
    """This is a plugins for suppression of all Qt binding plugins."""
    plugin_name = 'no-qt'
    plugin_desc = 'Disable all Qt bindings for standalone mode.'
    warned_about = set()

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if False:
            print('Hello World!')
        top_package_name = module_name.getTopLevelPackageName()
        if isStandaloneMode():
            if top_package_name in getQtBindingNames():
                if top_package_name not in self.warned_about:
                    self.info('Unwanted import of \'%(unwanted)s\' that is forbidden encountered, preventing\nits use. As a result an "ImportError" might be given at run time. Uninstall\nit for full compatible behavior with the uncompiled code to debug it.' % {'unwanted': top_package_name})
                    self.warned_about.add(top_package_name)
                return (False, 'Not included due to all Qt bindings disallowed.')