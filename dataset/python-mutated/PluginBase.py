"""
Plugins: Welcome to Nuitka! This is your shortest way to become part of it.

This is to provide the base class for all plugins. Some of which are part of
proper Nuitka, and some of which are waiting to be created and submitted for
inclusion by you.

The base class will serve as documentation. And it will point to examples of
it being used.
"""
import ast
import functools
import inspect
import os
import sys
from nuitka import Options
from nuitka.containers.Namedtuples import makeNamedtupleClass
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.freezer.IncludedDataFiles import decodeDataFileTags, makeIncludedDataDirectory, makeIncludedDataFile, makeIncludedEmptyDirectory, makeIncludedGeneratedDataFile, makeIncludedPackageDataFiles
from nuitka.freezer.IncludedEntryPoints import makeDllEntryPoint, makeExeEntryPoint
from nuitka.ModuleRegistry import addModuleInfluencingCondition, getModuleInclusionInfoByName
from nuitka.Options import hasPythonFlagNoAnnotations, hasPythonFlagNoAsserts, hasPythonFlagNoDocStrings, isDeploymentMode, isStandaloneMode, shallCreateAppBundle, shallMakeModule, shallShowExecutedCommands
from nuitka.PythonFlavors import isAnacondaPython, isDebianPackagePython
from nuitka.PythonVersions import getTestExecutionPythonVersions, python_version
from nuitka.Tracing import plugins_logger
from nuitka.utils.Distributions import getDistributionFromModuleName, getDistributionName, isDistributionCondaPackage
from nuitka.utils.Execution import NuitkaCalledProcessError, check_output
from nuitka.utils.FileOperations import changeFilenameExtension, getFileContents
from nuitka.utils.Importing import isBuiltinModuleName
from nuitka.utils.ModuleNames import ModuleName, makeTriggerModuleName, post_module_load_trigger_name, pre_module_load_trigger_name
from nuitka.utils.SharedLibraries import locateDLL, locateDLLsInDirectory
from nuitka.utils.SlotMetaClasses import getMetaClassBase
from nuitka.utils.Utils import getArchitecture, isLinux, isMacOS, isWin32Windows
_warned_unused_plugins = set()
_package_versions = {}
control_tags = {}

def _convertVersionToTuple(version_str):
    if False:
        print('Hello World!')

    def numberize(v):
        if False:
            while True:
                i = 10
        return int(''.join((d for d in v if d.isdigit())))
    return tuple((numberize(d) for d in version_str.split('.')))

def _getPackageNameFromDistributionName(distribution_name):
    if False:
        i = 10
        return i + 15
    if distribution_name in ('opencv-python', 'opencv-python-headless'):
        return 'cv2'
    elif distribution_name == 'pyobjc':
        return 'objc'
    else:
        return distribution_name

def _getDistributionNameFromPackageName(package_name):
    if False:
        i = 10
        return i + 15
    return getDistributionName(getDistributionFromModuleName(package_name))

def _getPackageVersion(distribution_name):
    if False:
        return 10
    if distribution_name not in _package_versions:
        try:
            if python_version >= 896:
                from importlib.metadata import version
            else:
                from importlib_metadata import version
            result = _convertVersionToTuple(version(distribution_name))
        except ImportError:
            try:
                from pkg_resources import DistributionNotFound, extern, get_distribution
            except ImportError:
                result = None
            else:
                try:
                    result = _convertVersionToTuple(get_distribution(distribution_name).version)
                except DistributionNotFound:
                    result = None
                except extern.packaging.version.InvalidVersion:
                    result = None
        if result is None:
            try:
                result = _convertVersionToTuple(__import__(_getPackageNameFromDistributionName(distribution_name)).__version__)
            except ImportError:
                result = None
        _package_versions[distribution_name] = result
    return _package_versions[distribution_name]

def _isPluginActive(plugin_name):
    if False:
        i = 10
        return i + 15
    from .Plugins import getUserActivatedPluginNames
    return plugin_name in getUserActivatedPluginNames()

class NuitkaPluginBase(getMetaClassBase('Plugin', require_slots=False)):
    """Nuitka base class for all plugins.

    Derive your plugin from "NuitkaPluginBase" please.
    For instructions, see https://github.com/Nuitka/Nuitka/blob/orsiris/UserPlugin-Creation.rst

    Plugins allow to adapt Nuitka's behavior in a number of ways as explained
    below at the individual methods.

    It is used to deal with special requirements some packages may have (e.g. PyQt
    and tkinter), data files to be included (e.g. "certifi"), inserting hidden
    code, coping with otherwise undetectable needs, or issuing messages in
    certain situations.

    A plugin in general must be enabled to be used by Nuitka. This happens by
    specifying "--enable-plugin" (standard plugins) or by "--user-plugin" (user
    plugins) in the Nuitka command line. However, some plugins are always enabled
    and invisible to the user.

    Nuitka comes with a number of "standard" plugins to be enabled as needed.
    What they are can be displayed using "nuitka --plugin-list file.py" (filename
    required but ignored).

    User plugins may be specified (and implicitly enabled) using their Python
    script pathname.
    """
    plugin_name = None

    @staticmethod
    def isAlwaysEnabled():
        if False:
            while True:
                i = 10
        'Request to be always enabled.\n\n        Notes:\n            Setting this to true is only applicable to standard plugins. In\n            this case, the plugin will be enabled upon Nuitka start-up. Any\n            plugin detector class will then be ignored. Method isRelevant() may\n            also be present and can be used to fine-control enabling the\n            plugin: A to-be-enabled, but irrelevant plugin will still not be\n            activated.\n        Returns:\n            True or False\n        '
        return False

    @classmethod
    def isRelevant(cls):
        if False:
            for i in range(10):
                print('nop')
        'Consider if the plugin is relevant.\n\n        Notes:\n            A plugin may only be a needed on a certain OS, or with some options,\n            but this is only a class method, so you will not have much run time\n            information.\n\n        Returns:\n            True or False\n\n        '
        return not cls.isDeprecated()

    @classmethod
    def isDeprecated(cls):
        if False:
            print('Hello World!')
        'Is this a deprecated plugin, i.e. one that has no use anymore.'
        return False

    @classmethod
    def isDetector(cls):
        if False:
            return 10
        'Is this a detection plugin, i.e. one which is only there to inform.'
        return hasattr(cls, 'detector_for')

    @classmethod
    def addPluginCommandLineOptions(cls, group):
        if False:
            while True:
                i = 10
        pass

    def isRequiredImplicitImport(self, module, full_name):
        if False:
            while True:
                i = 10
        'Indicate whether an implicitly imported module should be accepted.\n\n        Notes:\n            You may negate importing a module specified as "implicit import",\n            although this is an unexpected event.\n\n        Args:\n            module: the module object\n            full_name: of the implicitly import module\n        Returns:\n            True or False\n        '
        return True

    def getImplicitImports(self, module):
        if False:
            while True:
                i = 10
        'Return the implicit imports for a given module (iterator).\n\n        Args:\n            module: the module object\n        Yields:\n            implicit imports for the module\n        '
        return ()

    def onModuleSourceCode(self, module_name, source_filename, source_code):
        if False:
            while True:
                i = 10
        'Inspect or modify source code.\n\n        Args:\n            module_name: (str) name of module\n            source_code: (str) its source code\n        Returns:\n            source_code (str)\n        Notes:\n            Default implementation forwards to `checkModuleSourceCode` which is\n            going to allow simply checking the source code without the need to\n            pass it back.\n        '
        self.checkModuleSourceCode(module_name, source_code)
        return source_code

    def checkModuleSourceCode(self, module_name, source_code):
        if False:
            return 10
        'Inspect source code.\n\n        Args:\n            module_name: (str) name of module\n            source_code: (str) its source code\n        Returns:\n            None\n        '

    def onFrozenModuleBytecode(self, module_name, is_package, bytecode):
        if False:
            print('Hello World!')
        'Inspect or modify frozen module byte code.\n\n        Args:\n            module_name: (str) name of module\n            is_package: (bool) True indicates a package\n            bytecode: (bytes) byte code\n        Returns:\n            bytecode (bytes)\n        '
        return bytecode

    @staticmethod
    def createPreModuleLoadCode(module):
        if False:
            print('Hello World!')
        'Create code to execute before importing a module.\n\n        Notes:\n            Called by @onModuleDiscovered.\n\n        Args:\n            module: the module object\n        Returns:\n            None (does not apply, default)\n            tuple (code, documentary string)\n            tuple (code, documentary string, flags)\n        '
        return None

    @staticmethod
    def createPostModuleLoadCode(module):
        if False:
            return 10
        'Create code to execute after loading to a module.\n\n        Notes:\n            Called by @onModuleDiscovered.\n\n        Args:\n            module: the module object\n\n        Returns:\n            None (does not apply, default)\n            tuple (code, documentary string)\n            tuple (code, documentary string, flags)\n        '
        return None

    @staticmethod
    def createFakeModuleDependency(module):
        if False:
            print('Hello World!')
        'Create module to depend on.\n\n        Notes:\n            Called by @onModuleDiscovered.\n\n        Args:\n            module: the module object\n\n        Returns:\n            None (does not apply, default)\n            tuple (code, reason)\n            tuple (code, reason, flags)\n        '
        return None

    @staticmethod
    def hasPreModuleLoadCode(module_name):
        if False:
            print('Hello World!')
        return getModuleInclusionInfoByName(makeTriggerModuleName(module_name, pre_module_load_trigger_name)) is not None

    @staticmethod
    def hasPostModuleLoadCode(module_name):
        if False:
            for i in range(10):
                print('nop')
        return getModuleInclusionInfoByName(makeTriggerModuleName(module_name, post_module_load_trigger_name)) is not None

    def onModuleDiscovered(self, module):
        if False:
            print('Hello World!')
        'Called with a module to be loaded.\n\n        Notes:\n            We may specify code to be prepended and/or appended to this module.\n            This code is stored in the appropriate dict.\n            For every imported module and each of these two options, only one plugin may do this.\n            We check this condition here.\n\n        Args:\n            module: the module object\n        Returns:\n            None\n        '
        return None

    def getPackageExtraScanPaths(self, package_name, package_dir):
        if False:
            for i in range(10):
                print('nop')
        'Provide other directories to consider submodules to live in.\n\n        Args:\n            module_name: full module name\n            package_dir: directory of the package\n\n        Returns:\n            Iterable list of directories, non-existent ones are ignored.\n        '
        return ()

    def onModuleEncounter(self, using_module_name, module_name, module_filename, module_kind):
        if False:
            return 10
        'Help decide whether to include a module.\n\n        Args:\n            using_module_name: module that does this (can be None if user)\n            module_name: full module name\n            module_filename: filename\n            module_kind: one of "py", "extension" (shared library)\n        Returns:\n            True or False\n        '
        return None

    def onModuleUsageLookAhead(self, module_name, module_filename, module_kind, get_module_source):
        if False:
            print('Hello World!')
        'React to tentative recursion of a module coming up.\n\n        For definite usage, use onModuleRecursion where it\'s a fact and\n        happening next. This may be a usage that is later optimized away\n        and doesn\'t impact anything. The main usage is to setup e.g.\n        hard imports as a factory, e.g. with detectable lazy loaders.\n\n        Args:\n            module_name: full module name\n            module_filename: filename\n            module_kind: one of "py", "extension" (shared library)\n            get_module_source: callable to get module source code if any\n        Returns:\n            None\n        '

    def onModuleRecursion(self, module_name, module_filename, module_kind, using_module_name, source_ref, reason):
        if False:
            return 10
        'React to recursion of a module coming up.\n\n        Args:\n            module_name: full module name\n            module_filename: filename\n            module_kind: one of "py", "extension" (shared library)\n            using_module_name: name of module that does the usage (None if it is a user choice)\n            source_ref: code making the import (None if it is a user choice)\n        Returns:\n            None\n        '

    def onModuleInitialSet(self):
        if False:
            while True:
                i = 10
        'Provide extra modules to the initial root module set.\n\n        Args:\n            None\n        Returns:\n            Iterable of modules, may yield.\n        '
        return ()

    def onModuleCompleteSet(self, module_set):
        if False:
            while True:
                i = 10
        'Provide extra modules to the initial root module set.\n\n        Args:\n            module_set - tuple of module objects\n        Returns:\n            None\n        Notes:\n            You must not change anything, this is purely for warning\n            and error checking, and potentially for later stages to\n            prepare.\n        '

    def onModuleCompleteSetGUI(self, module_set, plugin_binding_name):
        if False:
            for i in range(10):
                print('nop')
        from .Plugins import getOtherGUIBindingNames, getQtBindingNames
        for module in module_set:
            module_name = module.getFullName()
            if module_name == plugin_binding_name:
                continue
            if module_name in getOtherGUIBindingNames():
                if plugin_binding_name in getQtBindingNames():
                    recommendation = "Use '--nofollow-import-to=%s'" % module_name
                    if module_name in getQtBindingNames():
                        problem = 'conflicts with'
                    else:
                        problem = 'is redundant with'
                else:
                    recommendation = "Use '--enable-plugin=no-qt'"
                    problem = 'is redundant with'
                self.warning("Unwanted import of '%(unwanted)s' that %(problem)s '%(binding_name)s' encountered. %(recommendation)s or uninstall it for best compatibility with pure Python execution." % {'unwanted': module_name, 'binding_name': plugin_binding_name, 'recommendation': recommendation, 'problem': problem})

    @staticmethod
    def locateModule(module_name):
        if False:
            while True:
                i = 10
        'Provide a filename / -path for a to-be-imported module.\n\n        Args:\n            module_name: (str or ModuleName) full name of module\n        Returns:\n            filename for module\n        '
        from nuitka.importing.Importing import locateModule
        (_module_name, module_filename, _module_kind, _finding) = locateModule(module_name=ModuleName(module_name), parent_package=None, level=0)
        return module_filename

    @staticmethod
    def locateModules(module_name):
        if False:
            print('Hello World!')
        'Provide a filename / -path for a to-be-imported module.\n\n        Args:\n            module_name: (str or ModuleName) full name of module\n        Returns:\n            list of ModuleName\n        '
        from nuitka.importing.Importing import locateModules
        return locateModules(module_name)

    @classmethod
    def locateDLL(cls, dll_name):
        if False:
            while True:
                i = 10
        'Locate a DLL by name.'
        return locateDLL(dll_name)

    @classmethod
    def locateDLLsInDirectory(cls, directory):
        if False:
            while True:
                i = 10
        'Locate all DLLs in a folder\n\n        Returns:\n            list of (filename, filename_relative, dll_extension)\n        '
        return locateDLLsInDirectory(directory)

    def makeDllEntryPoint(self, source_path, dest_path, module_name, package_name, reason):
        if False:
            print('Hello World!')
        'Create an entry point, as expected to be provided by getExtraDlls.'
        return makeDllEntryPoint(logger=self, source_path=source_path, dest_path=dest_path, module_name=module_name, package_name=package_name, reason=reason)

    def makeExeEntryPoint(self, source_path, dest_path, module_name, package_name, reason):
        if False:
            while True:
                i = 10
        'Create an entry point, as expected to be provided by getExtraDlls.'
        return makeExeEntryPoint(logger=self, source_path=source_path, dest_path=dest_path, module_name=module_name, package_name=package_name, reason=reason)

    def reportFileCount(self, module_name, count, section=None):
        if False:
            while True:
                i = 10
        if count:
            msg = 'Found %d %s DLLs from %s%s installation.' % (count, 'file' if count < 2 else 'files', '' if not section else "'%s' " % section, module_name.asString())
            self.info(msg)

    def getExtraDlls(self, module):
        if False:
            while True:
                i = 10
        'Provide IncludedEntryPoint named tuples describing extra needs of the module.\n\n        Args:\n            module: the module object needing the binaries\n        Returns:\n            yields IncludedEntryPoint objects\n\n        '
        return ()

    def onCopiedDLL(self, dll_filename):
        if False:
            i = 10
            return i + 15
        'Chance for a plugin to modify DLLs after copy, e.g. to compress it, remove attributes, etc.\n\n        Args:\n            dll_filename: the filename of the DLL\n\n        Notes:\n            Do not remove or add any files in this method, this will not work well, there\n            is e.g. getExtraDLLs API to add things. This is only for post processing as\n            described above.\n\n        '
        return None

    def getModuleSpecificDllPaths(self, module_name):
        if False:
            while True:
                i = 10
        'Provide a list of directories, where DLLs should be searched for this package (or module).\n\n        Args:\n            module_name: name of a package or module, for which the DLL path addition applies.\n        Returns:\n            iterable of paths\n        '
        return ()

    def getModuleSysPathAdditions(self, module_name):
        if False:
            for i in range(10):
                print('nop')
        "Provide a list of directories, that should be considered in 'PYTHONPATH' when this module is used.\n\n        Args:\n            module_name: name of a package or module\n        Returns:\n            iterable of paths\n        "
        return ()

    def removeDllDependencies(self, dll_filename, dll_filenames):
        if False:
            return 10
        'Yield any DLLs / shared libraries not to be included in distribution.\n\n        Args:\n            dll_filename: DLL name\n            dll_filenames: list of DLLs\n        Yields:\n            yielded filenames to exclude\n        '
        return ()

    def considerDataFiles(self, module):
        if False:
            while True:
                i = 10
        'Yield data file names (source|func, target) for inclusion (iterator).\n\n        Args:\n            module: module object that may need extra data files\n        Yields:\n            Data file description pairs, either (source, dest) or (func, dest)\n            where the func will be called to create the content dynamically.\n\n        '
        return ()

    def isAcceptableMissingDLL(self, package_name, dll_basename):
        if False:
            while True:
                i = 10
        'Check if a missing DLL is acceptable to the plugin.\n\n        Args:\n            package_name: name of the package using the DLL\n            dll_basename : basename of the DLL, i.e. no suffix\n        Returns:\n            None (no opinion for that file), True (yes) or False (no)\n        '
        return None

    def makeIncludedDataFile(self, source_path, dest_path, reason, tags=''):
        if False:
            while True:
                i = 10
        return makeIncludedDataFile(source_path=source_path, dest_path=dest_path, reason=reason, tracer=self, tags=tags)

    def makeIncludedAppBundleResourceFile(self, source_path, dest_path, reason, tags=''):
        if False:
            return 10
        tags = decodeDataFileTags(tags)
        tags.add('framework_resource')
        assert isMacOS() and shallCreateAppBundle()
        dest_path = os.path.join('..', 'Resources', dest_path)
        return self.makeIncludedDataFile(source_path=source_path, dest_path=dest_path, reason=reason, tags=tags)

    def makeIncludedGeneratedDataFile(self, data, dest_path, reason, tags=''):
        if False:
            return 10
        return makeIncludedGeneratedDataFile(data=data, dest_path=dest_path, reason=reason, tracer=self, tags=tags)

    def makeIncludedDataDirectory(self, source_path, dest_path, reason, tags='', ignore_dirs=(), ignore_filenames=(), ignore_suffixes=(), only_suffixes=(), normalize=True):
        if False:
            print('Hello World!')
        return makeIncludedDataDirectory(source_path=source_path, dest_path=dest_path, reason=reason, tracer=self, tags=tags, ignore_dirs=ignore_dirs, ignore_filenames=ignore_filenames, ignore_suffixes=ignore_suffixes, only_suffixes=only_suffixes, normalize=normalize)

    def makeIncludedEmptyDirectory(self, dest_path, reason, tags):
        if False:
            for i in range(10):
                print('nop')
        return makeIncludedEmptyDirectory(dest_path=dest_path, reason=reason, tracer=self, tags=tags)

    def makeIncludedPackageDataFiles(self, package_name, package_directory, pattern, reason, tags):
        if False:
            i = 10
            return i + 15
        return makeIncludedPackageDataFiles(tracer=self, package_name=ModuleName(package_name), package_directory=package_directory, pattern=pattern, reason=reason, tags=tags)

    def updateDataFileTags(self, included_datafile):
        if False:
            for i in range(10):
                print('nop')
        'Add or remove data file tags.'

    def onDataFileTags(self, included_datafile):
        if False:
            while True:
                i = 10
        'Action on data file tags.'

    def onBeforeCodeParsing(self):
        if False:
            i = 10
            return i + 15
        'Prepare for code parsing, normally not needed.'

    def onStandaloneDistributionFinished(self, dist_dir):
        if False:
            return 10
        'Called after successfully creating a standalone distribution.\n\n        Note:\n            It is up to the plugin to take subsequent action. Examples are:\n            insert additional information (license, copyright, company or\n            application description), create installation material, further\n            folder clean-up, start downstream applications etc.\n\n        Args:\n            dist_dir: the created distribution folder\n\n        Returns:\n            None\n        '
        return None

    def onOnefileFinished(self, filename):
        if False:
            for i in range(10):
                print('nop')
        'Called after successfully creating a onefile executable.\n\n        Note:\n            It is up to the plugin to take subsequent action. Examples are:\n            insert additional information (license, copyright, company or\n            application description), create installation material, further\n            folder clean-up, start downstream applications etc.\n\n        Args:\n            filename: the created onefile executable\n\n        Returns:\n            None\n        '
        return None

    def onBootstrapBinary(self, filename):
        if False:
            print('Hello World!')
        'Called after successfully creating a bootstrap binary, but without payload.\n\n        Args:\n            filename: the created bootstrap binary, will be modified later\n\n        Returns:\n            None\n        '
        return None

    def onStandaloneBinary(self, filename):
        if False:
            print('Hello World!')
        'Called after successfully creating a standalone binary.\n\n        Args:\n            filename: the created standalone binary\n\n        Returns:\n            None\n        '
        return None

    def onFinalResult(self, filename):
        if False:
            while True:
                i = 10
        "Called after successfully finishing a compilation.\n\n        Note:\n            Plugins normally don't need this, and what filename is will be\n            heavily dependent on compilation modes. Actions can be take here,\n            e.g. commercial plugins output generated keys near that executable\n            path.\n        Args:\n            filename: the created binary (module, accelerated exe, dist exe, onefile exe)\n\n        Returns:\n            None\n        "
        return None

    def suppressUnknownImportWarning(self, importing, module_name, source_ref):
        if False:
            i = 10
            return i + 15
        'Suppress import warnings for unknown modules.\n\n        Args:\n            importing: the module object\n            module_name: name of module\n            source_ref: ???\n        Returns:\n            True or False\n        '
        return False

    def decideCompilation(self, module_name):
        if False:
            for i in range(10):
                print('nop')
        'Decide whether to compile a module (or just use its bytecode).\n\n        Notes:\n            The first plugin not returning None makes the decision. Thereafter,\n            no other plugins will be checked. If all plugins return None, the\n            module will be compiled.\n\n        Args:\n            module_name: name of module\n\n        Returns:\n            "compiled" or "bytecode" or None (no opinion, use by default)\n        '
        return None

    def getPreprocessorSymbols(self):
        if False:
            for i in range(10):
                print('nop')
        'Decide which C defines to be used in compilation.\n\n        Notes:\n            The plugins can each contribute, but are hopefully using\n            a namespace for their defines.\n\n        Returns:\n            None for no defines, otherwise dictionary of key to be\n            defined, and non-None values if any, i.e. no "-Dkey" only\n        '
        return None

    def getBuildDefinitions(self):
        if False:
            i = 10
            return i + 15
        'Decide C source defines to be used in compilation.\n\n        Notes:\n            Make sure to use a namespace for your defines, and prefer\n            `getPreprocessorSymbols` if you can.\n\n        Returns:\n            dict or None for no values\n        '
        return None

    def getExtraIncludeDirectories(self):
        if False:
            for i in range(10):
                print('nop')
        'Decide which extra directories to use for C includes in compilation.\n\n        Returns:\n            List of directories or None by default\n        '
        return None

    @classmethod
    def getPluginDataFilesDir(cls):
        if False:
            for i in range(10):
                print('nop')
        'Helper function that returns path, where data files for the plugin are stored.'
        plugin_filename = sys.modules[cls.__module__].__file__
        return changeFilenameExtension(plugin_filename, '')

    def getPluginDataFileContents(self, filename):
        if False:
            return 10
        'Helper function that returns contents of a plugin data file.'
        return getFileContents(os.path.join(self.getPluginDataFilesDir(), filename))

    def getExtraCodeFiles(self):
        if False:
            i = 10
            return i + 15
        'Add extra code files to the compilation.\n\n        Notes:\n            This is generally a bad idea to use unless you absolutely\n            know what you are doing.\n\n        Returns:\n            None for no extra codes, otherwise dictionary of key to be\n            filename, and value to be source code.\n        '
        return None

    def getExtraLinkLibraries(self):
        if False:
            i = 10
            return i + 15
        'Decide which link library should be added.\n\n        Notes:\n            Names provided multiple times, e.g. by multiple plugins are\n            only added once.\n\n        Returns:\n            None for no extra link library, otherwise the name as a **str**\n            or an iterable of names of link libraries.\n        '
        return None

    def getExtraLinkDirectories(self):
        if False:
            print('Hello World!')
        'Decide which link directories should be added.\n\n        Notes:\n            Directories provided multiple times, e.g. by multiple plugins are\n            only added once.\n\n        Returns:\n            None for no extra link directory, otherwise the name as a **str**\n            or an iterable of names of link directories.\n        '
        return None

    def warnUnusedPlugin(self, message):
        if False:
            for i in range(10):
                print('nop')
        'An inactive plugin may issue a warning if it believes this may be wrong.\n\n        Returns:\n            None\n        '
        if self.plugin_name not in _warned_unused_plugins:
            _warned_unused_plugins.add(self.plugin_name)
            plugins_logger.warning("Use '--enable-plugin=%s' for: %s" % (self.plugin_name, message))

    def onDataComposerRun(self):
        if False:
            while True:
                i = 10
        'Internal use only.\n\n        Returns:\n            None\n        '
        return None

    def onDataComposerResult(self, blob_filename):
        if False:
            print('Hello World!')
        'Internal use only.\n\n        Returns:\n            None\n        '
        return None

    def encodeDataComposerName(self, data_name):
        if False:
            for i in range(10):
                print('nop')
        'Internal use only.\n\n        Returns:\n            None\n        '
        return None
    _runtime_information_cache = {}

    def queryRuntimeInformationMultiple(self, info_name, setup_codes, values):
        if False:
            while True:
                i = 10
        info_name = self.plugin_name.replace('-', '_') + '_' + info_name
        if info_name in self._runtime_information_cache:
            return self._runtime_information_cache[info_name]
        keys = []
        query_codes = []
        for (key, value_expression) in values:
            keys.append(key)
            query_codes.append('print(repr(%s))' % value_expression)
            query_codes.append('print("-" * 27)')
        if type(setup_codes) is str:
            setup_codes = setup_codes.splitlines()
        if not setup_codes:
            setup_codes = ['pass']
        cmd = '\\\nfrom __future__ import print_function\nfrom __future__ import absolute_import\n\ntry:\n%(setup_codes)s\nexcept ImportError:\n    import sys\n    sys.exit(38)\n%(query_codes)s\n' % {'setup_codes': '\n'.join(('   %s' % line for line in setup_codes)), 'query_codes': '\n'.join(query_codes)}
        if shallShowExecutedCommands():
            self.info('Executing query command:\n%s' % cmd)
        try:
            feedback = check_output([sys.executable, '-c', cmd])
        except NuitkaCalledProcessError as e:
            if e.returncode == 38:
                return None
            if Options.is_debug:
                self.info(cmd)
            raise
        if str is not bytes:
            feedback = feedback.decode('utf8')
        if shallShowExecutedCommands():
            self.info('Result of query command:\n%s' % feedback)
        feedback = [line.strip() for line in feedback.splitlines()]
        if feedback.count('-' * 27) != len(keys):
            self.sysexit('Error, mismatch in output retrieving %r information.' % info_name)
        feedback = [line for line in feedback if line != '-' * 27]
        NamedtupleResultClass = makeNamedtupleClass(info_name, keys)
        self._runtime_information_cache[info_name] = NamedtupleResultClass(*(ast.literal_eval(value) for value in feedback))
        return self._runtime_information_cache[info_name]

    def queryRuntimeInformationSingle(self, setup_codes, value, info_name=None):
        if False:
            while True:
                i = 10
        if info_name is None:
            info_name = 'temp_info_for_' + self.plugin_name.replace('-', '_')
        return self.queryRuntimeInformationMultiple(info_name=info_name, setup_codes=setup_codes, values=(('key', value),)).key

    def onFunctionBodyParsing(self, module_name, function_name, body):
        if False:
            while True:
                i = 10
        'Provide a different function body for the function of that module.\n\n        Should return a boolean, indicating if any actual change was done.\n        '
        return False

    def getCacheContributionValues(self, module_name):
        if False:
            i = 10
            return i + 15
        'Provide values that represent the include of a plugin on the compilation.\n\n        This must be used to invalidate cache results, e.g. when using the\n        onFunctionBodyParsing function, and other things, that do not directly\n        affect the source code. By default a plugin being enabled changes the\n        result unless it makes it clear that is not the case.\n        '
        return self.plugin_name

    def getExtraConstantDefaultPopulation(self):
        if False:
            print('Hello World!')
        'Provide extra global constant values to code generation.'
        return ()

    def decideAllowOutsideDependencies(self, module_name):
        if False:
            i = 10
            return i + 15
        'Decide if outside of Python dependencies are allowed.\n\n        Returns:\n            None (no opinion for that module), True (yes) or False (no)\n        '
        return None

    @staticmethod
    def getPackageVersion(module_name):
        if False:
            for i in range(10):
                print('nop')
        'Provide package version of a distribution.'
        distribution_name = _getDistributionNameFromPackageName(module_name)
        return _getPackageVersion(distribution_name)

    def getEvaluationConditionControlTags(self):
        if False:
            print('Hello World!')
        return {}

    def evaluateCondition(self, full_name, condition):
        if False:
            for i in range(10):
                print('nop')
        if condition == 'True':
            return True
        if condition == 'False':
            return False
        context = TagContext(logger=self, full_name=full_name)
        context.update(control_tags)
        context.update({'macos': isMacOS(), 'win32': isWin32Windows(), 'linux': isLinux(), 'anaconda': isAnacondaPython(), 'is_conda_package': isDistributionCondaPackage, 'debian_python': isDebianPackagePython(), 'standalone': isStandaloneMode(), 'module_mode': shallMakeModule(), 'deployment': isDeploymentMode(), 'version': _getPackageVersion, 'get_dist_name': _getDistributionNameFromPackageName, 'plugin': _isPluginActive, 'no_asserts': hasPythonFlagNoAsserts(), 'no_docstrings': hasPythonFlagNoDocStrings(), 'no_annotations': hasPythonFlagNoAnnotations(), 'has_builtin_module': isBuiltinModuleName})
        if isWin32Windows():
            context.update({'arch_x86': getArchitecture() == 'x86', 'arch_amd64': getArchitecture() == 'x86_64', 'arch_arm64': getArchitecture() == 'arm64'})
        versions = getTestExecutionPythonVersions()
        for version in versions:
            (big, major) = version.split('.')
            numeric_version = int(big) * 256 + int(major) * 16
            is_same_or_higher_version = python_version >= numeric_version
            context['python' + big + major + '_or_higher'] = is_same_or_higher_version
            context['before_python' + big + major] = not is_same_or_higher_version
        context['before_python3'] = python_version < 768
        context['python3_or_higher'] = python_version >= 768
        try:
            result = eval(condition, context)
        except Exception as e:
            if Options.is_debug:
                raise
            self.sysexit("Error, failed to evaluate condition '%s' in this context, exception was '%s'." % (condition, e))
        if type(result) is not bool:
            self.sysexit("Error, condition '%s' for module '%s' did not evaluate to boolean result." % (condition, full_name))
        addModuleInfluencingCondition(module_name=full_name, plugin_name=self.plugin_name, condition=condition, control_tags=context.used_tags, result=result)
        return result

    @classmethod
    def warning(cls, message, **kwargs):
        if False:
            i = 10
            return i + 15
        mnemonic = kwargs.pop('mnemonic', None)
        if kwargs:
            plugins_logger.sysexit('Illegal keyword arguments for self.warning')
        plugins_logger.warning(cls.plugin_name + ': ' + message, mnemonic=mnemonic)

    @classmethod
    def info(cls, message):
        if False:
            i = 10
            return i + 15
        plugins_logger.info(cls.plugin_name + ': ' + message)

    @classmethod
    def sysexit(cls, message, mnemonic=None, reporting=True):
        if False:
            while True:
                i = 10
        plugins_logger.sysexit(cls.plugin_name + ': ' + message, mnemonic=mnemonic, reporting=reporting)

def standalone_only(func):
    if False:
        return 10
    'For plugins that have functionality that should be done in standalone mode only.'

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if False:
            return 10
        if isStandaloneMode():
            return func(*args, **kwargs)
        elif inspect.isgeneratorfunction(func):
            return ()
        else:
            return None
    return wrapped

class TagContext(dict):

    def __init__(self, logger, full_name, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        dict.__init__(self, *args, **kwargs)
        self.logger = logger
        self.full_name = full_name
        self.used_tags = OrderedSet()

    def __getitem__(self, key):
        if False:
            return 10
        try:
            self.used_tags.add(key)
            return dict.__getitem__(self, key)
        except KeyError:
            if key.startswith('use_'):
                return False
            self.logger.sysexit("Identifier '%s' in 'when' configuration of module '%s' is unknown." % (key, self.full_name))