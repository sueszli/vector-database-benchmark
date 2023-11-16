""" This is the main actions of Nuitka.

This can do all the steps to translate one module to a target language using
the Python C/API, to compile it to either an executable or an extension
module, potentially with bytecode included and used library copied into
a distribution folder.

"""
import os
import sys
from nuitka.build.DataComposerInterface import runDataComposer
from nuitka.build.SconsUtils import getSconsReportValue, readSconsReport
from nuitka.code_generation.ConstantCodes import addDistributionMetadataValue, getDistributionMetadataValues
from nuitka.freezer.IncludedDataFiles import addIncludedDataFilesFromFileOptions, addIncludedDataFilesFromPackageOptions, addIncludedDataFilesFromPlugins, copyDataFiles
from nuitka.freezer.IncludedEntryPoints import addExtensionModuleEntryPoint, addIncludedEntryPoints, getStandaloneEntryPoints, setMainEntryPoint
from nuitka.importing import Importing, Recursion
from nuitka.Options import getPythonPgoInput, hasPythonFlagIsolated, hasPythonFlagNoAnnotations, hasPythonFlagNoAsserts, hasPythonFlagNoDocStrings, hasPythonFlagNoWarnings, hasPythonFlagUnbuffered
from nuitka.plugins.Plugins import Plugins
from nuitka.PostProcessing import executePostProcessing
from nuitka.Progress import closeProgressBar, reportProgressBar, setupProgressBar
from nuitka.PythonFlavors import isApplePython, isDebianPackagePython, isFedoraPackagePython, isNuitkaPython, isPyenvPython
from nuitka.PythonVersions import getModuleLinkerLibs, getPythonABI, getSupportedPythonVersions, python_version, python_version_str
from nuitka.Serialization import ConstantAccessor
from nuitka.Tracing import general, inclusion_logger
from nuitka.tree import SyntaxErrors
from nuitka.tree.ReformulationMultidist import createMultidistMainSourceCode
from nuitka.utils import InstanceCounters
from nuitka.utils.Distributions import getDistribution
from nuitka.utils.Execution import callProcess, withEnvironmentVarOverridden, wrapCommandForDebuggerForExec
from nuitka.utils.FileOperations import changeFilenameExtension, deleteFile, getExternalUsePath, openTextFile, removeDirectory, resetDirectory
from nuitka.utils.Importing import getSharedLibrarySuffix
from nuitka.utils.MemoryUsage import reportMemoryUsage, showMemoryTrace
from nuitka.utils.ModuleNames import ModuleName
from nuitka.utils.ReExecute import callExecProcess, reExecuteNuitka
from nuitka.utils.StaticLibraries import getSystemStaticLibPythonPath
from nuitka.utils.Utils import getArchitecture, isMacOS, isWin32Windows
from nuitka.Version import getCommercialVersion, getNuitkaVersion
from . import ModuleRegistry, Options, OutputDirectories
from .build.SconsInterface import asBoolStr, cleanSconsDirectory, runScons, setCommonSconsOptions
from .code_generation import CodeGeneration, LoaderCodes, Reports
from .finalizations import Finalization
from .freezer.Onefile import getCompressorPython, packDistFolderToOnefile
from .freezer.Standalone import checkFreezingModuleSet, copyDllsUsed, detectUsedDLLs
from .optimizations.Optimization import optimizeModules
from .pgo.PGO import readPGOInputFile
from .reports.Reports import writeCompilationReports
from .tree.Building import buildMainModuleTree
from .tree.SourceHandling import writeSourceCode
from .TreeXML import dumpTreeXMLToFile

def _setupFromMainFilenames():
    if False:
        i = 10
        return i + 15
    main_filenames = Options.getMainEntryPointFilenames()
    for filename in main_filenames:
        Importing.addMainScriptDirectory(main_dir=os.path.dirname(os.path.abspath(filename)))

def _createMainModule():
    if False:
        print('Hello World!')
    'Create a node tree.\n\n    Turn that source code into a node tree structure. If following into\n    imported modules is allowed, more trees will be available during\n    optimization, or even immediately through forcefully included\n    directory paths.\n\n    '
    Plugins.onBeforeCodeParsing()
    main_filenames = Options.getMainEntryPointFilenames()
    if len(main_filenames) > 1:
        assert not Options.shallMakeModule()
        main_module = buildMainModuleTree(filename=main_filenames[0], source_code=createMultidistMainSourceCode(main_filenames))
    else:
        main_module = buildMainModuleTree(filename=main_filenames[0], source_code=None)
    OutputDirectories.setMainModule(main_module)
    for distribution_name in Options.getShallIncludeDistributionMetadata():
        distribution = getDistribution(distribution_name)
        if distribution is None:
            general.sysexit("Error, could not find distribution '%s' for which metadata was asked to be included." % distribution_name)
        addDistributionMetadataValue(distribution_name, distribution)
    source_dir = OutputDirectories.getSourceDirectoryPath()
    if not Options.shallOnlyExecCCompilerCall():
        cleanSconsDirectory(source_dir)
    if Options.isStandaloneMode():
        standalone_dir = OutputDirectories.getStandaloneDirectoryPath(bundle=False)
        resetDirectory(path=standalone_dir, ignore_errors=True)
        if Options.shallCreateAppBundle():
            resetDirectory(path=changeFilenameExtension(standalone_dir, '.app'), ignore_errors=True)
    deleteFile(path=OutputDirectories.getResultFullpath(onefile=False), must_exist=False)
    if Options.isOnefileMode():
        deleteFile(path=OutputDirectories.getResultFullpath(onefile=True), must_exist=False)
        getCompressorPython()
    for plugin_filename in Options.getShallFollowExtra():
        Recursion.checkPluginPath(plugin_filename=plugin_filename, module_package=None)
    for pattern in Options.getShallFollowExtraFilePatterns():
        Recursion.checkPluginFilenamePattern(pattern=pattern)
    for package_name in Options.getMustIncludePackages():
        (package_name, package_directory, _module_kind, finding) = Importing.locateModule(module_name=ModuleName(package_name), parent_package=None, level=0)
        if finding != 'absolute':
            inclusion_logger.sysexit("Error, failed to locate package '%s' you asked to include." % package_name)
        Recursion.checkPluginPath(plugin_filename=package_directory, module_package=package_name.getPackageName())
    for module_name in Options.getMustIncludeModules():
        (module_name, module_filename, _module_kind, finding) = Importing.locateModule(module_name=ModuleName(module_name), parent_package=None, level=0)
        if finding != 'absolute':
            inclusion_logger.sysexit("Error, failed to locate module '%s' you asked to include." % module_name.asString())
        Recursion.checkPluginSinglePath(plugin_filename=module_filename, module_package=module_name.getPackageName(), package_only=True)
    Plugins.onModuleInitialSet()
    optimizeModules(main_module.getOutputFilename())
    if Options.isStandaloneMode():
        checkFreezingModuleSet()
    for (distribution_name, (package_name, _metadata, _entry_points)) in getDistributionMetadataValues():
        if not ModuleRegistry.hasDoneModule(package_name):
            inclusion_logger.sysexit("Error, including metadata for distribution '%s' without including related package '%s'." % (distribution_name, package_name))
    Plugins.onModuleCompleteSet()
    if Options.isExperimental('check_xml_persistence'):
        for module in ModuleRegistry.getRootModules():
            if module.isMainModule():
                return module
        assert False
    else:
        return main_module

def dumpTreeXML():
    if False:
        print('Hello World!')
    filename = Options.getXMLDumpOutputFilename()
    if filename is not None:
        with openTextFile(filename, 'w') as output_file:
            for module in ModuleRegistry.getDoneModules():
                dumpTreeXMLToFile(tree=module.asXml(), output_file=output_file)
        general.info("XML dump of node state written to file '%s'." % filename)

def pickSourceFilenames(source_dir, modules):
    if False:
        print('Hello World!')
    'Pick the names for the C files of each module.\n\n    Args:\n        source_dir - the directory to put the module sources will be put into\n        modules    - all the modules to build.\n\n    Returns:\n        Dictionary mapping modules to filenames in source_dir.\n\n    Notes:\n        These filenames can collide, due to e.g. mixed case usage, or there\n        being duplicate copies, e.g. a package named the same as the main\n        binary.\n\n        Conflicts are resolved by appending @<number> with a count in the\n        list of sorted modules. We try to be reproducible here, so we get\n        still good caching for external tools.\n    '
    collision_filenames = set()

    def _getModuleFilenames(module):
        if False:
            i = 10
            return i + 15
        base_filename = os.path.join(source_dir, 'module.' + module.getFullName())
        collision_filename = base_filename.lower()
        return (base_filename, collision_filename)
    seen_filenames = set()
    for module in modules:
        if module.isPythonExtensionModule():
            continue
        (_base_filename, collision_filename) = _getModuleFilenames(module)
        if collision_filename in seen_filenames:
            collision_filenames.add(collision_filename)
        seen_filenames.add(collision_filename)
    module_filenames = {}
    collision_counts = {}
    for module in sorted(modules, key=lambda x: x.getFullName()):
        if module.isPythonExtensionModule():
            continue
        (base_filename, collision_filename) = _getModuleFilenames(module)
        if collision_filename in collision_filenames:
            collision_counts[collision_filename] = collision_counts.get(collision_filename, 0) + 1
            base_filename += '@%d' % collision_counts[collision_filename]
        module_filenames[module] = base_filename + '.c'
    return module_filenames

def makeSourceDirectory():
    if False:
        return 10
    'Get the full list of modules imported, create code for all of them.'
    for any_case_module in Options.getShallFollowModules():
        if '*' in any_case_module or '{' in any_case_module:
            continue
        if not ModuleRegistry.hasDoneModule(any_case_module) and (not ModuleRegistry.hasRootModule(any_case_module)):
            general.warning("Did not follow import to unused '%s', consider include options." % any_case_module)
    for module in ModuleRegistry.getDoneModules():
        if module.isCompiledPythonModule():
            Finalization.prepareCodeGeneration(module)
    compiled_modules = []
    for module in ModuleRegistry.getDoneModules():
        if module.isCompiledPythonModule():
            compiled_modules.append(module)
            if Options.isShowInclusion():
                inclusion_logger.info("Included compiled module '%s'." % module.getFullName())
        elif module.isPythonExtensionModule():
            addExtensionModuleEntryPoint(module)
            if Options.isShowInclusion():
                inclusion_logger.info("Included extension module '%s'." % module.getFullName())
        elif module.isUncompiledPythonModule():
            if Options.isShowInclusion():
                inclusion_logger.info("Included uncompiled module '%s'." % module.getFullName())
        else:
            assert False, module
    source_dir = OutputDirectories.getSourceDirectoryPath()
    module_filenames = pickSourceFilenames(source_dir=source_dir, modules=compiled_modules)
    setupProgressBar(stage='C Source Generation', unit='module', total=len(compiled_modules))
    for module in compiled_modules:
        c_filename = module_filenames[module]
        reportProgressBar(item=module.getFullName())
        source_code = CodeGeneration.generateModuleCode(module=module, data_filename=os.path.basename(c_filename[:-2] + '.const'))
        writeSourceCode(filename=c_filename, source_code=source_code)
    closeProgressBar()
    (helper_decl_code, helper_impl_code, constants_header_code, constants_body_code) = CodeGeneration.generateHelpersCode()
    writeSourceCode(filename=os.path.join(source_dir, '__helpers.h'), source_code=helper_decl_code)
    writeSourceCode(filename=os.path.join(source_dir, '__helpers.c'), source_code=helper_impl_code)
    writeSourceCode(filename=os.path.join(source_dir, '__constants.h'), source_code=constants_header_code)
    writeSourceCode(filename=os.path.join(source_dir, '__constants.c'), source_code=constants_body_code)

def _runPgoBinary():
    if False:
        i = 10
        return i + 15
    pgo_executable = OutputDirectories.getPgoRunExecutable()
    if not os.path.isfile(pgo_executable):
        general.sysexit("Error, failed to produce PGO binary '%s'" % pgo_executable)
    return callProcess([getExternalUsePath(pgo_executable)] + Options.getPgoArgs(), shell=False)

def _wasMsvcMode():
    if False:
        print('Hello World!')
    if not isWin32Windows():
        return False
    return getSconsReportValue(source_dir=OutputDirectories.getSourceDirectoryPath(), key='msvc_mode') == 'True'

def _deleteMsvcPGOFiles(pgo_mode):
    if False:
        while True:
            i = 10
    assert _wasMsvcMode()
    msvc_pgc_filename = OutputDirectories.getResultBasePath(onefile=False) + '!1.pgc'
    deleteFile(msvc_pgc_filename, must_exist=False)
    if pgo_mode == 'use':
        msvc_pgd_filename = OutputDirectories.getResultBasePath(onefile=False) + '.pgd'
        deleteFile(msvc_pgd_filename, must_exist=False)
    return msvc_pgc_filename

def _runCPgoBinary():
    if False:
        print('Hello World!')
    general.info('Running created binary to produce C level PGO information:', style='blue')
    if _wasMsvcMode():
        msvc_pgc_filename = _deleteMsvcPGOFiles(pgo_mode='generate')
        with withEnvironmentVarOverridden('PATH', getSconsReportValue(source_dir=OutputDirectories.getSourceDirectoryPath(), key='PATH')):
            _exit_code = _runPgoBinary()
        pgo_data_collected = os.path.exists(msvc_pgc_filename)
    else:
        _exit_code = _runPgoBinary()
        gcc_constants_pgo_filename = os.path.join(OutputDirectories.getSourceDirectoryPath(), '__constants.gcda')
        pgo_data_collected = os.path.exists(gcc_constants_pgo_filename)
    if not pgo_data_collected:
        general.sysexit('Error, no PGO information produced, did the created binary run at all?')
    general.info('Successfully collected C level PGO information.', style='blue')

def _runPythonPgoBinary():
    if False:
        i = 10
        return i + 15
    pgo_filename = OutputDirectories.getPgoRunInputFilename()
    with withEnvironmentVarOverridden('NUITKA_PGO_OUTPUT', pgo_filename):
        _exit_code = _runPgoBinary()
    if not os.path.exists(pgo_filename):
        general.sysexit('Error, no Python PGO information produced, did the created binary run at all?')
    return pgo_filename

def runSconsBackend():
    if False:
        i = 10
        return i + 15
    options = {'result_name': OutputDirectories.getResultBasePath(onefile=False), 'source_dir': OutputDirectories.getSourceDirectoryPath(), 'nuitka_python': asBoolStr(isNuitkaPython()), 'debug_mode': asBoolStr(Options.is_debug), 'python_debug': asBoolStr(Options.shallUsePythonDebug()), 'module_mode': asBoolStr(Options.shallMakeModule()), 'full_compat': asBoolStr(Options.is_full_compat), 'experimental': ','.join(Options.getExperimentalIndications()), 'trace_mode': asBoolStr(Options.shallTraceExecution()), 'python_version': python_version_str, 'file_reference_mode': Options.getFileReferenceMode(), 'module_count': '%d' % len(ModuleRegistry.getDoneModules())}
    if Options.isLowMemory():
        options['low_memory'] = asBoolStr(True)
    if not Options.shallMakeModule():
        options['result_exe'] = OutputDirectories.getResultFullpath(onefile=False)
        main_module = ModuleRegistry.getRootTopModule()
        assert main_module.isMainModule()
        main_module_name = main_module.getFullName()
        if main_module_name != '__main__':
            options['main_module_name'] = main_module_name
    if Options.shallUseStaticLibPython():
        options['static_libpython'] = getSystemStaticLibPythonPath()
    if isDebianPackagePython():
        options['debian_python'] = asBoolStr(True)
    if isFedoraPackagePython():
        options['fedora_python'] = asBoolStr(True)
    if isApplePython():
        options['apple_python'] = asBoolStr(True)
    if isPyenvPython():
        options['pyenv_python'] = asBoolStr(True)
    if Options.isStandaloneMode():
        options['standalone_mode'] = asBoolStr(True)
    if Options.isOnefileMode():
        options['onefile_mode'] = asBoolStr(True)
        if Options.isOnefileTempDirMode():
            options['onefile_temp_mode'] = asBoolStr(True)
    if Options.getForcedStdoutPath():
        options['forced_stdout_path'] = Options.getForcedStdoutPath()
    if Options.getForcedStderrPath():
        options['forced_stderr_path'] = Options.getForcedStderrPath()
    if Options.shallTreatUninstalledPython():
        options['uninstalled_python'] = asBoolStr(True)
    if ModuleRegistry.getUncompiledTechnicalModules():
        options['frozen_modules'] = str(len(ModuleRegistry.getUncompiledTechnicalModules()))
    if Options.isProfile():
        options['profile_mode'] = asBoolStr(True)
    if hasPythonFlagNoWarnings():
        options['no_python_warnings'] = asBoolStr(True)
    if hasPythonFlagNoAsserts():
        options['python_sysflag_optimize'] = str(2 if hasPythonFlagNoDocStrings() else 1)
        options['python_flag_no_asserts'] = asBoolStr(True)
    if hasPythonFlagNoDocStrings():
        options['python_flag_no_docstrings'] = asBoolStr(True)
    if hasPythonFlagNoAnnotations():
        options['python_flag_no_annotations'] = asBoolStr(True)
    if python_version < 768 and sys.flags.py3k_warning:
        options['python_sysflag_py3k_warning'] = asBoolStr(True)
    if python_version < 768 and (sys.flags.division_warning or sys.flags.py3k_warning):
        options['python_sysflag_division_warning'] = asBoolStr(True)
    if sys.flags.bytes_warning:
        options['python_sysflag_bytes_warning'] = asBoolStr(True)
    if int(os.environ.get('NUITKA_NOSITE_FLAG', Options.hasPythonFlagNoSite())):
        options['python_sysflag_no_site'] = asBoolStr(True)
    if Options.hasPythonFlagTraceImports():
        options['python_sysflag_verbose'] = asBoolStr(True)
    if Options.hasPythonFlagNoRandomization():
        options['python_sysflag_no_randomization'] = asBoolStr(True)
    if python_version < 768 and sys.flags.unicode:
        options['python_sysflag_unicode'] = asBoolStr(True)
    if python_version >= 880 and sys.flags.utf8_mode:
        options['python_sysflag_utf8'] = asBoolStr(True)
    if hasPythonFlagUnbuffered():
        options['python_sysflag_unbuffered'] = asBoolStr(True)
    if hasPythonFlagIsolated():
        options['python_sysflag_isolated'] = asBoolStr(True)
    abiflags = getPythonABI()
    if abiflags:
        options['abiflags'] = abiflags
    if Options.shallMakeModule():
        options['module_suffix'] = getSharedLibrarySuffix(preferred=True)
    link_module_libs = getModuleLinkerLibs()
    if link_module_libs:
        options['link_module_libs'] = ','.join(link_module_libs)
    env_values = setCommonSconsOptions(options)
    env_values.update(Plugins.getBuildDefinitions())
    if Options.shallCreatePgoInput():
        options['pgo_mode'] = 'python'
        result = runScons(options=options, env_values=env_values, scons_filename='Backend.scons')
        if not result:
            return (result, options)
        executePostProcessing()
        _runPythonPgoBinary()
        return (True, options)
    if Options.isPgoMode():
        if Options.isPgoMode():
            options['pgo_mode'] = 'generate'
            result = runScons(options=options, env_values=env_values, scons_filename='Backend.scons')
            if not result:
                return (result, options)
            executePostProcessing()
            _runCPgoBinary()
            options['pgo_mode'] = 'use'
    result = (runScons(options=options, env_values=env_values, scons_filename='Backend.scons'), options)
    if options.get('pgo_mode') == 'use' and _wasMsvcMode():
        _deleteMsvcPGOFiles(pgo_mode='use')
    return result

def callExecPython(args, clean_path, add_path):
    if False:
        while True:
            i = 10
    old_python_path = os.environ.get('PYTHONPATH')
    if clean_path and old_python_path is not None:
        os.environ['PYTHONPATH'] = ''
    if add_path:
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] += ':' + Options.getOutputDir()
        else:
            os.environ['PYTHONPATH'] = Options.getOutputDir()
    args += Options.getPositionalArgs()[1:] + Options.getMainArgs()
    callExecProcess(args)

def executeMain(binary_filename, clean_path):
    if False:
        return 10
    if Options.shallRunInDebugger() and (not Options.shallCreateCmdFileForExecution()):
        args = wrapCommandForDebuggerForExec(binary_filename)
    else:
        args = (binary_filename, binary_filename)
    callExecPython(clean_path=clean_path, add_path=False, args=args)

def executeModule(tree, clean_path):
    if False:
        while True:
            i = 10
    'Execute the extension module just created.'
    if python_version < 832:
        python_command_template = "import os, imp;assert os.path.normcase(os.path.abspath(os.path.normpath(imp.find_module('%(module_name)s')[1]))) == %(expected_filename)r,'Error, cannot launch extension module %(module_name)s, original package is in the way.'"
    else:
        python_command_template = "import os, importlib.util;assert os.path.normcase(os.path.abspath(os.path.normpath(importlib.util.find_spec('%(module_name)s').origin))) == %(expected_filename)r,'Error, cannot launch extension module %(module_name)s, original package is in the way.'"
    python_command_template += ";__import__('%(module_name)s')"
    python_command = python_command_template % {'module_name': tree.getName(), 'expected_filename': os.path.normcase(os.path.abspath(os.path.normpath(OutputDirectories.getResultFullpath(onefile=False))))}
    if Options.shallRunInDebugger():
        args = wrapCommandForDebuggerForExec(sys.executable, '-c', python_command)
    else:
        args = (sys.executable, 'python', '-c', python_command)
    callExecPython(clean_path=clean_path, add_path=True, args=args)

def compileTree():
    if False:
        for i in range(10):
            print('nop')
    source_dir = OutputDirectories.getSourceDirectoryPath()
    general.info('Completed Python level compilation and optimization.')
    if not Options.shallOnlyExecCCompilerCall():
        general.info('Generating source code for C backend compiler.')
        reportMemoryUsage('before_c_code_generation', 'Total memory usage before generating C code:' if Options.isShowProgress() or Options.isShowMemory() else None)
        makeSourceDirectory()
        bytecode_accessor = ConstantAccessor(data_filename='__bytecode.const', top_level_name='bytecode_data')
        loader_code = LoaderCodes.getMetaPathLoaderBodyCode(bytecode_accessor)
        writeSourceCode(filename=os.path.join(source_dir, '__loader.c'), source_code=loader_code)
    else:
        source_dir = OutputDirectories.getSourceDirectoryPath()
        if not os.path.isfile(os.path.join(source_dir, '__helpers.h')):
            general.sysexit('Error, no previous build directory exists.')
    reportMemoryUsage('before_running_scons', 'Total memory usage before running scons' if Options.isShowProgress() or Options.isShowMemory() else None)
    if Options.isShowMemory():
        InstanceCounters.printStats()
    if Options.is_debug:
        Reports.doMissingOptimizationReport()
    if Options.shallNotDoExecCCompilerCall():
        return (True, {})
    general.info('Running data composer tool for optimal constant value handling.')
    runDataComposer(source_dir)
    Plugins.writeExtraCodeFiles(onefile=False)
    general.info('Running C compilation via Scons.')
    (result, options) = runSconsBackend()
    return (result, options)

def handleSyntaxError(e):
    if False:
        print('Hello World!')
    error_message = SyntaxErrors.formatOutput(e)
    if not Options.is_full_compat:
        suggested_python_version_str = getSupportedPythonVersions()[-1]
        error_message += "\n\nNuitka is very syntax compatible with standard Python. It is currently running\nwith Python version '%s', you might want to specify more clearly with the use\nof the precise Python interpreter binary and '-m nuitka', e.g. use this\n'python%s -m nuitka' option, if that's not the one the program expects.\n" % (python_version_str, suggested_python_version_str)
    sys.exit(error_message)

def _main():
    if False:
        i = 10
        return i + 15
    'Main program flow of Nuitka\n\n    At this point, options will be parsed already, Nuitka will be executing\n    in the desired version of Python with desired flags, and we just get\n    to execute the task assigned.\n\n    We might be asked to only re-compile generated C, dump only an XML\n    representation of the internal node tree after optimization, etc.\n    '
    pgo_filename = getPythonPgoInput()
    if pgo_filename is not None:
        readPGOInputFile(pgo_filename)
    general.info('Starting Python compilation with Nuitka %r on Python %r commercial grade %r.' % (getNuitkaVersion(), python_version_str, getCommercialVersion() or 'not installed'))
    reportMemoryUsage('after_launch', 'Total memory usage before processing:' if Options.isShowProgress() or Options.isShowMemory() else None)
    _setupFromMainFilenames()
    addIncludedDataFilesFromFileOptions()
    addIncludedDataFilesFromPackageOptions()
    try:
        main_module = _createMainModule()
    except (SyntaxError, IndentationError) as e:
        handleSyntaxError(e)
    addIncludedDataFilesFromPlugins()
    dumpTreeXML()
    (result, options) = compileTree()
    if not result:
        general.sysexit(exit_code=1, reporting=True)
    if Options.shallCreatePgoInput():
        pgo_filename = OutputDirectories.getPgoRunInputFilename()
        general.info("Restarting compilation using collected information from '%s'." % pgo_filename)
        reExecuteNuitka(pgo_filename=pgo_filename)
    if Options.shallNotDoExecCCompilerCall():
        if Options.isShowMemory():
            showMemoryTrace()
        sys.exit(0)
    executePostProcessing()
    copyDataFiles()
    if Options.isStandaloneMode():
        binary_filename = options['result_exe']
        setMainEntryPoint(binary_filename)
        for module in ModuleRegistry.getDoneModules():
            addIncludedEntryPoints(Plugins.considerExtraDlls(module))
        detectUsedDLLs(standalone_entry_points=getStandaloneEntryPoints(), source_dir=OutputDirectories.getSourceDirectoryPath())
        dist_dir = OutputDirectories.getStandaloneDirectoryPath()
        copyDllsUsed(dist_dir=dist_dir, standalone_entry_points=getStandaloneEntryPoints())
        Plugins.onStandaloneDistributionFinished(dist_dir)
        if Options.isOnefileMode():
            packDistFolderToOnefile(dist_dir)
            if Options.isRemoveBuildDir():
                general.info("Removing dist folder '%s'." % dist_dir)
                removeDirectory(path=dist_dir, ignore_errors=False)
            else:
                general.info("Keeping dist folder '%s' for inspection, no need to use it." % dist_dir)
    source_dir = OutputDirectories.getSourceDirectoryPath()
    if Options.isRemoveBuildDir():
        general.info("Removing build directory '%s'." % source_dir)
        readSconsReport(source_dir)
        removeDirectory(path=source_dir, ignore_errors=False)
        assert not os.path.exists(source_dir)
    else:
        general.info("Keeping build directory '%s'." % source_dir)
    final_filename = OutputDirectories.getResultFullpath(onefile=Options.isOnefileMode())
    if Options.isStandaloneMode() and isMacOS():
        general.info('Created binary that runs on macOS %s (%s) or higher.' % (options['macos_min_version'], options['macos_target_arch']))
        if options['macos_target_arch'] != getArchitecture():
            general.warning("It will only work as well as 'arch -%s %s %s' does." % (options['macos_target_arch'], sys.executable, Options.getMainEntryPointFilenames()[0]), mnemonic='macos-cross-compile')
    Plugins.onFinalResult(final_filename)
    if Options.shallMakeModule():
        base_path = OutputDirectories.getResultBasePath(onefile=False)
        if os.path.isdir(base_path) and os.path.isfile(os.path.join(base_path, '__init__.py')):
            general.warning("The compilation result is hidden by package directory '%s'. Importing will not use compiled code while it exists." % base_path)
    general.info("Successfully created '%s'." % final_filename)
    writeCompilationReports(aborted=False)
    run_filename = OutputDirectories.getResultRunFilename(onefile=Options.isOnefileMode())
    if Options.shallExecuteImmediately():
        general.info("Launching '%s'" % run_filename)
        if Options.shallMakeModule():
            executeModule(tree=main_module, clean_path=Options.shallClearPythonPathEnvironment())
        else:
            executeMain(binary_filename=run_filename, clean_path=Options.shallClearPythonPathEnvironment())
    elif run_filename != final_filename:
        general.info("Execute it by launching '%s', the batch file needs to set environment." % run_filename)

def main():
    if False:
        for i in range(10):
            print('nop')
    try:
        _main()
    except BaseException:
        try:
            writeCompilationReports(aborted=True)
        except BaseException as e:
            general.warning('Report writing was prevented by exception %s' % e)
        raise