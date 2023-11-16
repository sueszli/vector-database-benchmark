""" Control the flow of optimizations applied to node tree.

Applies abstract execution on all so far known modules until no more
optimization is possible. Every successful optimization to anything might
make others possible.
"""
import inspect
from nuitka import ModuleRegistry, Options, Variables
from nuitka.importing.Importing import addExtraSysPaths
from nuitka.importing.Recursion import considerUsedModules
from nuitka.plugins.Plugins import Plugins
from nuitka.Progress import closeProgressBar, reportProgressBar, setupProgressBar
from nuitka.Tracing import general, optimization_logger, progress_logger
from nuitka.utils.MemoryUsage import MemoryWatch, reportMemoryUsage
from nuitka.utils.Timing import TimerReport
from . import Graphs
from .BytecodeDemotion import demoteCompiledModuleToBytecode
from .Tags import TagSet
from .TraceCollections import withChangeIndicationsTo
tag_set = None

def signalChange(tags, source_ref, message):
    if False:
        print('Hello World!')
    'Indicate a change to the optimization framework.'
    if message is not None:
        if Options.is_verbose:
            optimization_logger.info('{source_ref} : {tags} : {message}'.format(source_ref=source_ref.getAsString(), tags=tags, message=message() if inspect.isfunction(message) else message))
    tag_set.onSignal(tags)

def optimizeCompiledPythonModule(module):
    if False:
        while True:
            i = 10
    optimization_logger.info_if_file("Doing module local optimizations for '{module_name}'.".format(module_name=module.getFullName()), other_logger=progress_logger)
    touched = False
    if Options.isShowProgress() and Options.isShowMemory():
        memory_watch = MemoryWatch()
    unchanged_count = 0
    while True:
        tag_set.clear()
        try:
            with withChangeIndicationsTo(signalChange):
                scopes_were_incomplete = module.computeModule()
        except SystemExit:
            raise
        except BaseException:
            general.info("Interrupted while working on '%s'." % module)
            raise
        if scopes_were_incomplete:
            tag_set.add('var_usage')
        Graphs.onModuleOptimizationStep(module)
        if not tag_set:
            unchanged_count += 1
            if unchanged_count == 1 and pass_count == 1:
                optimization_logger.info_if_file('Not changed, but retrying one more time.', other_logger=progress_logger)
                continue
            optimization_logger.info_if_file('Finished with the module.', other_logger=progress_logger)
            break
        unchanged_count = 0
        optimization_logger.info_if_file('Not finished with the module due to following change kinds: %s' % ','.join(sorted(tag_set)), other_logger=progress_logger)
        touched = True
    if Options.isShowProgress() and Options.isShowMemory():
        memory_watch.finish("Memory usage changed during optimization of '%s'" % module.getFullName())
    considerUsedModules(module=module, pass_count=pass_count)
    return touched

def optimizeUncompiledPythonModule(module):
    if False:
        i = 10
        return i + 15
    full_name = module.getFullName()
    progress_logger.info("Doing module dependency considerations for '{module_name}':".format(module_name=full_name))
    module.attemptRecursion()
    considerUsedModules(module=module, pass_count=pass_count)
    Plugins.considerImplicitImports(module=module)

def optimizeExtensionModule(module):
    if False:
        print('Hello World!')
    module.attemptRecursion()
    Plugins.considerImplicitImports(module=module)

def optimizeModule(module):
    if False:
        print('Hello World!')
    global tag_set
    tag_set = TagSet()
    addExtraSysPaths(Plugins.getModuleSysPathAdditions(module.getFullName()))
    if module.isPythonExtensionModule():
        optimizeExtensionModule(module)
        changed = False
    elif module.isCompiledPythonModule():
        changed = optimizeCompiledPythonModule(module)
    else:
        optimizeUncompiledPythonModule(module)
        changed = False
    return changed
pass_count = 0
last_total = 0

def _restartProgress():
    if False:
        return 10
    global pass_count
    closeProgressBar()
    pass_count += 1
    optimization_logger.info_if_file('PASS %d:' % pass_count, other_logger=progress_logger)
    if not Options.is_verbose or optimization_logger.isFileOutput():
        setupProgressBar(stage='PASS %d' % pass_count, unit='module', total=ModuleRegistry.getRemainingModulesCount() + ModuleRegistry.getDoneModulesCount(), min_total=last_total)

def _traceProgressModuleStart(current_module):
    if False:
        while True:
            i = 10
    optimization_logger.info_if_file("Optimizing module '{module_name}', {remaining:d} more modules to go after that.".format(module_name=current_module.getFullName(), remaining=ModuleRegistry.getRemainingModulesCount()), other_logger=progress_logger)
    reportProgressBar(item=current_module.getFullName(), total=ModuleRegistry.getRemainingModulesCount() + ModuleRegistry.getDoneModulesCount(), update=False)
    if Options.isShowProgress() and Options.isShowMemory():
        reportMemoryUsage('optimization/%d/%s' % (pass_count, current_module.getFullName()), "Total memory usage before optimizing module '%s'" % current_module.getFullName() if Options.isShowProgress() or Options.isShowMemory() else None)

def _traceProgressModuleEnd(current_module):
    if False:
        while True:
            i = 10
    reportProgressBar(item=current_module.getFullName(), total=ModuleRegistry.getRemainingModulesCount() + ModuleRegistry.getDoneModulesCount(), update=True)

def _endProgress():
    if False:
        for i in range(10):
            print('nop')
    global last_total
    last_total = closeProgressBar()

def restoreFromXML(text):
    if False:
        i = 10
        return i + 15
    from nuitka.nodes.NodeBases import fromXML
    from nuitka.TreeXML import fromString
    xml = fromString(text)
    module = fromXML(provider=None, xml=xml)
    return module

def makeOptimizationPass():
    if False:
        print('Hello World!')
    'Make a single pass for optimization, indication potential completion.'
    finished = True
    ModuleRegistry.startTraversal()
    _restartProgress()
    main_module = None
    stdlib_phase_done = False
    while True:
        current_module = ModuleRegistry.nextModule()
        if current_module is None:
            if main_module is not None and pass_count == 1:
                considerUsedModules(module=main_module, pass_count=-1)
                stdlib_phase_done = True
                main_module = None
                continue
            break
        if current_module.isMainModule() and (not stdlib_phase_done):
            main_module = current_module
        _traceProgressModuleStart(current_module)
        module_name = current_module.getFullName()
        with TimerReport(message='Optimizing %s' % module_name, decider=False) as module_timer:
            changed = optimizeModule(current_module)
        ModuleRegistry.addModuleOptimizationTimeInformation(module_name=module_name, pass_number=pass_count, time_used=module_timer.getDelta())
        _traceProgressModuleEnd(current_module)
        if changed:
            finished = False
    for current_module in ModuleRegistry.getDoneModules():
        if current_module.isCompiledPythonModule():
            for unused_function in current_module.getUnusedFunctions():
                Variables.updateVariablesFromCollection(old_collection=unused_function.trace_collection, new_collection=None, source_ref=unused_function.getSourceReference())
                unused_function.trace_collection = None
                unused_function.finalize()
            current_module.subnode_functions = tuple((function for function in current_module.subnode_functions if function in current_module.getUsedFunctions()))
    _endProgress()
    return finished

def optimizeModules(output_filename):
    if False:
        for i in range(10):
            print('nop')
    Graphs.startGraph()
    finished = makeOptimizationPass()
    for module in ModuleRegistry.getDoneModules():
        if module.isCompiledPythonModule() and module.getCompilationMode() == 'bytecode':
            demoteCompiledModuleToBytecode(module)
    while not finished:
        finished = makeOptimizationPass()
    Graphs.endGraph(output_filename)