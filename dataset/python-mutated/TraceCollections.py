""" Trace collection (also often still referred to as constraint collection).

At the core of value propagation there is the collection of constraints that
allow to propagate knowledge forward or not.

This is about collecting these constraints and to manage them.
"""
import contextlib
from collections import defaultdict
from contextlib import contextmanager
from nuitka import Variables
from nuitka.__past__ import iterItems
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.importing.Importing import locateModule, makeModuleUsageAttempt
from nuitka.importing.Recursion import decideRecursion
from nuitka.ModuleRegistry import addUsedModule
from nuitka.nodes.NodeMakingHelpers import getComputationResult
from nuitka.nodes.shapes.StandardShapes import tshape_uninitialized
from nuitka.Tracing import inclusion_logger, printError, printLine, printSeparator
from nuitka.tree.SourceHandling import readSourceLine
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances
from nuitka.utils.Timing import TimerReport
from .ValueTraces import ValueTraceAssign, ValueTraceAssignUnescapable, ValueTraceAssignUnescapablePropagated, ValueTraceAssignVeryTrusted, ValueTraceDeleted, ValueTraceEscaped, ValueTraceInit, ValueTraceInitStarArgs, ValueTraceInitStarDict, ValueTraceLoopComplete, ValueTraceLoopIncomplete, ValueTraceMerge, ValueTraceUninitialized, ValueTraceUnknown
signalChange = None

@contextmanager
def withChangeIndicationsTo(signal_change):
    if False:
        print('Hello World!')
    'Decide where change indications should go to.'
    global signalChange
    old = signalChange
    signalChange = signal_change
    yield
    signalChange = old

class CollectionUpdateMixin(object):
    """Mixin to use in every collection to add traces."""
    __slots__ = ()

    def hasVariableTrace(self, variable, version):
        if False:
            i = 10
            return i + 15
        return (variable, version) in self.variable_traces

    def getVariableTrace(self, variable, version):
        if False:
            return 10
        return self.variable_traces[variable, version]

    def getVariableTraces(self, variable):
        if False:
            for i in range(10):
                print('nop')
        result = []
        for (key, variable_trace) in iterItems(self.variable_traces):
            candidate = key[0]
            if variable is candidate:
                result.append(variable_trace)
        return result

    def getVariableTracesAll(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_traces

    def addVariableTrace(self, variable, version, trace):
        if False:
            return 10
        key = (variable, version)
        assert key not in self.variable_traces, (key, self)
        self.variable_traces[key] = trace

    def addVariableMergeMultipleTrace(self, variable, traces):
        if False:
            i = 10
            return i + 15
        version = variable.allocateTargetNumber()
        trace_merge = ValueTraceMerge(traces)
        self.addVariableTrace(variable, version, trace_merge)
        return version

class CollectionStartPointMixin(CollectionUpdateMixin):
    """Mixin to use in start points of collections.

    These are modules, functions, etc. typically entry points.
    """
    __slots__ = ()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.variable_versions = {}
        self.variable_traces = {}
        self.break_collections = None
        self.continue_collections = None
        self.return_collections = None
        self.exception_collections = None
        self.outline_functions = None

    def getLoopBreakCollections(self):
        if False:
            return 10
        return self.break_collections

    def onLoopBreak(self, collection=None):
        if False:
            print('Hello World!')
        if collection is None:
            collection = self
        self.break_collections.append(TraceCollectionBranch(parent=collection, name='loop break'))

    def getLoopContinueCollections(self):
        if False:
            for i in range(10):
                print('nop')
        return self.continue_collections

    def onLoopContinue(self, collection=None):
        if False:
            for i in range(10):
                print('nop')
        if collection is None:
            collection = self
        self.continue_collections.append(TraceCollectionBranch(parent=collection, name='loop continue'))

    def onFunctionReturn(self, collection=None):
        if False:
            for i in range(10):
                print('nop')
        if collection is None:
            collection = self
        if self.return_collections is not None:
            self.return_collections.append(TraceCollectionBranch(parent=collection, name='return'))

    def onExceptionRaiseExit(self, raisable_exceptions, collection=None):
        if False:
            i = 10
            return i + 15
        'Indicate to the trace collection what exceptions may have occurred.\n\n        Args:\n            raisable_exception: Currently ignored, one or more exceptions that\n            could occur, e.g. "BaseException".\n            collection: To pass the collection that will be the parent\n        Notes:\n            Currently this is unused. Passing "collection" as an argument, so\n            we know the original collection to attach the branch to, is maybe\n            not the most clever way to do this\n\n            We also might want to specialize functions for specific exceptions,\n            there is little point in providing BaseException as an argument in\n            so many places.\n\n            The actual storage of the exceptions that can occur is currently\n            missing entirely. We just use this to detect "any exception" by\n            not being empty.\n        '
        if collection is None:
            collection = self
        if self.exception_collections is not None:
            self.exception_collections.append(TraceCollectionBranch(parent=collection, name='exception'))

    def getFunctionReturnCollections(self):
        if False:
            i = 10
            return i + 15
        return self.return_collections

    def getExceptionRaiseCollections(self):
        if False:
            while True:
                i = 10
        return self.exception_collections

    def hasEmptyTraces(self, variable):
        if False:
            print('Hello World!')
        traces = self.getVariableTraces(variable)
        return areEmptyTraces(traces)

    def hasReadOnlyTraces(self, variable):
        if False:
            print('Hello World!')
        traces = self.getVariableTraces(variable)
        return areReadOnlyTraces(traces)

    def initVariableUnknown(self, variable):
        if False:
            i = 10
            return i + 15
        trace = ValueTraceUnknown(owner=self.owner, previous=None)
        self.addVariableTrace(variable, 0, trace)
        return trace

    def initVariableModule(self, variable):
        if False:
            for i in range(10):
                print('nop')
        trace = ValueTraceUnknown(owner=self.owner, previous=None)
        self.addVariableTrace(variable, 0, trace)
        return trace

    def initVariableInit(self, variable):
        if False:
            print('Hello World!')
        trace = ValueTraceInit(self.owner)
        self.addVariableTrace(variable, 0, trace)
        return trace

    def initVariableInitStarArgs(self, variable):
        if False:
            i = 10
            return i + 15
        trace = ValueTraceInitStarArgs(self.owner)
        self.addVariableTrace(variable, 0, trace)
        return trace

    def initVariableInitStarDict(self, variable):
        if False:
            return 10
        trace = ValueTraceInitStarDict(self.owner)
        self.addVariableTrace(variable, 0, trace)
        return trace

    def initVariableUninitialized(self, variable):
        if False:
            return 10
        trace = ValueTraceUninitialized(owner=self.owner, previous=None)
        self.addVariableTrace(variable, 0, trace)
        return trace

    def updateVariablesFromCollection(self, old_collection, source_ref):
        if False:
            print('Hello World!')
        Variables.updateVariablesFromCollection(old_collection, self, source_ref)

    @contextlib.contextmanager
    def makeAbortStackContext(self, catch_breaks, catch_continues, catch_returns, catch_exceptions):
        if False:
            while True:
                i = 10
        if catch_breaks:
            old_break_collections = self.break_collections
            self.break_collections = []
        if catch_continues:
            old_continue_collections = self.continue_collections
            self.continue_collections = []
        if catch_returns:
            old_return_collections = self.return_collections
            self.return_collections = []
        if catch_exceptions:
            old_exception_collections = self.exception_collections
            self.exception_collections = []
        yield
        if catch_breaks:
            self.break_collections = old_break_collections
        if catch_continues:
            self.continue_collections = old_continue_collections
        if catch_returns:
            self.return_collections = old_return_collections
        if catch_exceptions:
            self.exception_collections = old_exception_collections

    def initVariable(self, variable):
        if False:
            for i in range(10):
                print('nop')
        return variable.initVariable(self)

    def addOutlineFunction(self, outline):
        if False:
            while True:
                i = 10
        if self.outline_functions is None:
            self.outline_functions = [outline]
        else:
            self.outline_functions.append(outline)

    def getOutlineFunctions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.outline_functions

    def onLocalsDictEscaped(self, locals_scope):
        if False:
            i = 10
            return i + 15
        locals_scope.preventLocalsDictPropagation()
        for variable in locals_scope.variables.values():
            self.markActiveVariableAsEscaped(variable)
        for variable in self.variable_actives:
            if variable.isTempVariable() or variable.isModuleVariable():
                continue
            self.markActiveVariableAsEscaped(variable)

    def onUsedFunction(self, function_body):
        if False:
            return 10
        owning_module = function_body.getParentModule()
        addUsedModule(module=owning_module, using_module=None, usage_tag='function', reason='Function %s' % self.name, source_ref=owning_module.source_ref)
        needs_visit = owning_module.addUsedFunction(function_body)
        if needs_visit or function_body.isExpressionFunctionPureBody():
            function_body.computeFunctionRaw(self)

class TraceCollectionBase(object):
    """This contains for logic for maintaining active traces.

    They are kept for "variable" and versions.
    """
    __slots__ = ('owner', 'parent', 'name', 'variable_actives')
    if isCountingInstances():
        __del__ = counted_del()

    @counted_init
    def __init__(self, owner, name, parent):
        if False:
            print('Hello World!')
        self.owner = owner
        self.parent = parent
        self.name = name
        self.variable_actives = {}

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s for %s at 0x%x>' % (self.__class__.__name__, self.name, id(self))

    def getOwner(self):
        if False:
            while True:
                i = 10
        return self.owner

    def dumpActiveTraces(self, indent=''):
        if False:
            for i in range(10):
                print('nop')
        printSeparator()
        printLine('Active are:')
        for (variable, version) in sorted(self.variable_actives.items(), key=lambda var: var[0].variable_name):
            printLine('%s %s:' % (variable, version))
            self.getVariableCurrentTrace(variable).dump(indent)
        printSeparator()

    def getVariableCurrentTrace(self, variable):
        if False:
            for i in range(10):
                print('nop')
        'Get the current value trace associated to this variable\n\n        It is also created on the fly if necessary. We create them\n        lazy so to keep the tracing branches minimal where possible.\n        '
        return self.getVariableTrace(variable=variable, version=self._getCurrentVariableVersion(variable))

    def markCurrentVariableTrace(self, variable, version):
        if False:
            return 10
        self.variable_actives[variable] = version

    def _getCurrentVariableVersion(self, variable):
        if False:
            print('Hello World!')
        try:
            return self.variable_actives[variable]
        except KeyError:
            if not self.hasVariableTrace(variable, 0):
                self.initVariable(variable)
            self.markCurrentVariableTrace(variable, 0)
            return self.variable_actives[variable]

    def markActiveVariableAsEscaped(self, variable):
        if False:
            while True:
                i = 10
        current = self.getVariableCurrentTrace(variable)
        if current.isTraceThatNeedsEscape():
            version = variable.allocateTargetNumber()
            self.addVariableTrace(variable, version, ValueTraceEscaped(owner=self.owner, previous=current))
            self.markCurrentVariableTrace(variable, version)

    def markClosureVariableAsUnknown(self, variable):
        if False:
            while True:
                i = 10
        current = self.getVariableCurrentTrace(variable)
        if not current.isUnknownTrace():
            version = variable.allocateTargetNumber()
            self.addVariableTrace(variable, version, ValueTraceUnknown(owner=self.owner, previous=current))
            self.markCurrentVariableTrace(variable, version)

    def markActiveVariableAsUnknown(self, variable):
        if False:
            return 10
        current = self.getVariableCurrentTrace(variable)
        if not current.isUnknownOrVeryTrustedTrace():
            version = variable.allocateTargetNumber()
            self.addVariableTrace(variable, version, ValueTraceUnknown(owner=self.owner, previous=current))
            self.markCurrentVariableTrace(variable, version)

    def markActiveVariableAsLoopMerge(self, loop_node, current, variable, shapes, incomplete):
        if False:
            i = 10
            return i + 15
        if incomplete:
            result = ValueTraceLoopIncomplete(loop_node, current, shapes)
        else:
            if not shapes:
                shapes.add(tshape_uninitialized)
            result = ValueTraceLoopComplete(loop_node, current, shapes)
        version = variable.allocateTargetNumber()
        self.addVariableTrace(variable, version, result)
        self.markCurrentVariableTrace(variable, version)
        return result

    @staticmethod
    def signalChange(tags, source_ref, message):
        if False:
            return 10
        signalChange(tags, source_ref, message)

    @staticmethod
    def mustAlias(a, b):
        if False:
            print('Hello World!')
        if a.isExpressionVariableRef() and b.isExpressionVariableRef():
            return a.getVariable() is b.getVariable()
        return False

    @staticmethod
    def mustNotAlias(a, b):
        if False:
            return 10
        if a.isExpressionConstantRef() and b.isExpressionConstantRef():
            if a.isMutable() or b.isMutable():
                return True
        return False

    def onControlFlowEscape(self, node):
        if False:
            for i in range(10):
                print('nop')
        for variable in self.variable_actives:
            variable.onControlFlowEscape(self)

    def removeKnowledge(self, node):
        if False:
            while True:
                i = 10
        if node.isExpressionVariableRef():
            node.variable.removeKnowledge(self)

    def onValueEscapeStr(self, node):
        if False:
            print('Hello World!')
        pass

    def removeAllKnowledge(self):
        if False:
            return 10
        for variable in self.variable_actives:
            variable.removeAllKnowledge(self)

    def onVariableSet(self, variable, version, assign_node):
        if False:
            i = 10
            return i + 15
        variable_trace = ValueTraceAssign(owner=self.owner, assign_node=assign_node, previous=self.getVariableCurrentTrace(variable))
        self.addVariableTrace(variable, version, variable_trace)
        self.markCurrentVariableTrace(variable, version)
        return variable_trace

    def onVariableSetAliasing(self, variable, version, assign_node, source):
        if False:
            print('Hello World!')
        other_variable_trace = source.variable_trace
        if other_variable_trace.__class__ is ValueTraceAssignUnescapable:
            return self.onVariableSetToUnescapableValue(variable=variable, version=version, assign_node=assign_node)
        elif other_variable_trace.__class__ is ValueTraceAssignVeryTrusted:
            return self.onVariableSetToVeryTrustedValue(variable=variable, version=version, assign_node=assign_node)
        else:
            result = self.onVariableSet(variable=variable, version=version, assign_node=assign_node)
            self.removeKnowledge(source)
            return result

    def onVariableSetToUnescapableValue(self, variable, version, assign_node):
        if False:
            print('Hello World!')
        variable_trace = ValueTraceAssignUnescapable(owner=self.owner, assign_node=assign_node, previous=self.getVariableCurrentTrace(variable))
        self.addVariableTrace(variable, version, variable_trace)
        self.markCurrentVariableTrace(variable, version)
        return variable_trace

    def onVariableSetToVeryTrustedValue(self, variable, version, assign_node):
        if False:
            print('Hello World!')
        variable_trace = ValueTraceAssignVeryTrusted(owner=self.owner, assign_node=assign_node, previous=self.getVariableCurrentTrace(variable))
        self.addVariableTrace(variable, version, variable_trace)
        self.markCurrentVariableTrace(variable, version)
        return variable_trace

    def onVariableSetToUnescapablePropagatedValue(self, variable, version, assign_node, replacement):
        if False:
            return 10
        variable_trace = ValueTraceAssignUnescapablePropagated(owner=self.owner, assign_node=assign_node, previous=self.getVariableCurrentTrace(variable), replacement=replacement)
        self.addVariableTrace(variable, version, variable_trace)
        self.markCurrentVariableTrace(variable, version)
        return variable_trace

    def onVariableDel(self, variable, version, del_node):
        if False:
            while True:
                i = 10
        old_trace = self.getVariableCurrentTrace(variable)
        variable_trace = ValueTraceDeleted(owner=self.owner, del_node=del_node, previous=old_trace)
        self.addVariableTrace(variable, version, variable_trace)
        self.markCurrentVariableTrace(variable, version)
        return variable_trace

    def onLocalsUsage(self, locals_scope):
        if False:
            print('Hello World!')
        self.onLocalsDictEscaped(locals_scope)
        result = []
        scope_locals_variables = locals_scope.getLocalsRelevantVariables()
        for variable in self.variable_actives:
            if variable.isLocalVariable() and variable in scope_locals_variables:
                variable_trace = self.getVariableCurrentTrace(variable)
                variable_trace.addNameUsage()
                result.append((variable, variable_trace))
        return result

    def onVariableContentEscapes(self, variable):
        if False:
            for i in range(10):
                print('nop')
        self.markActiveVariableAsEscaped(variable)

    def onExpression(self, expression, allow_none=False):
        if False:
            return 10
        if expression is None and allow_none:
            return None
        parent = expression.parent
        assert parent, expression
        (new_node, change_tags, change_desc) = expression.computeExpressionRaw(self)
        if change_tags is not None:
            self.signalChange(change_tags, expression.getSourceReference(), change_desc)
        if new_node is not expression:
            parent.replaceChild(expression, new_node)
        return new_node

    def onStatement(self, statement):
        if False:
            print('Hello World!')
        try:
            (new_statement, change_tags, change_desc) = statement.computeStatement(self)
            if new_statement is not statement:
                self.signalChange(change_tags, statement.getSourceReference(), change_desc)
            return new_statement
        except Exception:
            printError('Problem with statement at %s:\n-> %s' % (statement.source_ref.getAsString(), readSourceLine(statement.source_ref)))
            raise

    def computedStatementResult(self, statement, change_tags, change_desc):
        if False:
            i = 10
            return i + 15
        'Make sure the replacement statement is computed.\n\n        Use this when a replacement expression needs to be seen by the trace\n        collection and be computed, without causing any duplication, but where\n        otherwise there would be loss of annotated effects.\n\n        This may e.g. be true for nodes that need an initial run to know their\n        exception result and type shape.\n        '
        new_statement = statement.computeStatement(self)
        if new_statement[0] is not statement:
            self.signalChange(change_tags, statement.getSourceReference(), change_desc)
            return new_statement
        else:
            return (statement, change_tags, change_desc)

    def computedExpressionResult(self, expression, change_tags, change_desc):
        if False:
            i = 10
            return i + 15
        'Make sure the replacement expression is computed.\n\n        Use this when a replacement expression needs to be seen by the trace\n        collection and be computed, without causing any duplication, but where\n        otherwise there would be loss of annotated effects.\n\n        This may e.g. be true for nodes that need an initial run to know their\n        exception result and type shape.\n        '
        new_expression = expression.computeExpression(self)
        if new_expression[0] is not expression:
            self.signalChange(change_tags, expression.getSourceReference(), change_desc)
            return new_expression
        else:
            return (expression, change_tags, change_desc)

    def computedExpressionResultRaw(self, expression, change_tags, change_desc):
        if False:
            return 10
        'Make sure the replacement expression is computed.\n\n        Use this when a replacement expression needs to be seen by the trace\n        collection and be computed, without causing any duplication, but where\n        otherwise there would be loss of annotated effects.\n\n        This may e.g. be true for nodes that need an initial run to know their\n        exception result and type shape.\n\n        This is for raw, i.e. subnodes are not yet computed automatically.\n        '
        new_expression = expression.computeExpressionRaw(self)
        if new_expression[0] is not expression:
            self.signalChange(change_tags, expression.getSourceReference(), change_desc)
            return new_expression
        else:
            return (expression, change_tags, change_desc)

    def mergeBranches(self, collection_yes, collection_no):
        if False:
            print('Hello World!')
        'Merge two alternative branches into this trace.\n\n        This is mostly for merging conditional branches, or other ways\n        of having alternative control flow. This deals with up to two\n        alternative branches to both change this collection.\n        '
        if collection_yes is None:
            if collection_no is not None:
                collection1 = self
                collection2 = collection_no
            else:
                return
        elif collection_no is None:
            collection1 = self
            collection2 = collection_yes
        else:
            collection1 = collection_yes
            collection2 = collection_no
        variable_versions = {}
        for (variable, version) in iterItems(collection1.variable_actives):
            variable_versions[variable] = version
        for (variable, version) in iterItems(collection2.variable_actives):
            if variable not in variable_versions:
                if version != 0:
                    variable_versions[variable] = (0, version)
                else:
                    variable_versions[variable] = 0
            else:
                other = variable_versions[variable]
                if other != version:
                    variable_versions[variable] = (other, version)
        for variable in variable_versions:
            if variable not in collection2.variable_actives:
                if variable_versions[variable] != 0:
                    variable_versions[variable] = (variable_versions[variable], 0)
        self.variable_actives = {}
        for (variable, versions) in iterItems(variable_versions):
            if type(versions) is tuple:
                trace1 = self.getVariableTrace(variable, versions[0])
                trace2 = self.getVariableTrace(variable, versions[1])
                if trace1.isEscapeTrace() and trace1.previous is trace2:
                    version = versions[0]
                elif trace2 is trace1.isEscapeTrace() and trace2.previous is trace1:
                    version = versions[1]
                else:
                    version = self.addVariableMergeMultipleTrace(variable=variable, traces=(trace1, trace2))
            else:
                version = versions
            self.markCurrentVariableTrace(variable, version)

    def mergeMultipleBranches(self, collections):
        if False:
            print('Hello World!')
        assert collections
        merge_size = len(collections)
        if merge_size == 1:
            self.replaceBranch(collections[0])
            return
        elif merge_size == 2:
            return self.mergeBranches(*collections)
        with TimerReport(message='Running merge for %s took %%.2f seconds' % collections, decider=lambda : 0):
            variable_versions = defaultdict(OrderedSet)
            for collection in collections:
                for (variable, version) in iterItems(collection.variable_actives):
                    variable_versions[variable].add(version)
            for collection in collections:
                for (variable, versions) in iterItems(variable_versions):
                    if variable not in collection.variable_actives:
                        versions.add(0)
            self.variable_actives = {}
            for (variable, versions) in iterItems(variable_versions):
                if len(versions) == 1:
                    (version,) = versions
                else:
                    traces = []
                    escaped = []
                    winner_version = None
                    for version in versions:
                        trace = self.getVariableTrace(variable, version)
                        if trace.isEscapeTrace():
                            winner_version = version
                            escaped_trace = trace.previous
                            if escaped_trace in traces:
                                traces.remove(trace.previous)
                            escaped.append(escaped)
                            traces.append(trace)
                        elif trace not in escaped:
                            traces.append(trace)
                    if len(traces) == 1:
                        version = winner_version
                        assert winner_version is not None
                    else:
                        version = self.addVariableMergeMultipleTrace(variable=variable, traces=tuple(traces))
                self.markCurrentVariableTrace(variable, version)

    def replaceBranch(self, collection_replace):
        if False:
            for i in range(10):
                print('nop')
        self.variable_actives.update(collection_replace.variable_actives)
        collection_replace.variable_actives = None

    def onLoopBreak(self, collection=None):
        if False:
            return 10
        if collection is None:
            collection = self
        return self.parent.onLoopBreak(collection)

    def onLoopContinue(self, collection=None):
        if False:
            for i in range(10):
                print('nop')
        if collection is None:
            collection = self
        return self.parent.onLoopContinue(collection)

    def onFunctionReturn(self, collection=None):
        if False:
            print('Hello World!')
        if collection is None:
            collection = self
        return self.parent.onFunctionReturn(collection)

    def onExceptionRaiseExit(self, raisable_exceptions, collection=None):
        if False:
            i = 10
            return i + 15
        if collection is None:
            collection = self
        return self.parent.onExceptionRaiseExit(raisable_exceptions, collection)

    def getLoopBreakCollections(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getLoopBreakCollections()

    def getLoopContinueCollections(self):
        if False:
            i = 10
            return i + 15
        return self.parent.getLoopContinueCollections()

    def getFunctionReturnCollections(self):
        if False:
            while True:
                i = 10
        return self.parent.getFunctionReturnCollections()

    def getExceptionRaiseCollections(self):
        if False:
            return 10
        return self.parent.getExceptionRaiseCollections()

    def makeAbortStackContext(self, catch_breaks, catch_continues, catch_returns, catch_exceptions):
        if False:
            i = 10
            return i + 15
        return self.parent.makeAbortStackContext(catch_breaks=catch_breaks, catch_continues=catch_continues, catch_returns=catch_returns, catch_exceptions=catch_exceptions)

    def onLocalsDictEscaped(self, locals_scope):
        if False:
            for i in range(10):
                print('nop')
        self.parent.onLocalsDictEscaped(locals_scope)

    def getCompileTimeComputationResult(self, node, computation, description, user_provided=False):
        if False:
            i = 10
            return i + 15
        (new_node, change_tags, message) = getComputationResult(node=node, computation=computation, description=description, user_provided=user_provided)
        if change_tags == 'new_raise':
            self.onExceptionRaiseExit(BaseException)
        return (new_node, change_tags, message)

    def addOutlineFunction(self, outline):
        if False:
            while True:
                i = 10
        self.parent.addOutlineFunction(outline)

    def getVeryTrustedModuleVariables(self):
        if False:
            while True:
                i = 10
        return self.parent.getVeryTrustedModuleVariables()

    def onUsedFunction(self, function_body):
        if False:
            print('Hello World!')
        return self.parent.onUsedFunction(function_body)

    def onModuleUsageAttempt(self, module_usage_attempt):
        if False:
            print('Hello World!')
        self.parent.onModuleUsageAttempt(module_usage_attempt)

    def onDistributionUsed(self, distribution_name, node, success):
        if False:
            while True:
                i = 10
        self.parent.onDistributionUsed(distribution_name=distribution_name, node=node, success=success)

class TraceCollectionBranch(CollectionUpdateMixin, TraceCollectionBase):
    __slots__ = ('variable_traces',)

    def __init__(self, name, parent):
        if False:
            i = 10
            return i + 15
        TraceCollectionBase.__init__(self, owner=parent.owner, name=name, parent=parent)
        self.variable_actives = dict(parent.variable_actives)
        self.variable_traces = parent.variable_traces

    def computeBranch(self, branch):
        if False:
            for i in range(10):
                print('nop')
        assert branch.isStatementsSequence()
        result = branch.computeStatementsSequence(self)
        if result is not branch:
            branch.parent.replaceChild(branch, result)
        return result

    def initVariable(self, variable):
        if False:
            i = 10
            return i + 15
        variable_trace = self.parent.initVariable(variable)
        self.variable_actives[variable] = 0
        return variable_trace

class TraceCollectionFunction(CollectionStartPointMixin, TraceCollectionBase):
    __slots__ = ('variable_versions', 'variable_traces', 'break_collections', 'continue_collections', 'return_collections', 'exception_collections', 'outline_functions', 'very_trusted_module_variables')

    def __init__(self, parent, function_body):
        if False:
            print('Hello World!')
        assert function_body.isExpressionFunctionBody() or function_body.isExpressionGeneratorObjectBody() or function_body.isExpressionCoroutineObjectBody() or function_body.isExpressionAsyncgenObjectBody(), function_body
        CollectionStartPointMixin.__init__(self)
        TraceCollectionBase.__init__(self, owner=function_body, name='collection_' + function_body.getCodeName(), parent=parent)
        if parent is not None:
            self.very_trusted_module_variables = parent.getVeryTrustedModuleVariables()
        else:
            self.very_trusted_module_variables = ()
        if function_body.isExpressionFunctionBody():
            parameters = function_body.getParameters()
            for parameter_variable in parameters.getTopLevelVariables():
                self.initVariableInit(parameter_variable)
                self.variable_actives[parameter_variable] = 0
            list_star_variable = parameters.getListStarArgVariable()
            if list_star_variable is not None:
                self.initVariableInitStarArgs(list_star_variable)
                self.variable_actives[list_star_variable] = 0
            dict_star_variable = parameters.getDictStarArgVariable()
            if dict_star_variable is not None:
                self.initVariableInitStarDict(dict_star_variable)
                self.variable_actives[dict_star_variable] = 0
        for closure_variable in function_body.getClosureVariables():
            self.initVariableUnknown(closure_variable)
            self.variable_actives[closure_variable] = 0
        locals_scope = function_body.getLocalsScope()
        if locals_scope is not None:
            if not locals_scope.isMarkedForPropagation():
                for locals_dict_variable in locals_scope.variables.values():
                    self.initVariableUninitialized(locals_dict_variable)
            else:
                function_body.locals_scope = None

    def initVariableModule(self, variable):
        if False:
            while True:
                i = 10
        trusted_node = self.very_trusted_module_variables.get(variable)
        if trusted_node is None:
            return CollectionStartPointMixin.initVariableModule(self, variable)
        assign_trace = ValueTraceAssign(self.owner, assign_node=trusted_node.getParent(), previous=None)
        self.addVariableTrace(variable, 0, assign_trace)
        self.markActiveVariableAsEscaped(variable)
        return self.getVariableCurrentTrace(variable)

class TraceCollectionPureFunction(TraceCollectionFunction):
    """Pure functions don't feed their parent."""
    __slots__ = ('used_functions',)

    def __init__(self, function_body):
        if False:
            while True:
                i = 10
        TraceCollectionFunction.__init__(self, parent=None, function_body=function_body)
        self.used_functions = OrderedSet()

    def getUsedFunctions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.used_functions

    def onUsedFunction(self, function_body):
        if False:
            while True:
                i = 10
        self.used_functions.add(function_body)
        TraceCollectionFunction.onUsedFunction(self, function_body=function_body)

class TraceCollectionModule(CollectionStartPointMixin, TraceCollectionBase):
    __slots__ = ('variable_versions', 'variable_traces', 'break_collections', 'continue_collections', 'return_collections', 'exception_collections', 'outline_functions', 'very_trusted_module_variables', 'module_usage_attempts', 'distribution_names')

    def __init__(self, module, very_trusted_module_variables):
        if False:
            while True:
                i = 10
        assert module.isCompiledPythonModule(), module
        CollectionStartPointMixin.__init__(self)
        TraceCollectionBase.__init__(self, owner=module, name='module:' + module.getFullName(), parent=None)
        self.very_trusted_module_variables = very_trusted_module_variables
        self.module_usage_attempts = OrderedSet()
        self.distribution_names = OrderedDict()

    def getVeryTrustedModuleVariables(self):
        if False:
            for i in range(10):
                print('nop')
        return self.very_trusted_module_variables

    def updateVeryTrustedModuleVariables(self, very_trusted_module_variables):
        if False:
            return 10
        result = self.very_trusted_module_variables != very_trusted_module_variables
        self.very_trusted_module_variables = very_trusted_module_variables
        return result

    def getModuleUsageAttempts(self):
        if False:
            for i in range(10):
                print('nop')
        return self.module_usage_attempts

    def onModuleUsageAttempt(self, module_usage_attempt):
        if False:
            while True:
                i = 10
        if module_usage_attempt.finding not in ('not-found', 'built-in'):
            (decision, _reason) = decideRecursion(using_module_name=self.owner.getFullName(), module_name=module_usage_attempt.module_name, module_filename=module_usage_attempt.filename, module_kind=module_usage_attempt.module_kind)
            if decision is False:
                parent_package_name = module_usage_attempt.module_name.getPackageName()
                if parent_package_name is not None:
                    (package_module_name, module_filename, module_kind, finding) = locateModule(module_name=parent_package_name, parent_package=None, level=0)
                    assert finding != 'not-found', package_module_name
                    (decision, _reason) = decideRecursion(using_module_name=self.owner.getFullName(), module_name=package_module_name, module_filename=module_filename, module_kind=module_kind)
                    if decision is True:
                        self.onModuleUsageAttempt(makeModuleUsageAttempt(module_name=package_module_name, filename=module_filename, finding=finding, module_kind=module_kind, level=0, source_ref=module_usage_attempt.source_ref, reason='parent import'))
        self.module_usage_attempts.add(module_usage_attempt)

    def getUsedDistributions(self):
        if False:
            while True:
                i = 10
        return self.distribution_names

    def onDistributionUsed(self, distribution_name, node, success):
        if False:
            i = 10
            return i + 15
        inclusion_logger.info_to_file_only("Cannot find distribution '%s' at '%s', expect potential run time problem, unless this is unused code." % (distribution_name, node.source_ref.getAsString()))
        self.distribution_names[distribution_name] = success

def areEmptyTraces(variable_traces):
    if False:
        i = 10
        return i + 15
    'Do these traces contain any writes or accesses.'
    for variable_trace in variable_traces:
        if variable_trace.isAssignTrace():
            return False
        elif variable_trace.isInitTrace():
            return False
        elif variable_trace.isDeletedTrace():
            return False
        elif variable_trace.isUninitializedTrace():
            if variable_trace.getUsageCount():
                return False
        elif variable_trace.isUnknownTrace():
            if variable_trace.getUsageCount():
                return False
        elif variable_trace.isEscapeTrace():
            if variable_trace.getUsageCount():
                return False
        elif variable_trace.isMergeTrace():
            if variable_trace.getUsageCount():
                return False
        elif variable_trace.isLoopTrace():
            return False
        else:
            assert False, variable_trace
    return True

def areReadOnlyTraces(variable_traces):
    if False:
        for i in range(10):
            print('nop')
    'Do these traces contain any writes.'
    for variable_trace in variable_traces:
        if variable_trace.isAssignTrace():
            return False
        elif variable_trace.isInitTrace():
            pass
        elif variable_trace.isDeletedTrace():
            return False
        elif variable_trace.isUninitializedTrace():
            pass
        elif variable_trace.isUnknownTrace():
            return False
        elif variable_trace.isEscapeTrace():
            pass
        elif variable_trace.isMergeTrace():
            pass
        elif variable_trace.isLoopTrace():
            pass
        else:
            assert False, variable_trace
    return True