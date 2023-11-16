""" Assignment related nodes.

The most simple assignment statement ``a = b`` is what we have here. All others
are either re-formulated using temporary variables, e.g. ``a, b = c`` or are
attribute, slice, subscript assignments.

The deletion is a separate node unlike in CPython where assigning to ``NULL`` is
internally what deletion is. But deleting is something entirely different to us
during code generation, which is why we keep them separate.

Tracing assignments in SSA form is the core of optimization for which we use
the traces.

"""
from abc import abstractmethod
from nuitka.ModuleRegistry import getOwnerFromCodeName
from nuitka.Options import isExperimental
from .ConstantRefNodes import makeConstantRefNode
from .NodeMakingHelpers import makeStatementExpressionOnlyReplacementNode, makeStatementsSequenceReplacementNode
from .shapes.ControlFlowDescriptions import ControlFlowDescriptionElementBasedEscape, ControlFlowDescriptionFullEscape, ControlFlowDescriptionNoEscape
from .shapes.StandardShapes import tshape_iterator, tshape_unknown
from .StatementBasesGenerated import StatementAssignmentVariableConstantImmutableBase, StatementAssignmentVariableConstantMutableBase, StatementAssignmentVariableFromTempVariableBase, StatementAssignmentVariableFromVariableBase, StatementAssignmentVariableGenericBase, StatementAssignmentVariableHardValueBase, StatementAssignmentVariableIteratorBase
from .VariableDelNodes import makeStatementDelVariable
from .VariableRefNodes import ExpressionTempVariableRef

class StatementAssignmentVariableMixin(object):
    """Assignment to a variable from an expression.

    All assignment forms that are not to attributes, slices, subscripts
    use this.

    The source might be a complex expression. The target can be any kind
    of variable, temporary, local, global, etc.

    Assigning a variable is something we trace in a new version, this is
    hidden behind target variable reference, which has this version once
    it can be determined.
    """
    __slots__ = ()

    @staticmethod
    def isStatementAssignmentVariable():
        if False:
            i = 10
            return i + 15
        return True

    def finalize(self):
        if False:
            return 10
        del self.variable
        del self.variable_trace
        self.subnode_source.finalize()
        del self.subnode_source

    def getDetailsForDisplay(self):
        if False:
            i = 10
            return i + 15
        return {'variable_name': self.getVariableName(), 'is_temp': self.variable.isTempVariable(), 'var_type': self.variable.getVariableType(), 'owner': self.variable.getOwner().getCodeName()}

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            for i in range(10):
                print('nop')
        owner = getOwnerFromCodeName(args['owner'])
        if args['is_temp'] == 'True':
            variable = owner.createTempVariable(args['variable_name'], temp_type=['var_type'])
        else:
            variable = owner.getProvidedVariable(args['variable_name'])
        del args['is_temp']
        del args['var_type']
        del args['owner']
        version = variable.allocateTargetNumber()
        return cls(variable=variable, version=version, source_ref=source_ref, **args)

    def makeClone(self):
        if False:
            for i in range(10):
                print('nop')
        version = self.variable.allocateTargetNumber()
        return self.__class__(source=self.subnode_source.makeClone(), variable=self.variable, variable_version=version, source_ref=self.source_ref)

    def getVariableName(self):
        if False:
            while True:
                i = 10
        return self.variable.getName()

    def getVariable(self):
        if False:
            return 10
        return self.variable

    def setVariable(self, variable):
        if False:
            print('Hello World!')
        self.variable = variable
        self.variable_version = variable.allocateTargetNumber()

    def getVariableTrace(self):
        if False:
            return 10
        return self.variable_trace

    def markAsInplaceSuspect(self):
        if False:
            for i in range(10):
                print('nop')
        self.inplace_suspect = True

    def isInplaceSuspect(self):
        if False:
            for i in range(10):
                print('nop')
        return self.inplace_suspect

    def removeMarkAsInplaceSuspect(self):
        if False:
            while True:
                i = 10
        self.inplace_suspect = False

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_source.mayRaiseException(exception_type)

    def needsReleasePreviousValue(self):
        if False:
            for i in range(10):
                print('nop')
        previous = self.variable_trace.getPrevious()
        if previous.mustNotHaveValue():
            return False
        elif previous.mustHaveValue():
            return True
        else:
            return None

    @staticmethod
    def getStatementNiceName():
        if False:
            while True:
                i = 10
        return 'variable assignment statement'

    def getTypeShape(self):
        if False:
            return 10
        try:
            source = self.subnode_source
        except AttributeError:
            return tshape_unknown
        return source.getTypeShape()

    @staticmethod
    def mayHaveSideEffects():
        if False:
            while True:
                i = 10
        return True

    def _transferState(self, result):
        if False:
            for i in range(10):
                print('nop')
        self.variable_trace.assign_node = result
        result.variable_trace = self.variable_trace
        self.variable_trace = None

    def _considerSpecialization(self, old_source):
        if False:
            for i in range(10):
                print('nop')
        source = self.subnode_source
        if source is old_source:
            return (self, None, None)
        if source.isCompileTimeConstant():
            result = makeStatementAssignmentVariableConstant(source=source, variable=self.variable, variable_version=self.variable_version, very_trusted=old_source.isExpressionImportName(), source_ref=self.source_ref)
            self._transferState(result)
            return (result, 'new_statements', "Assignment source of '%s' is now compile time constant." % self.getVariableName())
        if source.isExpressionVariableRef():
            result = StatementAssignmentVariableFromVariable(source=source, variable=self.variable, variable_version=self.variable_version, source_ref=self.source_ref)
            self._transferState(result)
            return (result, 'new_statements', 'Assignment source is now variable reference.')
        if source.isExpressionTempVariableRef():
            result = StatementAssignmentVariableFromTempVariable(source=source, variable=self.variable, variable_version=self.variable_version, source_ref=self.source_ref)
            self._transferState(result)
            return (result, 'new_statements', 'Assignment source is now temp variable reference.')
        if source.getTypeShape().isShapeIterator():
            result = StatementAssignmentVariableIterator(source=source, variable=self.variable, variable_version=self.variable_version, source_ref=self.source_ref)
            self._transferState(result)
            return (result, 'new_statements', 'Assignment source is now known to be iterator.')
        if source.hasVeryTrustedValue():
            result = StatementAssignmentVariableHardValue(source=source, variable=self.variable, variable_version=self.variable_version, source_ref=self.source_ref)
            self._transferState(result)
            return (result, 'new_statements', 'Assignment source is now known to be hard import.')
        return (self, None, None)

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            for i in range(10):
                print('nop')
        emit_write(self.variable)
        self.subnode_source.collectVariableAccesses(emit_read, emit_write)

    @abstractmethod
    def hasVeryTrustedValue(self):
        if False:
            while True:
                i = 10
        'Does this assignment node have a very trusted value.'

class StatementAssignmentVariableGeneric(StatementAssignmentVariableMixin, StatementAssignmentVariableGenericBase):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_GENERIC'
    named_children = ('source|setter',)
    nice_children = ('assignment source',)
    node_attributes = ('variable', 'variable_version')
    auto_compute_handling = 'post_init'
    __slots__ = ('variable_trace', 'inplace_suspect')

    def postInitNode(self):
        if False:
            for i in range(10):
                print('nop')
        self.variable_trace = None
        self.inplace_suspect = None

    @staticmethod
    def getReleaseEscape():
        if False:
            while True:
                i = 10
        return ControlFlowDescriptionFullEscape

    def computeStatement(self, trace_collection):
        if False:
            return 10
        old_source = self.subnode_source
        variable = self.variable
        if old_source.isExpressionSideEffects():
            statements = [makeStatementExpressionOnlyReplacementNode(side_effect, self) for side_effect in old_source.subnode_side_effects]
            statements.append(self)
            parent = self.parent
            self.setChildSource(old_source.subnode_expression)
            result = makeStatementsSequenceReplacementNode(statements=statements, node=self)
            result.parent = parent
            return (result.computeStatementsSequence(trace_collection), 'new_statements', 'Side effects of assignments promoted to statements.')
        source = trace_collection.onExpression(self.subnode_source)
        if source.willRaiseAnyException():
            result = makeStatementExpressionOnlyReplacementNode(expression=source, node=self)
            del self.parent
            return (result, 'new_raise', 'Assignment raises exception in assigned value, removed assignment.')
        if not variable.isModuleVariable() and source.isExpressionVariableRef() and (source.getVariable() is variable):
            if source.mayHaveSideEffects():
                result = makeStatementExpressionOnlyReplacementNode(expression=source, node=self)
                return (result, 'new_statements', 'Lowered assignment of %s from itself to mere access of it.' % variable.getDescription())
            else:
                return (None, 'new_statements', 'Removed assignment of %s from itself which is known to be defined.' % variable.getDescription())
        self.variable_trace = trace_collection.onVariableSet(variable=variable, version=self.variable_version, assign_node=self)
        trace_collection.removeKnowledge(source)
        return self._considerSpecialization(old_source)

    def hasVeryTrustedValue(self):
        if False:
            for i in range(10):
                print('nop')
        'Does this assignment node have a very trusted value.'
        return self.subnode_source.hasVeryTrustedValue()

class StatementAssignmentVariableIterator(StatementAssignmentVariableMixin, StatementAssignmentVariableIteratorBase):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_ITERATOR'
    named_children = ('source',)
    nice_children = ('assignment source',)
    node_attributes = ('variable', 'variable_version')
    auto_compute_handling = 'post_init'
    __slots__ = ('variable_trace', 'inplace_suspect', 'type_shape', 'temp_scope', 'tmp_iterated_variable', 'tmp_iteration_count_variable', 'tmp_iteration_next_variable', 'is_indexable')

    def postInitNode(self):
        if False:
            i = 10
            return i + 15
        self.variable_trace = None
        self.inplace_suspect = None
        self.type_shape = tshape_iterator
        self.temp_scope = None
        self.tmp_iterated_variable = None
        self.tmp_iteration_count_variable = None
        self.tmp_iteration_next_variable = None
        self.is_indexable = None

    def getTypeShape(self):
        if False:
            i = 10
            return i + 15
        return self.type_shape

    @staticmethod
    def getReleaseEscape():
        if False:
            i = 10
            return i + 15
        return ControlFlowDescriptionElementBasedEscape

    def getIterationIndexDesc(self):
        if False:
            for i in range(10):
                print('nop')
        'For use in optimization outputs only, here and using nodes.'
        return "'%s[%s]'" % (self.tmp_iterated_variable.getName(), self.tmp_iteration_count_variable.getName())

    def _replaceWithDirectAccess(self, trace_collection, provider):
        if False:
            print('Hello World!')
        self.temp_scope = provider.allocateTempScope('iterator_access')
        self.tmp_iterated_variable = provider.allocateTempVariable(temp_scope=self.temp_scope, name='iterated_value', temp_type='object')
        reference_iterated = ExpressionTempVariableRef(variable=self.tmp_iterated_variable, source_ref=self.subnode_source.source_ref)
        iterated_value = self.subnode_source.subnode_value
        assign_iterated = makeStatementAssignmentVariable(source=iterated_value, variable=self.tmp_iterated_variable, variable_version=None, source_ref=iterated_value.source_ref)
        self.tmp_iteration_count_variable = provider.allocateTempVariable(temp_scope=self.temp_scope, name='iteration_count', temp_type='object')
        assign_iteration_count = makeStatementAssignmentVariable(source=makeConstantRefNode(constant=0, source_ref=self.source_ref), variable=self.tmp_iteration_count_variable, variable_version=None, source_ref=iterated_value.source_ref)
        self.subnode_source.setChildValue(reference_iterated)
        assign_iterated.computeStatement(trace_collection)
        assign_iteration_count.computeStatement(trace_collection)
        reference_iterated.computeExpressionRaw(trace_collection)
        self.variable_trace = trace_collection.onVariableSet(variable=self.variable, version=self.variable_version, assign_node=self)
        self.tmp_iteration_next_variable = provider.allocateTempVariable(temp_scope=self.temp_scope, name='next_value', temp_type='object')
        result = makeStatementsSequenceReplacementNode((assign_iteration_count, assign_iterated, self), self)
        return (result, 'new_statements', lambda : 'Enabling indexing of iterated value through %s.' % self.getIterationIndexDesc())

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        source = self.subnode_source
        variable = self.variable
        provider = trace_collection.getOwner()
        source = trace_collection.onExpression(self.subnode_source)
        if self.tmp_iterated_variable is None and self.is_indexable is None and source.isExpressionBuiltinIterForUnpack() and isExperimental('iterator-optimization'):
            if variable.hasAccessesOutsideOf(provider) is False:
                last_trace = variable.getMatchingUnescapedAssignTrace(self)
                if last_trace is not None:
                    self.is_indexable = source.subnode_value.getTypeShape().hasShapeIndexLookup()
                    if self.is_indexable:
                        return self._replaceWithDirectAccess(trace_collection=trace_collection, provider=provider)
        if source.willRaiseAnyException():
            result = makeStatementExpressionOnlyReplacementNode(expression=source, node=self)
            del self.parent
            return (result, 'new_raise', 'Assignment raises exception in assigned value, removed assignment.')
        self.type_shape = source.getTypeShape()
        self.variable_trace = trace_collection.onVariableSet(variable=variable, version=self.variable_version, assign_node=self)
        return (self, None, None)

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            print('Hello World!')
        'Does this assignment node have a very trusted value.'
        return False

class StatementAssignmentVariableConstantMutable(StatementAssignmentVariableMixin, StatementAssignmentVariableConstantMutableBase):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_CONSTANT_MUTABLE'
    named_children = ('source',)
    nice_children = ('assignment source',)
    node_attributes = ('variable', 'variable_version')
    auto_compute_handling = 'post_init'
    __slots__ = ('variable_trace', 'inplace_suspect')

    def postInitNode(self):
        if False:
            print('Hello World!')
        self.variable_trace = None
        self.inplace_suspect = None

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def getReleaseEscape():
        if False:
            for i in range(10):
                print('nop')
        return ControlFlowDescriptionNoEscape

    def computeStatement(self, trace_collection):
        if False:
            while True:
                i = 10
        variable = self.variable
        self.variable_trace = trace_collection.onVariableSet(variable=variable, version=self.variable_version, assign_node=self)
        provider = trace_collection.getOwner()
        if variable.hasAccessesOutsideOf(provider) is False:
            last_trace = variable.getMatchingAssignTrace(self)
            if last_trace is not None and (not last_trace.getMergeOrNameUsageCount()):
                if variable.isModuleVariable() or variable.owner.locals_scope.isUnoptimizedFunctionScope():
                    pass
                elif not last_trace.getUsageCount():
                    if not last_trace.getPrevious().isUnassignedTrace():
                        result = makeStatementDelVariable(variable=self.variable, version=self.variable_version, tolerant=True, source_ref=self.source_ref)
                    else:
                        result = None
                    return (result, 'new_statements', "Dropped dead assignment statement to '%s'." % self.getVariableName())
        return (self, None, None)

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            while True:
                i = 10
        'Does this assignment node have a very trusted value.'
        return False

class StatementAssignmentVariableConstantImmutable(StatementAssignmentVariableMixin, StatementAssignmentVariableConstantImmutableBase):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_CONSTANT_IMMUTABLE'
    named_children = ('source',)
    nice_children = ('assignment source',)
    node_attributes = ('variable', 'variable_version')
    auto_compute_handling = 'post_init'
    __slots__ = ('variable_trace', 'inplace_suspect')

    def postInitNode(self):
        if False:
            i = 10
            return i + 15
        self.variable_trace = None
        self.inplace_suspect = None

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def getReleaseEscape():
        if False:
            return 10
        return ControlFlowDescriptionNoEscape

    def computeStatement(self, trace_collection):
        if False:
            i = 10
            return i + 15
        variable = self.variable
        provider = trace_collection.getOwner()
        if variable.hasAccessesOutsideOf(provider) is False:
            last_trace = variable.getMatchingAssignTrace(self)
            if last_trace is not None and (not last_trace.getMergeOrNameUsageCount()):
                if variable.isModuleVariable() or variable.owner.locals_scope.isUnoptimizedFunctionScope():
                    pass
                else:
                    if not last_trace.getUsageCount():
                        if not last_trace.getPrevious().isUnassignedTrace():
                            return trace_collection.computedStatementResult(statement=makeStatementDelVariable(variable=self.variable, version=self.variable_version, tolerant=True, source_ref=self.source_ref), change_tags='new_statements', change_desc="Lowered dead assignment statement to '%s' to previous value 'del'." % self.getVariableName())
                        else:
                            return (None, 'new_statements', "Dropped dead assignment statement to '%s'." % self.getVariableName())
                    self.variable_trace = trace_collection.onVariableSetToUnescapablePropagatedValue(variable=variable, version=self.variable_version, assign_node=self, replacement=lambda _replaced_node: self.subnode_source.makeClone())
                    if not last_trace.getPrevious().isUnassignedTrace():
                        result = makeStatementDelVariable(variable=self.variable, version=self.variable_version, tolerant=True, source_ref=self.source_ref)
                    else:
                        result = None
                    return (result, 'new_statements', "Dropped propagated assignment statement to '%s'." % self.getVariableName())
        self.variable_trace = trace_collection.onVariableSetToUnescapableValue(variable=variable, version=self.variable_version, assign_node=self)
        return (self, None, None)

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            while True:
                i = 10
        'Does this assignment node have a very trusted value.'
        return False

class StatementAssignmentVariableConstantMutableTrusted(StatementAssignmentVariableConstantImmutable):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_CONSTANT_MUTABLE_TRUSTED'

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            for i in range(10):
                print('nop')
        return True

class StatementAssignmentVariableConstantImmutableTrusted(StatementAssignmentVariableConstantImmutable):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_CONSTANT_IMMUTABLE_TRUSTED'

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            while True:
                i = 10
        return True

class StatementAssignmentVariableHardValue(StatementAssignmentVariableMixin, StatementAssignmentVariableHardValueBase):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_HARD_VALUE'
    named_children = ('source',)
    nice_children = ('assignment source',)
    node_attributes = ('variable', 'variable_version')
    auto_compute_handling = 'post_init'
    __slots__ = ('variable_trace', 'inplace_suspect')

    def postInitNode(self):
        if False:
            for i in range(10):
                print('nop')
        self.variable_trace = None
        self.inplace_suspect = None

    @staticmethod
    def getReleaseEscape():
        if False:
            for i in range(10):
                print('nop')
        return ControlFlowDescriptionNoEscape

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        variable = self.variable
        source = trace_collection.onExpression(self.subnode_source)
        if source.willRaiseAnyException():
            result = makeStatementExpressionOnlyReplacementNode(expression=source, node=self)
            del self.parent
            return (result, 'new_raise', 'Assignment raises exception in assigned value, removed assignment.')
        self.variable_trace = trace_collection.onVariableSetToVeryTrustedValue(variable=variable, version=self.variable_version, assign_node=self)
        return (self, None, None)

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            return 10
        'Does this assignment node have a very trusted value.'
        return True

class StatementAssignmentVariableFromVariable(StatementAssignmentVariableMixin, StatementAssignmentVariableFromVariableBase):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_FROM_VARIABLE'
    named_children = ('source',)
    nice_children = ('assignment source',)
    node_attributes = ('variable', 'variable_version')
    auto_compute_handling = 'post_init'
    __slots__ = ('variable_trace', 'inplace_suspect')

    def postInitNode(self):
        if False:
            i = 10
            return i + 15
        self.variable_trace = None
        self.inplace_suspect = None

    @staticmethod
    def getReleaseEscape():
        if False:
            i = 10
            return i + 15
        return ControlFlowDescriptionFullEscape

    def computeStatement(self, trace_collection):
        if False:
            while True:
                i = 10
        old_source = self.subnode_source
        variable = self.variable
        if not variable.isModuleVariable() and old_source.getVariable() is variable:
            if old_source.mayHaveSideEffects():
                result = makeStatementExpressionOnlyReplacementNode(expression=old_source, node=self)
                result = trace_collection.onStatement(result)
                return (result, 'new_statements', 'Lowered assignment of %s from itself to mere access of it.' % variable.getDescription())
            else:
                return (None, 'new_statements', 'Removed assignment of %s from itself which is known to be defined.' % variable.getDescription())
        source = trace_collection.onExpression(self.subnode_source)
        if source.isExpressionVariableRef():
            self.variable_trace = trace_collection.onVariableSetAliasing(variable=variable, version=self.variable_version, assign_node=self, source=source)
        else:
            self.variable_trace = trace_collection.onVariableSet(variable=variable, version=self.variable_version, assign_node=self)
            trace_collection.removeKnowledge(source)
        return self._considerSpecialization(old_source)

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            return 10
        'Does this assignment node have a very trusted value.'
        return False

class StatementAssignmentVariableFromTempVariable(StatementAssignmentVariableMixin, StatementAssignmentVariableFromTempVariableBase):
    kind = 'STATEMENT_ASSIGNMENT_VARIABLE_FROM_TEMP_VARIABLE'
    named_children = ('source',)
    nice_children = ('assignment source',)
    node_attributes = ('variable', 'variable_version')
    auto_compute_handling = 'post_init'
    __slots__ = ('variable_trace', 'inplace_suspect')

    def postInitNode(self):
        if False:
            for i in range(10):
                print('nop')
        self.variable_trace = None
        self.inplace_suspect = None

    @staticmethod
    def getReleaseEscape():
        if False:
            while True:
                i = 10
        return ControlFlowDescriptionFullEscape

    def computeStatement(self, trace_collection):
        if False:
            return 10
        old_source = self.subnode_source
        variable = self.variable
        if old_source.getVariable() is variable:
            return (None, 'new_statements', 'Removed assignment of %s from itself which is known to be defined.' % variable.getDescription())
        source = trace_collection.onExpression(self.subnode_source)
        self.variable_trace = trace_collection.onVariableSet(variable=variable, version=self.variable_version, assign_node=self)
        trace_collection.removeKnowledge(source)
        return self._considerSpecialization(old_source)

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            return 10
        'Does this assignment node have a very trusted value.'
        return False

def makeStatementAssignmentVariableConstant(source, variable, variable_version, very_trusted, source_ref):
    if False:
        while True:
            i = 10
    if source.isMutable():
        if very_trusted:
            return StatementAssignmentVariableConstantMutableTrusted(source=source, variable=variable, source_ref=source_ref, variable_version=variable_version)
        else:
            return StatementAssignmentVariableConstantMutable(source=source, variable=variable, source_ref=source_ref, variable_version=variable_version)
    elif very_trusted:
        return StatementAssignmentVariableConstantImmutableTrusted(source=source, variable=variable, source_ref=source_ref, variable_version=variable_version)
    else:
        return StatementAssignmentVariableConstantImmutable(source=source, variable=variable, source_ref=source_ref, variable_version=variable_version)

def makeStatementAssignmentVariable(source, variable, source_ref, variable_version=None):
    if False:
        i = 10
        return i + 15
    assert source is not None, source_ref
    if variable_version is None:
        variable_version = variable.allocateTargetNumber()
    if source.isCompileTimeConstant():
        return makeStatementAssignmentVariableConstant(source=source, variable=variable, variable_version=variable_version, very_trusted=False, source_ref=source_ref)
    elif source.isExpressionVariableRef():
        return StatementAssignmentVariableFromVariable(source=source, variable=variable, variable_version=variable_version, source_ref=source_ref)
    elif source.isExpressionTempVariableRef():
        return StatementAssignmentVariableFromTempVariable(source=source, variable=variable, variable_version=variable_version, source_ref=source_ref)
    elif source.getTypeShape().isShapeIterator():
        return StatementAssignmentVariableIterator(source=source, variable=variable, variable_version=variable_version, source_ref=source_ref)
    elif source.hasVeryTrustedValue():
        return StatementAssignmentVariableHardValue(source=source, variable=variable, variable_version=variable_version, source_ref=source_ref)
    else:
        return StatementAssignmentVariableGeneric(source=source, variable=variable, variable_version=variable_version, source_ref=source_ref)