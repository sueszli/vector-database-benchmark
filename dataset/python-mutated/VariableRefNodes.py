""" Node for variable references.

These represent all variable references in the node tree. Can be in assignments
and its expressions, changing the meaning of course dramatically.

"""
from nuitka import Builtins, Variables
from nuitka.ModuleRegistry import getOwnerFromCodeName
from nuitka.PythonVersions import getUnboundLocalErrorErrorTemplate, python_version
from nuitka.tree.TreeHelpers import makeStatementsSequenceFromStatements
from .ConstantRefNodes import makeConstantRefNode
from .DictionaryNodes import ExpressionDictOperationIn, ExpressionDictOperationItem, ExpressionDictOperationNotIn, StatementDictOperationRemove, StatementDictOperationSet
from .ExpressionBases import ExpressionBase, ExpressionNoSideEffectsMixin
from .ModuleAttributeNodes import ExpressionModuleAttributeLoaderRef, ExpressionModuleAttributeNameRef, ExpressionModuleAttributePackageRef, ExpressionModuleAttributeSpecRef
from .NodeMakingHelpers import makeRaiseExceptionReplacementExpression, makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue
from .OutlineNodes import ExpressionOutlineBody
from .ReturnNodes import makeStatementReturn
from .shapes.StandardShapes import tshape_unknown
from .SubscriptNodes import ExpressionSubscriptLookupForUnpack

class ExpressionVariableRefBase(ExpressionBase):
    __slots__ = ('variable', 'variable_trace')

    def __init__(self, variable, source_ref):
        if False:
            print('Hello World!')
        ExpressionBase.__init__(self, source_ref)
        self.variable = variable
        self.variable_trace = None

    def finalize(self):
        if False:
            return 10
        del self.parent
        del self.variable
        del self.variable_trace

    def getVariableName(self):
        if False:
            i = 10
            return i + 15
        return self.variable.getName()

    def getVariable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable

    def getVariableTrace(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_trace

    def getTypeShape(self):
        if False:
            i = 10
            return i + 15
        if self.variable_trace is None:
            return tshape_unknown
        else:
            return self.variable_trace.getTypeShape()

    def onContentEscapes(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onVariableContentEscapes(self.variable)

    def computeExpressionLen(self, len_node, trace_collection):
        if False:
            while True:
                i = 10
        if self.variable_trace is not None and self.variable_trace.isAssignTrace():
            value = self.variable_trace.getAssignNode().subnode_source
            shape = value.getValueShape()
            has_len = shape.hasShapeSlotLen()
            if has_len is False:
                trace_collection.onExceptionRaiseExit(BaseException)
                return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="object of type '%s' has no len()", operation='len', original_node=len_node, value_node=self)
            elif has_len is True:
                iter_length = value.getIterationLength()
                if iter_length is not None:
                    result = makeConstantRefNode(constant=int(iter_length), source_ref=len_node.getSourceReference())
                    return (result, 'new_constant', lambda : "Predicted 'len' result of variable '%s'." % self.getVariableName())
        trace_collection.markActiveVariableAsEscaped(self.variable)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (len_node, None, None)

    def computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection):
        if False:
            print('Hello World!')
        if self.variable_trace is not None:
            attribute_node = self.variable_trace.getAttributeNode()
            if attribute_node is not None:
                trace_collection.markActiveVariableAsEscaped(self.variable)
                return attribute_node.computeExpressionAttribute(lookup_node=lookup_node, attribute_name=attribute_name, trace_collection=trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.markActiveVariableAsEscaped(self.variable)
        if not self.isKnownToHaveAttribute(attribute_name):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, None, None)

    def mayRaiseExceptionAttributeLookup(self, exception_type, attribute_name):
        if False:
            return 10
        return not self.isKnownToHaveAttribute(attribute_name)

    def isKnownToHaveAttribute(self, attribute_name):
        if False:
            print('Hello World!')
        if self.variable_trace is not None:
            type_shape = self.variable_trace.getTypeShape()
            if type_shape.isKnownToHaveAttribute(attribute_name):
                return True
            attribute_node = self.variable_trace.getAttributeNode()
            if attribute_node is not None:
                return attribute_node.isKnownToHaveAttribute(attribute_name)
        return None

    def computeExpressionImportName(self, import_node, import_name, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return self.computeExpressionAttribute(lookup_node=import_node, attribute_name=import_name, trace_collection=trace_collection)

    def computeExpressionComparisonIn(self, in_node, value_node, trace_collection):
        if False:
            print('Hello World!')
        tags = None
        message = None
        trace_collection.onControlFlowEscape(in_node)
        if self.variable_trace.hasShapeDictionaryExact():
            tags = 'new_expression'
            message = "Check '%s' on dictionary lowered to dictionary '%s'." % (in_node.comparator, in_node.comparator)
            if in_node.comparator == 'In':
                in_node = ExpressionDictOperationIn(key=value_node, dict_arg=self, source_ref=in_node.getSourceReference())
            else:
                in_node = ExpressionDictOperationNotIn(key=value_node, dict_arg=self, source_ref=in_node.getSourceReference())
        if in_node.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (in_node, tags, message)

    def getExpressionDictInConstant(self, value):
        if False:
            return 10
        return self.variable_trace.getDictInValue(value)

    def computeExpressionSetSubscript(self, set_node, subscript, value_node, trace_collection):
        if False:
            return 10
        tags = None
        message = None
        if self.variable_trace.hasShapeDictionaryExact():
            result = StatementDictOperationSet(dict_arg=self, key=subscript, value=value_node, source_ref=set_node.getSourceReference())
            change_tags = 'new_statements'
            change_desc = 'Subscript assignment to dictionary lowered to dictionary assignment.'
            trace_collection.removeKnowledge(self)
            (result2, change_tags2, change_desc2) = result.computeStatementOperation(trace_collection)
            if result2 is not result:
                trace_collection.signalChange(tags=change_tags, source_ref=self.source_ref, message=change_desc)
                return (result2, change_tags2, change_desc2)
            else:
                return (result, change_tags, change_desc)
        trace_collection.removeKnowledge(self)
        trace_collection.removeKnowledge(value_node)
        trace_collection.onControlFlowEscape(self)
        if set_node.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (set_node, tags, message)

    def computeExpressionDelSubscript(self, del_node, subscript, trace_collection):
        if False:
            while True:
                i = 10
        tags = None
        message = None
        if self.variable_trace.hasShapeDictionaryExact():
            result = StatementDictOperationRemove(dict_arg=self, key=subscript, source_ref=del_node.getSourceReference())
            change_tags = 'new_statements'
            change_desc = 'Subscript del to dictionary lowered to dictionary del.'
            trace_collection.removeKnowledge(self)
            (result2, change_tags2, change_desc2) = result.computeStatementOperation(trace_collection)
            if result2 is not result:
                trace_collection.signalChange(tags=change_tags, source_ref=self.source_ref, message=change_desc)
                return (result2, change_tags2, change_desc2)
            else:
                return (result, change_tags, change_desc)
        trace_collection.onControlFlowEscape(self)
        if del_node.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (del_node, tags, message)

    def computeExpressionSubscript(self, lookup_node, subscript, trace_collection):
        if False:
            i = 10
            return i + 15
        tags = None
        message = None
        if self.variable_trace.hasShapeDictionaryExact():
            return trace_collection.computedExpressionResult(expression=ExpressionDictOperationItem(dict_arg=self, key=subscript, source_ref=lookup_node.getSourceReference()), change_tags='new_expression', change_desc='Subscript look-up to dictionary lowered to dictionary look-up.')
        trace_collection.onControlFlowEscape(self)
        if lookup_node.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, tags, message)

    def _applyReplacement(self, trace_collection, replacement):
        if False:
            while True:
                i = 10
        trace_collection.signalChange('new_expression', self.source_ref, "Value propagated for '%s' from '%s'." % (self.variable.getName(), replacement.getSourceReference().getAsString()))
        if self.parent.isExpressionOperationInplace():
            statement = self.parent.parent
            if statement.isStatementAssignmentVariable():
                statement.removeMarkAsInplaceSuspect()
        return replacement.computeExpressionRaw(trace_collection)

    def getTruthValue(self):
        if False:
            while True:
                i = 10
        return self.variable_trace.getTruthValue()

    def getComparisonValue(self):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_trace.getComparisonValue()
_hard_names = ('dir', 'eval', 'exec', 'execfile', 'locals', 'vars', 'super')

class ExpressionVariableRef(ExpressionVariableRefBase):
    kind = 'EXPRESSION_VARIABLE_REF'
    __slots__ = ()

    def __init__(self, variable, source_ref):
        if False:
            return 10
        assert variable is not None
        ExpressionVariableRefBase.__init__(self, variable=variable, source_ref=source_ref)

    @staticmethod
    def isExpressionVariableRef():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getDetails(self):
        if False:
            for i in range(10):
                print('nop')
        return {'variable': self.variable}

    def getDetailsForDisplay(self):
        if False:
            for i in range(10):
                print('nop')
        return {'variable_name': self.variable.getName(), 'owner': self.variable.getOwner().getCodeName()}

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            print('Hello World!')
        assert cls is ExpressionVariableRef, cls
        owner = getOwnerFromCodeName(args['owner'])
        variable = owner.getProvidedVariable(args['variable_name'])
        return cls(variable=variable, source_ref=source_ref)

    def getVariable(self):
        if False:
            return 10
        return self.variable

    def setVariable(self, variable):
        if False:
            print('Hello World!')
        assert isinstance(variable, Variables.Variable), repr(variable)
        self.variable = variable

    def computeExpressionRaw(self, trace_collection):
        if False:
            while True:
                i = 10
        variable = self.variable
        assert variable is not None
        self.variable_trace = trace_collection.getVariableCurrentTrace(variable=variable)
        replacement = self.variable_trace.getReplacementNode(self)
        if replacement is not None:
            return self._applyReplacement(trace_collection, replacement)
        if not self.variable_trace.mustHaveValue():
            trace_collection.onExceptionRaiseExit(BaseException)
        if variable.isModuleVariable() and (variable.hasDefiniteWrites() is False or variable.getName() == 'super'):
            variable_name = self.variable.getName()
            if variable_name in Builtins.builtin_exception_names:
                if not self.variable.getOwner().getLocalsScope().isEscaped():
                    from .BuiltinRefNodes import ExpressionBuiltinExceptionRef
                    new_node = ExpressionBuiltinExceptionRef(exception_name=self.variable.getName(), source_ref=self.source_ref)
                    change_tags = 'new_builtin_ref'
                    change_desc = "Module variable '%s' found to be built-in exception reference." % variable_name
                else:
                    self.variable_trace.addUsage()
                    new_node = self
                    change_tags = None
                    change_desc = None
            elif variable_name in Builtins.builtin_names:
                if variable_name in _hard_names or not self.variable.getOwner().getLocalsScope().isEscaped():
                    from .BuiltinRefNodes import makeExpressionBuiltinRef
                    new_node = makeExpressionBuiltinRef(builtin_name=variable_name, locals_scope=self.getFunctionsLocalsScope(), source_ref=self.source_ref)
                    change_tags = 'new_builtin_ref'
                    change_desc = "Module variable '%s' found to be built-in reference." % variable_name
                else:
                    self.variable_trace.addUsage()
                    new_node = self
                    change_tags = None
                    change_desc = None
            elif variable_name == '__name__':
                new_node = ExpressionModuleAttributeNameRef(variable=variable, source_ref=self.source_ref)
                change_tags = 'new_expression'
                change_desc = "Replaced read-only module attribute '__name__' with module attribute reference."
            elif variable_name == '__package__':
                new_node = ExpressionModuleAttributePackageRef(variable=variable, source_ref=self.source_ref)
                change_tags = 'new_expression'
                change_desc = "Replaced read-only module attribute '__package__' with module attribute reference."
            elif variable_name == '__loader__' and python_version >= 768:
                new_node = ExpressionModuleAttributeLoaderRef(variable=variable, source_ref=self.source_ref)
                change_tags = 'new_expression'
                change_desc = "Replaced read-only module attribute '__loader__' with module attribute reference."
            elif variable_name == '__spec__' and python_version >= 832:
                new_node = ExpressionModuleAttributeSpecRef(variable=variable, source_ref=self.source_ref)
                change_tags = 'new_expression'
                change_desc = "Replaced read-only module attribute '__spec__' with module attribute reference."
            else:
                self.variable_trace.addUsage()
                new_node = self
                change_tags = None
                change_desc = None
            return (new_node, change_tags, change_desc)
        self.variable_trace.addUsage()
        if self.variable_trace.mustNotHaveValue():
            assert self.variable.isLocalVariable(), self.variable
            variable_name = self.variable.getName()
            result = makeRaiseExceptionReplacementExpression(expression=self, exception_type='UnboundLocalError', exception_value=getUnboundLocalErrorErrorTemplate() % variable_name)
            return (result, 'new_raise', "Variable access of not initialized variable '%s'" % variable_name)
        return (self, None, None)

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            i = 10
            return i + 15
        if self.variable_trace is not None:
            attribute_node = self.variable_trace.getAttributeNode()
            if attribute_node is not None:
                trace_collection.markActiveVariableAsEscaped(self.variable)
                return attribute_node.computeExpressionCallViaVariable(call_node=call_node, variable_ref_node=self, call_args=call_args, call_kw=call_kw, trace_collection=trace_collection)
        self.onContentEscapes(trace_collection)
        if call_args is not None:
            call_args.onContentEscapes(trace_collection)
        if call_kw is not None:
            call_kw.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        if self.variable.getName() in _hard_names and self.variable.isIncompleteModuleVariable():
            trace_collection.onLocalsUsage(locals_scope=self.getFunctionsLocalsScope())
        return (call_node, None, None)

    def computeExpressionBool(self, trace_collection):
        if False:
            i = 10
            return i + 15
        if self.variable_trace is not None:
            attribute_node = self.variable_trace.getAttributeNode()
            if attribute_node is not None:
                if attribute_node.isCompileTimeConstant() and (not attribute_node.isMutable()):
                    return (bool(attribute_node.getCompileTimeConstant()), attribute_node.makeClone(), 'Using very trusted constant truth value.')
        if not self.mayRaiseException(BaseException) and self.mayRaiseExceptionBool(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (None, None, None)

    def collectVariableAccesses(self, emit_read, emit_write):
        if False:
            return 10
        emit_read(self.variable)

    def hasShapeListExact(self):
        if False:
            while True:
                i = 10
        return self.variable_trace is not None and self.variable_trace.hasShapeListExact()

    def hasShapeDictionaryExact(self):
        if False:
            return 10
        return self.variable_trace is not None and self.variable_trace.hasShapeDictionaryExact()

    def hasShapeStrExact(self):
        if False:
            while True:
                i = 10
        return self.variable_trace is not None and self.variable_trace.hasShapeStrExact()

    def hasShapeUnicodeExact(self):
        if False:
            while True:
                i = 10
        return self.variable_trace is not None and self.variable_trace.hasShapeUnicodeExact()

    def hasShapeBoolExact(self):
        if False:
            i = 10
            return i + 15
        return self.variable_trace is not None and self.variable_trace.hasShapeBoolExact()

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            while True:
                i = 10
        return None

    def mayHaveSideEffects(self):
        if False:
            return 10
        return self.variable_trace is None or not self.variable_trace.mustHaveValue()

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_trace is None or not self.variable_trace.mustHaveValue()

    def mayRaiseExceptionBool(self, exception_type):
        if False:
            return 10
        return self.variable_trace is None or not self.variable_trace.mustHaveValue() or (not self.variable_trace.getTypeShape().hasShapeSlotBool())

    def getFunctionsLocalsScope(self):
        if False:
            while True:
                i = 10
        return self.getParentVariableProvider().getLocalsScope()

class ExpressionVariableOrBuiltinRef(ExpressionVariableRef):
    kind = 'EXPRESSION_VARIABLE_OR_BUILTIN_REF'
    __slots__ = ('locals_scope',)

    def __init__(self, variable, locals_scope, source_ref):
        if False:
            while True:
                i = 10
        ExpressionVariableRef.__init__(self, variable=variable, source_ref=source_ref)
        self.locals_scope = locals_scope

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'variable': self.variable, 'locals_scope': self.locals_scope}

    def getFunctionsLocalsScope(self):
        if False:
            while True:
                i = 10
        return self.locals_scope

def makeExpressionVariableRef(variable, locals_scope, source_ref):
    if False:
        i = 10
        return i + 15
    if variable.getName() in _hard_names:
        return ExpressionVariableOrBuiltinRef(variable=variable, locals_scope=locals_scope, source_ref=source_ref)
    else:
        return ExpressionVariableRef(variable=variable, source_ref=source_ref)

class ExpressionTempVariableRef(ExpressionNoSideEffectsMixin, ExpressionVariableRefBase):
    kind = 'EXPRESSION_TEMP_VARIABLE_REF'

    def __init__(self, variable, source_ref):
        if False:
            for i in range(10):
                print('nop')
        assert variable.isTempVariable()
        ExpressionVariableRefBase.__init__(self, variable=variable, source_ref=source_ref)

    def getDetailsForDisplay(self):
        if False:
            print('Hello World!')
        return {'temp_name': self.variable.getName(), 'owner': self.variable.getOwner().getCodeName()}

    def getDetails(self):
        if False:
            while True:
                i = 10
        return {'variable': self.variable}

    @staticmethod
    def isExpressionTempVariableRef():
        if False:
            return 10
        return True

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            i = 10
            return i + 15
        assert cls is ExpressionTempVariableRef, cls
        owner = getOwnerFromCodeName(args['owner'])
        variable = owner.getTempVariable(None, args['temp_name'])
        return cls(variable=variable, source_ref=source_ref)

    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        self.variable_trace = trace_collection.getVariableCurrentTrace(variable=self.variable)
        replacement = self.variable_trace.getReplacementNode(self)
        if replacement is not None:
            return self._applyReplacement(trace_collection, replacement)
        self.variable_trace.addUsage()
        return (self, None, None)

    def _makeIterationNextReplacementNode(self, trace_collection, next_node, iterator_assign_node):
        if False:
            return 10
        from .OperatorNodes import makeExpressionOperationBinaryInplace
        from .VariableAssignNodes import makeStatementAssignmentVariable
        provider = trace_collection.getOwner()
        outline_body = ExpressionOutlineBody(provider=provider, name='next_value_accessor', source_ref=self.source_ref)
        if next_node.isExpressionSpecialUnpack():
            source = ExpressionSubscriptLookupForUnpack(expression=ExpressionTempVariableRef(variable=iterator_assign_node.tmp_iterated_variable, source_ref=self.source_ref), subscript=ExpressionTempVariableRef(variable=iterator_assign_node.tmp_iteration_count_variable, source_ref=self.source_ref), expected=next_node.getExpected(), source_ref=self.source_ref)
        else:
            source = ExpressionSubscriptLookupForUnpack(expression=ExpressionTempVariableRef(variable=iterator_assign_node.tmp_iterated_variable, source_ref=self.source_ref), subscript=ExpressionTempVariableRef(variable=iterator_assign_node.tmp_iteration_count_variable, source_ref=self.source_ref), expected=None, source_ref=self.source_ref)
        statements = (makeStatementAssignmentVariable(variable=iterator_assign_node.tmp_iteration_next_variable, source=source, source_ref=self.source_ref), makeStatementAssignmentVariable(variable=iterator_assign_node.tmp_iteration_count_variable, source=makeExpressionOperationBinaryInplace(left=ExpressionTempVariableRef(variable=iterator_assign_node.tmp_iteration_count_variable, source_ref=self.source_ref), right=makeConstantRefNode(constant=1, source_ref=self.source_ref), operator='IAdd', source_ref=self.source_ref), source_ref=self.source_ref), makeStatementReturn(expression=ExpressionTempVariableRef(variable=iterator_assign_node.tmp_iteration_next_variable, source_ref=self.source_ref), source_ref=self.source_ref))
        outline_body.setChildBody(makeStatementsSequenceFromStatements(*statements))
        return (False, trace_collection.computedExpressionResultRaw(outline_body, change_tags='new_expression', change_desc=lambda : "Iterator 'next' converted to %s." % iterator_assign_node.getIterationIndexDesc()))

    def computeExpressionNext1(self, next_node, trace_collection):
        if False:
            i = 10
            return i + 15
        iteration_source_node = self.variable_trace.getIterationSourceNode()
        if iteration_source_node is not None:
            if iteration_source_node.parent.isStatementAssignmentVariableIterator():
                iterator_assign_node = iteration_source_node.parent
                if iterator_assign_node.tmp_iterated_variable is not None:
                    return self._makeIterationNextReplacementNode(trace_collection=trace_collection, next_node=next_node, iterator_assign_node=iterator_assign_node)
            iteration_source_node.onContentIteratedEscapes(trace_collection)
            if iteration_source_node.mayHaveSideEffectsNext():
                trace_collection.onControlFlowEscape(self)
        else:
            self.onContentEscapes(trace_collection)
            if self.mayHaveSideEffectsNext():
                trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (True, (next_node, None, None))

    def mayRaiseExceptionImportName(self, exception_type, import_name):
        if False:
            i = 10
            return i + 15
        if self.variable_trace is not None and self.variable_trace.isAssignTrace():
            return self.variable_trace.getAssignNode().subnode_source.mayRaiseExceptionImportName(exception_type, import_name)
        else:
            return True

    @staticmethod
    def isKnownToBeIterableAtMin(count):
        if False:
            return 10
        return None