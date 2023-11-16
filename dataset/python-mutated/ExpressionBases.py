""" Expression base classes.

These classes provide the generic base classes available for
expressions. They have a richer interface, mostly related to
abstract execution, and different from statements.

"""
from abc import abstractmethod
from nuitka import Options
from nuitka.__past__ import long
from nuitka.code_generation.Reports import onMissingOverload
from nuitka.Constants import isCompileTimeConstantValue
from nuitka.PythonVersions import python_version
from .ChildrenHavingMixins import ChildHavingValueMixin
from .NodeBases import NodeBase
from .NodeMakingHelpers import makeConstantReplacementNode, makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue, wrapExpressionWithNodeSideEffects
from .shapes.BuiltinTypeShapes import tshape_bool, tshape_bytes, tshape_dict, tshape_list, tshape_str, tshape_type, tshape_unicode
from .shapes.StandardShapes import tshape_unknown

class ExpressionBase(NodeBase):
    __slots__ = ('code_generated',)

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_unknown

    def getValueShape(self):
        if False:
            return 10
        return self

    @staticmethod
    def isCompileTimeConstant():
        if False:
            i = 10
            return i + 15
        'Has a value that we can use at compile time.\n\n        Yes or no. If it has such a value, simulations can be applied at\n        compile time and e.g. operations or conditions, or even calls may\n        be executed against it.\n        '
        return False

    @staticmethod
    def getTruthValue():
        if False:
            i = 10
            return i + 15
        'Return known truth value. The "None" value indicates unknown.'
        return None

    @staticmethod
    def getComparisonValue():
        if False:
            while True:
                i = 10
        'Return known value used for compile time comparison. The "None" value indicates unknown.'
        return (False, None)

    @staticmethod
    def isMappingWithConstantStringKeys():
        if False:
            for i in range(10):
                print('nop')
        'Is this a mapping with constant string keys. Used for call optimization.'
        return False

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            while True:
                i = 10
        'Can be iterated at all (count is None) or exactly count times.\n\n        Yes or no. If it can be iterated a known number of times, it may\n        be asked to unpack itself.\n        '
        return False

    @staticmethod
    def isKnownToBeIterableAtMin(count):
        if False:
            return 10
        return False

    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        'Value that "len" or "PyObject_Size" would give, if known.\n\n        Otherwise it is "None" to indicate unknown.\n        '
        return None

    def getIterationMinLength(self):
        if False:
            for i in range(10):
                print('nop')
        'Value that "len" or "PyObject_Size" would give at minimum, if known.\n\n        Otherwise it is "None" to indicate unknown.\n        '
        return self.getIterationLength()

    @staticmethod
    def getStringValue():
        if False:
            i = 10
            return i + 15
        'Node as string value, if possible.'
        return None

    def getStrValue(self):
        if False:
            return 10
        'Value that "str" or "PyObject_Str" would give, if known.\n\n        Otherwise it is "None" to indicate unknown. Users must not\n        forget to take side effects into account, when replacing a\n        node with its string value.\n        '
        string_value = self.getStringValue()
        if string_value is not None:
            return makeConstantReplacementNode(node=self, constant=string_value, user_provided=False)
        return None

    def getTypeValue(self):
        if False:
            return 10
        'Type of the node.'
        from .TypeNodes import ExpressionBuiltinType1
        return ExpressionBuiltinType1(value=self.makeClone(), source_ref=self.source_ref)

    def getIterationHandle(self):
        if False:
            return 10
        return None

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        'Is the value hashable, i.e. suitable for dictionary/set key usage.'
        return None

    @staticmethod
    def extractUnhashableNodeType():
        if False:
            for i in range(10):
                print('nop')
        'Return the value type that is not hashable, if isKnowtoBeHashable() returns False.'
        return None

    def onRelease(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        pass

    def isKnownToHaveAttribute(self, attribute_name):
        if False:
            return 10
        return None

    @abstractmethod
    def computeExpressionRaw(self, trace_collection):
        if False:
            i = 10
            return i + 15
        'Abstract execution of the node.\n\n        Returns:\n            tuple(node, tags, description)\n\n            The return value can be node itself.\n\n        Notes:\n            Replaces a node with computation result. This is the low level\n            form for the few cases, where the children are not simply all\n            evaluated first, but this allows e.g. to deal with branches, do\n            not overload this unless necessary.\n        '

    def computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection):
        if False:
            print('Hello World!')
        if self.mayRaiseExceptionAttributeLookup(BaseException, attribute_name):
            trace_collection.onExceptionRaiseExit(BaseException)
        trace_collection.onControlFlowEscape(self)
        return (lookup_node, None, None)

    def computeExpressionAttributeSpecial(self, lookup_node, attribute_name, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, None, None)

    def computeExpressionImportName(self, import_node, import_name, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if self.mayRaiseExceptionImportName(BaseException, import_name):
            trace_collection.onExceptionRaiseExit(BaseException)
        trace_collection.onControlFlowEscape(self)
        return (import_node, None, None)

    def computeExpressionSetAttribute(self, set_node, attribute_name, value_node, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.removeKnowledge(self)
        trace_collection.removeKnowledge(value_node)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (set_node, None, None)

    def computeExpressionDelAttribute(self, set_node, attribute_name, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (set_node, None, None)

    def computeExpressionSubscript(self, lookup_node, subscript, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, None, None)

    def computeExpressionSetSubscript(self, set_node, subscript, value_node, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.removeKnowledge(value_node)
        trace_collection.removeKnowledge(subscript)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (set_node, None, None)

    def computeExpressionDelSubscript(self, del_node, subscript, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (del_node, None, None)

    def computeExpressionSlice(self, lookup_node, lower, upper, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, None, None)

    def computeExpressionSetSlice(self, set_node, lower, upper, value_node, trace_collection):
        if False:
            return 10
        trace_collection.removeKnowledge(value_node)
        trace_collection.removeKnowledge(self)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (set_node, None, None)

    def computeExpressionDelSlice(self, set_node, lower, upper, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.removeKnowledge(self)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (set_node, None, None)

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            while True:
                i = 10
        call_node.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (call_node, None, None)

    def computeExpressionCallViaVariable(self, call_node, variable_ref_node, call_args, call_kw, trace_collection):
        if False:
            print('Hello World!')
        self.onContentEscapes(trace_collection)
        if call_args is not None:
            call_args.onContentEscapes(trace_collection)
        if call_kw is not None:
            call_kw.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (call_node, None, None)

    def computeExpressionLen(self, len_node, trace_collection):
        if False:
            print('Hello World!')
        shape = self.getValueShape()
        has_len = shape.hasShapeSlotLen()
        if has_len is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="object of type '%s' has no len()", operation='len', original_node=len_node, value_node=self)
        elif has_len is True:
            iter_length = self.getIterationLength()
            if iter_length is not None:
                from .ConstantRefNodes import makeConstantRefNode
                result = makeConstantRefNode(constant=int(iter_length), source_ref=len_node.getSourceReference())
                result = wrapExpressionWithNodeSideEffects(new_node=result, old_node=self)
                return (result, 'new_constant', "Predicted 'len' result from value shape.")
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (len_node, None, None)

    def computeExpressionAbs(self, abs_node, trace_collection):
        if False:
            while True:
                i = 10
        shape = self.getTypeShape()
        if shape.hasShapeSlotAbs() is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="bad operand type for abs(): '%s'", operation='abs', original_node=abs_node, value_node=self)
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (abs_node, None, None)

    def computeExpressionInt(self, int_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        shape = self.getTypeShape()
        if shape.hasShapeSlotInt() is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="int() argument must be a string or a number, not '%s'" if python_version < 768 else "int() argument must be a string, a bytes-like object or a number, not '%s'", operation='int', original_node=int_node, value_node=self)
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (int_node, None, None)

    def computeExpressionLong(self, long_node, trace_collection):
        if False:
            i = 10
            return i + 15
        shape = self.getTypeShape()
        if shape.hasShapeSlotLong() is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="long() argument must be a string or a number, not '%s'", operation='long', original_node=long_node, value_node=self)
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (long_node, None, None)

    def computeExpressionFloat(self, float_node, trace_collection):
        if False:
            print('Hello World!')
        shape = self.getTypeShape()
        if shape.hasShapeSlotFloat() is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue('float() argument must be a string or a number' if Options.is_full_compat and python_version < 768 else "float() argument must be a string or a number, not '%s'", operation='long', original_node=float_node, value_node=self)
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (float_node, None, None)

    def computeExpressionBytes(self, bytes_node, trace_collection):
        if False:
            i = 10
            return i + 15
        shape = self.getTypeShape()
        if shape.hasShapeSlotBytes() is False and shape.hasShapeSlotInt() is False and (shape.hasShapeSlotIter() is False):
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue("'%s' object is not iterable", operation='bytes', original_node=bytes_node, value_node=self)
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (bytes_node, None, None)

    def computeExpressionComplex(self, complex_node, trace_collection):
        if False:
            i = 10
            return i + 15
        shape = self.getTypeShape()
        if shape.hasShapeSlotComplex() is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue('complex() argument must be a string or a number' if Options.is_full_compat and python_version < 768 else "complex() argument must be a string or a number, not '%s'", operation='complex', original_node=complex_node, value_node=self)
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (complex_node, None, None)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        shape = self.getTypeShape()
        if shape.hasShapeSlotIter() is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="'%s' object is not iterable", operation='iter', original_node=iter_node, value_node=self)
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (iter_node, None, None)

    def computeExpressionNext1(self, next_node, trace_collection):
        if False:
            i = 10
            return i + 15
        self.onContentEscapes(trace_collection)
        if self.mayHaveSideEffectsNext():
            trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (True, (next_node, None, None))

    def computeExpressionAsyncIter(self, iter_node, trace_collection):
        if False:
            return 10
        self.onContentEscapes(trace_collection)
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (iter_node, None, None)

    def computeExpressionOperationNot(self, not_node, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onControlFlowEscape(not_node)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (not_node, None, None)

    def computeExpressionOperationRepr(self, repr_node, trace_collection):
        if False:
            while True:
                i = 10
        type_shape = self.getTypeShape()
        escape_desc = type_shape.getOperationUnaryReprEscape()
        exception_raise_exit = escape_desc.getExceptionExit()
        if exception_raise_exit is not None:
            trace_collection.onExceptionRaiseExit(exception_raise_exit)
        if escape_desc.isValueEscaping():
            trace_collection.removeKnowledge(self)
        if escape_desc.isControlFlowEscape():
            trace_collection.onControlFlowEscape(self)
        return ((repr_node, None, None), escape_desc)

    def computeExpressionComparisonIn(self, in_node, value_node, trace_collection):
        if False:
            while True:
                i = 10
        shape = self.getTypeShape()
        assert shape is not None, self
        if shape.hasShapeSlotContains() is False:
            trace_collection.onExceptionRaiseExit(BaseException)
            return makeRaiseTypeErrorExceptionReplacementFromTemplateAndValue(template="argument of type '%s' object is not iterable", operation='in', original_node=in_node, value_node=self)
        trace_collection.onControlFlowEscape(in_node)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (in_node, None, None)

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            return 10
        if not self.mayHaveSideEffects():
            return (None, 'new_statements', lambda : 'Removed %s without effect.' % self.getDescription())
        return (statement, None, None)

    def computeExpressionBool(self, trace_collection):
        if False:
            i = 10
            return i + 15
        if not self.mayRaiseException(BaseException) and self.mayRaiseExceptionBool(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (None, None, None)

    @staticmethod
    def onContentEscapes(trace_collection):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    def onContentIteratedEscapes(trace_collection):
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    def mayRaiseExceptionBool(exception_type):
        if False:
            for i in range(10):
                print('nop')
        'Unless we are told otherwise, everything may raise being checked.'
        return True

    @staticmethod
    def mayRaiseExceptionAbs(exception_type):
        if False:
            for i in range(10):
                print('nop')
        "Unless we are told otherwise, everything may raise in 'abs'."
        return True

    @staticmethod
    def mayRaiseExceptionInt(exception_type):
        if False:
            return 10
        'Unless we are told otherwise, everything may raise in __int__.'
        return True

    @staticmethod
    def mayRaiseExceptionLong(exception_type):
        if False:
            print('Hello World!')
        'Unless we are told otherwise, everything may raise in __long__.'
        return True

    @staticmethod
    def mayRaiseExceptionFloat(exception_type):
        if False:
            print('Hello World!')
        'Unless we are told otherwise, everything may raise in __float__.'
        return True

    @staticmethod
    def mayRaiseExceptionBytes(exception_type):
        if False:
            print('Hello World!')
        'Unless we are told otherwise, everything may raise in __bytes__.'
        return True

    @staticmethod
    def mayRaiseExceptionIn(exception_type, checked_value):
        if False:
            i = 10
            return i + 15
        'Unless we are told otherwise, everything may raise being iterated.'
        return True

    @staticmethod
    def mayRaiseExceptionAttributeLookup(exception_type, attribute_name):
        if False:
            i = 10
            return i + 15
        'Unless we are told otherwise, everything may raise for attribute access.'
        return True

    @staticmethod
    def mayRaiseExceptionAttributeLookupSpecial(exception_type, attribute_name):
        if False:
            i = 10
            return i + 15
        'Unless we are told otherwise, everything may raise for attribute access.'
        return True

    @staticmethod
    def mayRaiseExceptionAttributeLookupObject(exception_type, attribute):
        if False:
            i = 10
            return i + 15
        'Unless we are told otherwise, everything may raise for attribute access.'
        return True

    @staticmethod
    def mayRaiseExceptionImportName(exception_type, import_name):
        if False:
            i = 10
            return i + 15
        'Unless we are told otherwise, everything may raise for name import.'
        return True

    @staticmethod
    def mayHaveSideEffectsBool():
        if False:
            while True:
                i = 10
        'Unless we are told otherwise, everything may have a side effect for bool check.'
        return True

    @staticmethod
    def mayHaveSideEffectsAbs():
        if False:
            for i in range(10):
                print('nop')
        'Unless we are told otherwise, everything may have a side effect for abs check.'
        return True

    def mayHaveSideEffectsNext(self):
        if False:
            i = 10
            return i + 15
        'The type shape tells us, if "next" may execute code.'
        return self.getTypeShape().hasShapeSlotNextCode()

    def hasShapeSlotLen(self):
        if False:
            i = 10
            return i + 15
        'The type shape tells us, if "len" is available.'
        return self.getTypeShape().hasShapeSlotLen()

    def hasShapeSlotIter(self):
        if False:
            return 10
        'The type shape tells us, if "iter" is available.'
        return self.getTypeShape().hasShapeSlotIter()

    def hasShapeSlotNext(self):
        if False:
            while True:
                i = 10
        'The type shape tells us, if "next" is available.'
        return self.getTypeShape().hasShapeSlotNext()

    @staticmethod
    def isIndexable():
        if False:
            return 10
        "Unless we are told otherwise, it's not indexable."
        return False

    @staticmethod
    def getIntegerValue():
        if False:
            return 10
        'Node as integer value, if possible.'
        return None

    @staticmethod
    def getIndexValue():
        if False:
            print('Hello World!')
        'Node as index value, if possible.\n\n        This should only work for int, bool, and long values, but e.g. not floats.\n        '
        return None

    @staticmethod
    def getIntValue():
        if False:
            return 10
        'Value that "int" or "PyNumber_Int" (sp) would give, if known.\n\n        Otherwise it is "None" to indicate unknown. Users must not\n        forget to take side effects into account, when replacing a\n        node with its string value.\n        '
        return None

    def getExpressionDictInConstant(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Value that the dict "in" operation would give, if known.\n\n        This is only called for values with known dict type shape. And those\n        nodes who are known to do it, have to overload it.\n        '
        if Options.is_debug:
            onMissingOverload(method_name='getExpressionDictInConstant', node=self)
        return None

    def hasShapeTrustedAttributes(self):
        if False:
            print('Hello World!')
        return self.getTypeShape().hasShapeTrustedAttributes()

    def hasShapeTypeExact(self):
        if False:
            return 10
        "Does a node have exactly a 'type' shape."
        return self.getTypeShape() is tshape_type

    def hasShapeListExact(self):
        if False:
            i = 10
            return i + 15
        'Does a node have exactly a list shape.'
        return self.getTypeShape() is tshape_list

    def hasShapeDictionaryExact(self):
        if False:
            return 10
        'Does a node have exactly a dictionary shape.'
        return self.getTypeShape() is tshape_dict

    def hasShapeStrExact(self):
        if False:
            return 10
        'Does an expression have exactly a string shape.'
        return self.getTypeShape() is tshape_str

    def hasShapeUnicodeExact(self):
        if False:
            for i in range(10):
                print('nop')
        'Does an expression have exactly a unicode shape.'
        return self.getTypeShape() is tshape_unicode
    if str is bytes:

        def hasShapeStrOrUnicodeExact(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.getTypeShape() in (tshape_str, tshape_unicode)
    else:

        def hasShapeStrOrUnicodeExact(self):
            if False:
                return 10
            return self.getTypeShape() is tshape_str

    def hasShapeBytesExact(self):
        if False:
            for i in range(10):
                print('nop')
        'Does an expression have exactly a bytes shape.'
        return self.getTypeShape() is tshape_bytes

    def hasShapeBoolExact(self):
        if False:
            print('Hello World!')
        'Does an expression have exactly a bool shape.'
        return self.getTypeShape() is tshape_bool

    @staticmethod
    def hasVeryTrustedValue():
        if False:
            print('Hello World!')
        'Trust that value will not be overwritten from the outside.'
        return False

class ExpressionNoSideEffectsMixin(object):
    __slots__ = ()

    @staticmethod
    def mayHaveSideEffects():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def extractSideEffects():
        if False:
            i = 10
            return i + 15
        return ()

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            print('Hello World!')
        return (None, 'new_statements', lambda : 'Removed %s that never has an effect.' % self.getDescription())

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return False

class CompileTimeConstantExpressionBase(ExpressionNoSideEffectsMixin, ExpressionBase):
    __slots__ = ('computed_attribute',)

    def __init__(self, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionBase.__init__(self, source_ref)
        self.computed_attribute = None

    @staticmethod
    def isCompileTimeConstant():
        if False:
            for i in range(10):
                print('nop')
        'Has a value that we can use at compile time.\n\n        Yes or no. If it has such a value, simulations can be applied at\n        compile time and e.g. operations or conditions, or even calls may\n        be executed against it.\n        '
        return True

    def getTruthValue(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.getCompileTimeConstant())

    def getComparisonValue(self):
        if False:
            i = 10
            return i + 15
        return (True, self.getCompileTimeConstant())

    @abstractmethod
    def getCompileTimeConstant(self):
        if False:
            while True:
                i = 10
        'Return compile time constant.\n\n        Notes: Only available after passing "isCompileTimeConstant()".\n\n        '

    @staticmethod
    def isMutable():
        if False:
            for i in range(10):
                print('nop')
        'Return if compile time constant is mutable.\n\n        Notes: Only useful after passing "isCompileTimeConstant()".\n        '
        return False

    @staticmethod
    def hasShapeTrustedAttributes():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def mayHaveSideEffectsBool():
        if False:
            return 10
        return False

    @staticmethod
    def mayRaiseExceptionBool(exception_type):
        if False:
            return 10
        return False

    def mayRaiseExceptionAttributeLookup(self, exception_type, attribute_name):
        if False:
            return 10
        return not self.computed_attribute

    def mayRaiseExceptionAttributeLookupSpecial(self, exception_type, attribute_name):
        if False:
            while True:
                i = 10
        return not self.computed_attribute

    def computeExpressionOperationNot(self, not_node, trace_collection):
        if False:
            return 10
        return trace_collection.getCompileTimeComputationResult(node=not_node, computation=lambda : not self.getCompileTimeConstant(), description='Compile time constant negation truth value pre-computed.')

    def computeExpressionOperationRepr(self, repr_node, trace_collection):
        if False:
            i = 10
            return i + 15
        return (trace_collection.getCompileTimeComputationResult(node=repr_node, computation=lambda : repr(self.getCompileTimeConstant()), description='Compile time constant repr value pre-computed.'), None)

    def computeExpressionLen(self, len_node, trace_collection):
        if False:
            i = 10
            return i + 15
        return trace_collection.getCompileTimeComputationResult(node=len_node, computation=lambda : len(self.getCompileTimeConstant()), description='Compile time constant len value pre-computed.')

    def computeExpressionAbs(self, abs_node, trace_collection):
        if False:
            return 10
        return trace_collection.getCompileTimeComputationResult(node=abs_node, computation=lambda : abs(self.getCompileTimeConstant()), description='Compile time constant abs value pre-computed.')

    def computeExpressionInt(self, int_node, trace_collection):
        if False:
            while True:
                i = 10
        return trace_collection.getCompileTimeComputationResult(node=int_node, computation=lambda : int(self.getCompileTimeConstant()), description='Compile time constant int value pre-computed.')

    def computeExpressionLong(self, long_node, trace_collection):
        if False:
            print('Hello World!')
        return trace_collection.getCompileTimeComputationResult(node=long_node, computation=lambda : long(self.getCompileTimeConstant()), description='Compile time constant long value pre-computed.')

    def computeExpressionFloat(self, float_node, trace_collection):
        if False:
            print('Hello World!')
        return trace_collection.getCompileTimeComputationResult(node=float_node, computation=lambda : float(self.getCompileTimeConstant()), description='Compile time constant float value pre-computed.')

    def computeExpressionBytes(self, bytes_node, trace_collection):
        if False:
            print('Hello World!')
        constant_value = self.getCompileTimeConstant()
        if type(constant_value) in (int, long):
            if constant_value > 1000:
                return (bytes_node, None, None)
        return trace_collection.getCompileTimeComputationResult(node=bytes_node, computation=lambda : bytes(constant_value), description='Compile time constant bytes value pre-computed.')

    def isKnownToHaveAttribute(self, attribute_name):
        if False:
            for i in range(10):
                print('nop')
        if self.computed_attribute is None:
            self.computed_attribute = hasattr(self.getCompileTimeConstant(), attribute_name)
        return self.computed_attribute

    def getKnownAttributeValue(self, attribute_name):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self.getCompileTimeConstant(), attribute_name)

    def computeExpressionAttribute(self, lookup_node, attribute_name, trace_collection):
        if False:
            print('Hello World!')
        value = self.getCompileTimeConstant()
        if self.computed_attribute is None:
            self.computed_attribute = hasattr(value, attribute_name)
        if not self.computed_attribute or isCompileTimeConstantValue(getattr(value, attribute_name, None)):
            return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : getattr(value, attribute_name), description="Attribute '%s' pre-computed." % attribute_name)
        return (lookup_node, None, None)

    def computeExpressionSubscript(self, lookup_node, subscript, trace_collection):
        if False:
            print('Hello World!')
        if subscript.isCompileTimeConstant():
            return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : self.getCompileTimeConstant()[subscript.getCompileTimeConstant()], description='Subscript of constant with constant value.')
        trace_collection.onControlFlowEscape(self)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, None, None)

    def computeExpressionSlice(self, lookup_node, lower, upper, trace_collection):
        if False:
            i = 10
            return i + 15
        if lower is not None:
            if upper is not None:
                if lower.isCompileTimeConstant() and upper.isCompileTimeConstant():
                    return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : self.getCompileTimeConstant()[lower.getCompileTimeConstant():upper.getCompileTimeConstant()], description='Slicing of constant with constant indexes.', user_provided=False)
            elif lower.isCompileTimeConstant():
                return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : self.getCompileTimeConstant()[lower.getCompileTimeConstant():], description='Slicing of constant with constant lower index only.', user_provided=False)
        elif upper is not None:
            if upper.isCompileTimeConstant():
                return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : self.getCompileTimeConstant()[:upper.getCompileTimeConstant()], description='Slicing of constant with constant upper index only.', user_provided=False)
        else:
            return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : self.getCompileTimeConstant()[:], description='Slicing of constant with no indexes.', user_provided=False)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (lookup_node, None, None)

    def computeExpressionComparisonIn(self, in_node, value_node, trace_collection):
        if False:
            i = 10
            return i + 15
        if value_node.isCompileTimeConstant():
            return trace_collection.getCompileTimeComputationResult(node=in_node, computation=lambda : in_node.getSimulator()(value_node.getCompileTimeConstant(), self.getCompileTimeConstant()), description="Predicted '%s' on compiled time constant values." % in_node.comparator, user_provided=False)
        trace_collection.onExceptionRaiseExit(BaseException)
        return (in_node, None, None)

    def computeExpressionBool(self, trace_collection):
        if False:
            while True:
                i = 10
        constant = self.getCompileTimeConstant()
        assert type(constant) is not bool
        truth_value = bool(constant)
        result = makeConstantReplacementNode(constant=truth_value, node=self, user_provided=False)
        return (truth_value, result, 'Predicted compile time constant truth value.')

class ExpressionSpecBasedComputationMixin(object):
    __slots__ = ()
    builtin_spec = None

    def computeBuiltinSpec(self, trace_collection, given_values):
        if False:
            for i in range(10):
                print('nop')
        assert self.builtin_spec is not None, self
        if not self.builtin_spec.isCompileTimeComputable(given_values):
            trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : self.builtin_spec.simulateCall(given_values), description="Built-in call to '%s' pre-computed." % self.builtin_spec.getName(), user_provided=self.builtin_spec.isUserProvided(given_values))

class ExpressionSpecBasedComputationNoRaiseMixin(object):
    __slots__ = ()
    builtin_spec = None

    def computeBuiltinSpec(self, trace_collection, given_values):
        if False:
            while True:
                i = 10
        assert self.builtin_spec is not None, self
        if not self.builtin_spec.isCompileTimeComputable(given_values):
            return (self, None, None)
        return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : self.builtin_spec.simulateCall(given_values), description="Built-in call to '%s' pre-computed." % self.builtin_spec.getName())

class ExpressionBuiltinSingleArgBase(ExpressionSpecBasedComputationMixin, ChildHavingValueMixin, ExpressionBase):
    named_children = ('value',)

    def __init__(self, value, source_ref):
        if False:
            i = 10
            return i + 15
        ChildHavingValueMixin.__init__(self, value=value)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        value = self.subnode_value
        assert value is not None
        if value is None:
            return self.computeBuiltinSpec(trace_collection=trace_collection, given_values=())
        else:
            return self.computeBuiltinSpec(trace_collection=trace_collection, given_values=(value,))