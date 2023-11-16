""" Node for constant expressions. Can be all common built-in types.

"""
import sys
from abc import abstractmethod
from nuitka import Options
from nuitka.__past__ import GenericAlias, UnionType, iterItems, long, unicode, xrange
from nuitka.Builtins import builtin_anon_values, builtin_exception_values_list, builtin_named_values
from nuitka.Constants import getUnhashableConstant, isConstant, isHashable, isMutable, the_empty_dict, the_empty_frozenset, the_empty_list, the_empty_set, the_empty_tuple, the_empty_unicode
from nuitka.PythonVersions import python_version
from nuitka.Tracing import optimization_logger
from .ExpressionBases import CompileTimeConstantExpressionBase
from .ExpressionShapeMixins import ExpressionBoolShapeExactMixin, ExpressionBytearrayShapeExactMixin, ExpressionBytesShapeExactMixin, ExpressionComplexShapeExactMixin, ExpressionDictShapeExactMixin, ExpressionEllipsisShapeExactMixin, ExpressionFloatShapeExactMixin, ExpressionFrozensetShapeExactMixin, ExpressionIntShapeExactMixin, ExpressionListShapeExactMixin, ExpressionLongShapeExactMixin, ExpressionNoneShapeExactMixin, ExpressionSetShapeExactMixin, ExpressionSliceShapeExactMixin, ExpressionStrShapeExactMixin, ExpressionTupleShapeExactMixin, ExpressionUnicodeShapeExactMixin
from .IterationHandles import ConstantBytearrayIterationHandle, ConstantBytesIterationHandle, ConstantDictIterationHandle, ConstantFrozensetIterationHandle, ConstantListIterationHandle, ConstantRangeIterationHandle, ConstantSetIterationHandle, ConstantStrIterationHandle, ConstantTupleIterationHandle, ConstantUnicodeIterationHandle
from .NodeMakingHelpers import makeRaiseExceptionReplacementExpression, wrapExpressionWithSideEffects
from .shapes.BuiltinTypeShapes import tshape_namedtuple, tshape_type, tshape_xrange

class ExpressionConstantUntrackedRefBase(CompileTimeConstantExpressionBase):
    __slots__ = ('constant',)

    def __init__(self, constant, source_ref):
        if False:
            return 10
        CompileTimeConstantExpressionBase.__init__(self, source_ref)
        self.constant = constant

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent
        del self.constant

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Node %s value %r at %s>' % (self.kind, self.constant, self.source_ref.getAsString())

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'constant': self.constant}

    def getDetailsForDisplay(self):
        if False:
            print('Hello World!')
        result = self.getDetails()
        if 'constant' in result:
            result['constant'] = repr(result['constant'])
        return result

    @staticmethod
    def isExpressionConstantRef():
        if False:
            while True:
                i = 10
        return True

    def computeExpressionRaw(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

    def computeExpression(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(TypeError)
        new_node = wrapExpressionWithSideEffects(new_node=makeRaiseExceptionReplacementExpression(expression=self, exception_type='TypeError', exception_value="'%s' object is not callable" % type(self.constant).__name__), old_node=call_node, side_effects=call_node.extractSideEffectsPreCall())
        return (new_node, 'new_raise', 'Predicted call of constant %s value to exception raise.' % type(self.constant))

    def computeExpressionCallViaVariable(self, call_node, variable_ref_node, call_args, call_kw, trace_collection):
        if False:
            return 10
        return self.computeExpressionCall(call_node=call_node, call_args=call_args, call_kw=call_kw, trace_collection=trace_collection)

    def getCompileTimeConstant(self):
        if False:
            i = 10
            return i + 15
        return self.constant

    def getComparisonValue(self):
        if False:
            while True:
                i = 10
        return (True, self.constant)

    @staticmethod
    def getIterationHandle():
        if False:
            print('Hello World!')
        return None

    def isMutable(self):
        if False:
            i = 10
            return i + 15
        assert False, self

    def isKnownToBeHashable(self):
        if False:
            print('Hello World!')
        assert False, self

    def extractUnhashableNodeType(self):
        if False:
            return 10
        value = getUnhashableConstant(self.constant)
        if value is not None:
            return makeConstantRefNode(constant=type(value), source_ref=self.source_ref)

    @staticmethod
    def isNumberConstant():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isIndexConstant():
        if False:
            while True:
                i = 10
        return False

    def isIndexable(self):
        if False:
            while True:
                i = 10
        return self.constant is None or self.isNumberConstant()

    def isKnownToBeIterable(self, count):
        if False:
            print('Hello World!')
        if self.isIterableConstant():
            return count is None or len(self.constant) == count
        else:
            return False

    def isKnownToBeIterableAtMin(self, count):
        if False:
            i = 10
            return i + 15
        length = self.getIterationLength()
        return length is not None and length >= count

    def canPredictIterationValues(self):
        if False:
            return 10
        return self.isKnownToBeIterable(None)

    def getIterationValue(self, count):
        if False:
            while True:
                i = 10
        assert count < len(self.constant)
        return makeConstantRefNode(constant=self.constant[count], source_ref=self.source_ref)

    def getIterationValueRange(self, start, stop):
        if False:
            print('Hello World!')
        return [makeConstantRefNode(constant=value, source_ref=self.source_ref) for value in self.constant[start:stop]]

    def getIterationValues(self):
        if False:
            i = 10
            return i + 15
        source_ref = self.source_ref
        return tuple((makeConstantRefNode(constant=value, source_ref=source_ref, user_provided=self.user_provided) for value in self.constant))

    def getIntegerValue(self):
        if False:
            print('Hello World!')
        if self.isNumberConstant():
            return int(self.constant)
        else:
            return None

    @abstractmethod
    def isIterableConstant(self):
        if False:
            for i in range(10):
                print('nop')
        'Is the constant type iterable.'

    def getIterationLength(self):
        if False:
            print('Hello World!')
        assert not self.isIterableConstant(), self
        return None

    def getStrValue(self):
        if False:
            i = 10
            return i + 15
        return makeConstantRefNode(constant=str(self.constant), user_provided=False, source_ref=self.source_ref)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            print('Hello World!')
        assert not self.isIterableConstant()
        return (iter_node, None, None)

class ExpressionConstantRefBase(ExpressionConstantUntrackedRefBase):
    """Constants reference base class.

    Use this for cases, for which it makes sense to track origin, e.g.
    large lists are from computation or from user literals.
    """
    __slots__ = ('user_provided',)

    def __init__(self, constant, user_provided, source_ref):
        if False:
            while True:
                i = 10
        ExpressionConstantUntrackedRefBase.__init__(self, constant=constant, source_ref=source_ref)
        self.user_provided = user_provided
        if not user_provided and Options.is_debug:
            try:
                if type(constant) in (str, unicode, bytes):
                    max_size = 1000
                elif type(constant) is xrange:
                    max_size = None
                else:
                    max_size = 256
                if max_size is not None and len(constant) > max_size:
                    optimization_logger.warning('Too large constant (%s %d) encountered at %s.' % (type(constant), len(constant), source_ref.getAsString()))
            except TypeError:
                pass

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'constant': self.constant, 'user_provided': self.user_provided}

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Node %s value %r at %s %s>' % (self.kind, self.constant, self.source_ref.getAsString(), self.user_provided)

    def getStrValue(self):
        if False:
            i = 10
            return i + 15
        try:
            return makeConstantRefNode(constant=str(self.constant), user_provided=self.user_provided, source_ref=self.source_ref)
        except UnicodeEncodeError:
            return None

class ExpressionConstantNoneRef(ExpressionNoneShapeExactMixin, ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_NONE_REF'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            return 10
        ExpressionConstantUntrackedRefBase.__init__(self, constant=None, source_ref=source_ref)

    @staticmethod
    def getDetails():
        if False:
            while True:
                i = 10
        return {}

    @staticmethod
    def isMutable():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            print('Hello World!')
        return False

class ExpressionConstantBoolRefBase(ExpressionBoolShapeExactMixin, ExpressionConstantUntrackedRefBase):

    @staticmethod
    def isExpressionConstantBoolRef():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def computeExpressionBool(trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return (None, None, None)

    @staticmethod
    def getDetails():
        if False:
            print('Hello World!')
        return {}

    @staticmethod
    def isMutable():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isNumberConstant():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isIndexConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isIterableConstant():
        if False:
            i = 10
            return i + 15
        return False

class ExpressionConstantTrueRef(ExpressionConstantBoolRefBase):
    kind = 'EXPRESSION_CONSTANT_TRUE_REF'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantBoolRefBase.__init__(self, constant=True, source_ref=source_ref)

    @staticmethod
    def getTruthValue():
        if False:
            while True:
                i = 10
        'Return known truth value.'
        return True

    @staticmethod
    def getIndexValue():
        if False:
            i = 10
            return i + 15
        return 1

class ExpressionConstantFalseRef(ExpressionConstantBoolRefBase):
    kind = 'EXPRESSION_CONSTANT_FALSE_REF'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantBoolRefBase.__init__(self, constant=False, source_ref=source_ref)

    @staticmethod
    def getTruthValue():
        if False:
            print('Hello World!')
        'Return known truth value.'
        return False

    @staticmethod
    def getIndexValue():
        if False:
            for i in range(10):
                print('nop')
        return 0

class ExpressionConstantEllipsisRef(ExpressionEllipsisShapeExactMixin, ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_ELLIPSIS_REF'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantUntrackedRefBase.__init__(self, constant=Ellipsis, source_ref=source_ref)

    @staticmethod
    def getDetails():
        if False:
            while True:
                i = 10
        return {}

    @staticmethod
    def isMutable():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            return 10
        return False

class ExpressionConstantDictRef(ExpressionDictShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_DICT_REF'

    def __init__(self, constant, user_provided, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantDictRef():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isMutable():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            return 10
        return True

    def getIterationHandle(self):
        if False:
            print('Hello World!')
        return ConstantDictIterationHandle(self)

    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        result = makeConstantRefNode(constant=tuple(self.constant), user_provided=self.user_provided, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        self.finalize()
        return (iter_node, 'new_constant', 'Iteration over constant dict lowered to tuple.')

    def isMappingWithConstantStringKeys(self):
        if False:
            print('Hello World!')
        return all((type(key) in (str, unicode) for key in self.constant))

    def getMappingStringKeyPairs(self):
        if False:
            while True:
                i = 10
        pairs = []
        for (key, value) in iterItems(self.constant):
            pairs.append((key, makeConstantRefNode(constant=value, user_provided=self.user_provided, source_ref=self.source_ref)))
        return pairs

    @staticmethod
    def getTruthValue():
        if False:
            for i in range(10):
                print('nop')
        'Return known truth value.\n\n        The empty dict is not allowed here, so we can hardcode it.\n        '
        return True

    def getExpressionDictInConstant(self, value):
        if False:
            i = 10
            return i + 15
        return value in self.constant

class EmptyContainerMixin(object):
    __slots__ = ()

    def getDetails(self):
        if False:
            return 10
        return {'user_provided': self.user_provided}

    @staticmethod
    def getIterationLength():
        if False:
            return 10
        return 0

    @staticmethod
    def getTruthValue():
        if False:
            while True:
                i = 10
        'Return known truth value.\n\n        The empty container is false, so we can hardcode it.\n        '
        return False

class ExpressionConstantDictEmptyRef(EmptyContainerMixin, ExpressionConstantDictRef):
    kind = 'EXPRESSION_CONSTANT_DICT_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            print('Hello World!')
        ExpressionConstantDictRef.__init__(self, constant=the_empty_dict, user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantTupleRef(ExpressionTupleShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_TUPLE_REF'
    __slots__ = ()

    def __init__(self, constant, user_provided, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantTupleRef():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isMutable():
        if False:
            for i in range(10):
                print('nop')
        return False

    def isKnownToBeHashable(self):
        if False:
            while True:
                i = 10
        return isHashable(self.constant)

    @staticmethod
    def isIterableConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getIterationHandle(self):
        if False:
            while True:
                i = 10
        return ConstantTupleIterationHandle(self)

    def getIterationLength(self):
        if False:
            return 10
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return (iter_node, None, None)

    @staticmethod
    def getTruthValue():
        if False:
            return 10
        'Return known truth value.\n\n        The empty dict is not allowed here, so we can hardcode it.\n        '
        return True

class ExpressionConstantTupleMutableRef(ExpressionConstantTupleRef):
    kind = 'EXPRESSION_CONSTANT_TUPLE_MUTABLE_REF'
    __slots__ = ()

    @staticmethod
    def isMutable():
        if False:
            return 10
        return True

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionConstantTupleEmptyRef(EmptyContainerMixin, ExpressionConstantTupleRef):
    kind = 'EXPRESSION_CONSTANT_TUPLE_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            while True:
                i = 10
        ExpressionConstantTupleRef.__init__(self, constant=the_empty_tuple, user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantListRef(ExpressionListShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_LIST_REF'
    __slots__ = ()

    def __init__(self, constant, user_provided, source_ref):
        if False:
            return 10
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantListRef():
        if False:
            return 10
        return True

    @staticmethod
    def isMutable():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isIterableConstant():
        if False:
            i = 10
            return i + 15
        return True

    def getIterationHandle(self):
        if False:
            print('Hello World!')
        return ConstantListIterationHandle(self)

    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            i = 10
            return i + 15
        result = makeConstantRefNode(constant=tuple(self.constant), user_provided=self.user_provided, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        self.finalize()
        return (iter_node, 'new_constant', 'Iteration over constant list lowered to tuple.')

class ExpressionConstantListEmptyRef(EmptyContainerMixin, ExpressionConstantListRef):
    kind = 'EXPRESSION_CONSTANT_LIST_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            return 10
        ExpressionConstantListRef.__init__(self, constant=the_empty_list, user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantSetRef(ExpressionSetShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_SET_REF'
    __slots__ = ()

    def __init__(self, constant, user_provided, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantSetRef():
        if False:
            return 10
        return True

    @staticmethod
    def isMutable():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            i = 10
            return i + 15
        return True

    def getIterationHandle(self):
        if False:
            return 10
        return ConstantSetIterationHandle(self)

    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            i = 10
            return i + 15
        result = makeConstantRefNode(constant=tuple(self.constant), user_provided=self.user_provided, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        self.finalize()
        return (iter_node, 'new_constant', 'Iteration over constant set lowered to tuple.')

class ExpressionConstantSetEmptyRef(EmptyContainerMixin, ExpressionConstantSetRef):
    kind = 'EXPRESSION_CONSTANT_SET_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            while True:
                i = 10
        ExpressionConstantSetRef.__init__(self, constant=the_empty_set, user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantFrozensetRef(ExpressionFrozensetShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_FROZENSET_REF'
    __slots__ = ()

    def __init__(self, constant, user_provided, source_ref):
        if False:
            return 10
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantFrozensetRef():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isMutable():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getIterationHandle(self):
        if False:
            print('Hello World!')
        return ConstantFrozensetIterationHandle(self)

    def getIterationLength(self):
        if False:
            return 10
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        result = makeConstantRefNode(constant=tuple(self.constant), user_provided=self.user_provided, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        self.finalize()
        return (iter_node, 'new_constant', 'Iteration over constant frozenset lowered to tuple.')

class ExpressionConstantFrozensetEmptyRef(EmptyContainerMixin, ExpressionConstantFrozensetRef):
    kind = 'EXPRESSION_CONSTANT_FROZENSET_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            print('Hello World!')
        ExpressionConstantFrozensetRef.__init__(self, constant=the_empty_frozenset, user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantIntRef(ExpressionIntShapeExactMixin, ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_INT_REF'
    __slots__ = ()

    def __init__(self, constant, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantUntrackedRefBase.__init__(self, constant=constant, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantIntRef():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isMutable():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isNumberConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isIndexConstant():
        if False:
            while True:
                i = 10
        return True

    def getIndexValue(self):
        if False:
            i = 10
            return i + 15
        return self.constant

    @staticmethod
    def isIterableConstant():
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionConstantLongRef(ExpressionLongShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_LONG_REF'
    __slots__ = ()

    def __init__(self, constant, user_provided, source_ref):
        if False:
            while True:
                i = 10
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantLongRef():
        if False:
            return 10
        return True

    @staticmethod
    def isMutable():
        if False:
            return 10
        return False

    @staticmethod
    def isNumberConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isIndexConstant():
        if False:
            return 10
        return True

    def getIndexValue(self):
        if False:
            while True:
                i = 10
        return int(self.constant)

    @staticmethod
    def isIterableConstant():
        if False:
            i = 10
            return i + 15
        return False

class ExpressionConstantStrRef(ExpressionStrShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_STR_REF'
    __slots__ = ()

    def __init__(self, constant, user_provided, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantStrRef():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isMutable():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getIterationHandle(self):
        if False:
            while True:
                i = 10
        return ConstantStrIterationHandle(self)

    def getIterationLength(self):
        if False:
            while True:
                i = 10
        return len(self.constant)

    def getStrValue(self):
        if False:
            i = 10
            return i + 15
        return self

    def getStringValue(self):
        if False:
            for i in range(10):
                print('nop')
        return self.constant

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return (iter_node, None, None)

class ExpressionConstantStrEmptyRef(EmptyContainerMixin, ExpressionConstantStrRef):
    kind = 'EXPRESSION_CONSTANT_STR_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            print('Hello World!')
        ExpressionConstantStrRef.__init__(self, constant='', user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantUnicodeRef(ExpressionUnicodeShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_UNICODE_REF'
    __slots__ = ()

    def __init__(self, constant, user_provided, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantUnicodeRef():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isMutable():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            i = 10
            return i + 15
        return True

    def getIterationHandle(self):
        if False:
            print('Hello World!')
        return ConstantUnicodeIterationHandle(self)

    def getIterationLength(self):
        if False:
            while True:
                i = 10
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            print('Hello World!')
        return (iter_node, None, None)

class ExpressionConstantUnicodeEmptyRef(EmptyContainerMixin, ExpressionConstantUnicodeRef):
    kind = 'EXPRESSION_CONSTANT_UNICODE_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            return 10
        ExpressionConstantUnicodeRef.__init__(self, constant=the_empty_unicode, user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantBytesRef(ExpressionBytesShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_BYTES_REF'

    def __init__(self, constant, user_provided, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantBytesRef():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isMutable():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getIterationHandle(self):
        if False:
            for i in range(10):
                print('nop')
        return ConstantBytesIterationHandle(self)

    def getIterationLength(self):
        if False:
            i = 10
            return i + 15
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            while True:
                i = 10
        return (iter_node, None, None)

class ExpressionConstantBytesEmptyRef(EmptyContainerMixin, ExpressionConstantBytesRef):
    kind = 'EXPRESSION_CONSTANT_BYTES_EMPTY_REF'
    __slots__ = ()

    def __init__(self, user_provided, source_ref):
        if False:
            return 10
        ExpressionConstantBytesRef.__init__(self, constant=b'', user_provided=user_provided, source_ref=source_ref)

class ExpressionConstantBytearrayRef(ExpressionBytearrayShapeExactMixin, ExpressionConstantRefBase):
    kind = 'EXPRESSION_CONSTANT_BYTEARRAY_REF'

    def __init__(self, constant, user_provided, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantRefBase.__init__(self, constant=constant, user_provided=user_provided, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantBytearrayRef():
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def isMutable():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isIterableConstant():
        if False:
            i = 10
            return i + 15
        return True

    def getIterationHandle(self):
        if False:
            while True:
                i = 10
        return ConstantBytearrayIterationHandle(self)

    def getIterationLength(self):
        if False:
            print('Hello World!')
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            i = 10
            return i + 15
        result = makeConstantRefNode(constant=bytes(self.constant), user_provided=self.user_provided, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        self.finalize()
        return (iter_node, 'new_constant', 'Iteration over constant bytearray lowered to bytes.')

class ExpressionConstantFloatRef(ExpressionFloatShapeExactMixin, ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_FLOAT_REF'
    __slots__ = ()

    def __init__(self, constant, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantUntrackedRefBase.__init__(self, constant=constant, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantFloatRef():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isMutable():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isNumberConstant():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isIterableConstant():
        if False:
            for i in range(10):
                print('nop')
        return False

class ExpressionConstantComplexRef(ExpressionComplexShapeExactMixin, ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_COMPLEX_REF'
    __slots__ = ()

    def __init__(self, constant, source_ref):
        if False:
            print('Hello World!')
        ExpressionConstantUntrackedRefBase.__init__(self, constant=constant, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantComplexRef():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isMutable():
        if False:
            return 10
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            return 10
        return False

    @staticmethod
    def isKnownToHaveAttribute(attribute_name):
        if False:
            return 10
        return hasattr(0j, attribute_name)

class ExpressionConstantSliceRef(ExpressionSliceShapeExactMixin, ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_SLICE_REF'
    __slots__ = ()

    def __init__(self, constant, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantUntrackedRefBase.__init__(self, constant=constant, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantSliceRef():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isMutable():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isIterableConstant():
        if False:
            i = 10
            return i + 15
        return False

class ExpressionConstantXrangeRef(ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_XRANGE_REF'
    __slots__ = ()

    def __init__(self, constant, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantUntrackedRefBase.__init__(self, constant=constant, source_ref=source_ref)

    @staticmethod
    def isExpressionConstantXrangeRef():
        if False:
            return 10
        return True

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_xrange

    @staticmethod
    def isMutable():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def isKnownToBeHashable():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def isIterableConstant():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getIterationHandle(self):
        if False:
            return 10
        return ConstantRangeIterationHandle(self)

    def getIterationLength(self):
        if False:
            return 10
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            return 10
        return (iter_node, None, None)

class ExpressionConstantTypeRef(ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_TYPE_REF'
    __slots__ = ()

    @staticmethod
    def isExpressionConstantTypeRef():
        if False:
            i = 10
            return i + 15
        return True

    @staticmethod
    def getTypeShape():
        if False:
            print('Hello World!')
        return tshape_type

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        if call_kw is not None and (not call_kw.isMappingWithConstantStringKeys()):
            return (call_node, None, None)
        else:
            from nuitka.optimizations.OptimizeBuiltinCalls import computeBuiltinCall
            (new_node, tags, message) = computeBuiltinCall(builtin_name=self.constant.__name__, call_node=call_node)
            return (new_node, tags, message)

    def computeExpressionCallViaVariable(self, call_node, variable_ref_node, call_args, call_kw, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return self.computeExpressionCall(call_node=call_node, call_args=call_args, call_kw=call_kw, trace_collection=trace_collection)

    @staticmethod
    def isMutable():
        if False:
            return 10
        return False

    @staticmethod
    def isKnownToBeHashable():
        if False:
            return 10
        return True

    @staticmethod
    def isIterableConstant():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def getTruthValue():
        if False:
            for i in range(10):
                print('nop')
        return True

class ExpressionConstantTypeSubscriptableMixin(object):
    __slots__ = ()
    if python_version >= 912:

        def computeExpressionSubscript(self, lookup_node, subscript, trace_collection):
            if False:
                i = 10
                return i + 15
            if subscript.isCompileTimeConstant():
                return trace_collection.getCompileTimeComputationResult(node=lookup_node, computation=lambda : self.getCompileTimeConstant()[subscript.getCompileTimeConstant()], description='Subscript of subscriptable type with constant value.')
            trace_collection.onExceptionRaiseExit(BaseException)
            return (lookup_node, None, None)

class ExpressionConstantConcreteTypeMixin(object):
    __slots__ = ()

    @staticmethod
    def getDetails():
        if False:
            for i in range(10):
                print('nop')
        return {}

class ExpressionConstantTypeDictRef(ExpressionConstantConcreteTypeMixin, ExpressionConstantTypeSubscriptableMixin, ExpressionConstantTypeRef):
    kind = 'EXPRESSION_CONSTANT_TYPE_DICT_REF'

    def __init__(self, source_ref):
        if False:
            while True:
                i = 10
        ExpressionConstantTypeRef.__init__(self, constant=dict, source_ref=source_ref)

class ExpressionConstantTypeSetRef(ExpressionConstantConcreteTypeMixin, ExpressionConstantTypeSubscriptableMixin, ExpressionConstantTypeRef):
    kind = 'EXPRESSION_CONSTANT_TYPE_SET_REF'

    def __init__(self, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantTypeRef.__init__(self, constant=set, source_ref=source_ref)

class ExpressionConstantTypeFrozensetRef(ExpressionConstantConcreteTypeMixin, ExpressionConstantTypeSubscriptableMixin, ExpressionConstantTypeRef):
    kind = 'EXPRESSION_CONSTANT_TYPE_FROZENSET_REF'

    def __init__(self, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionConstantTypeRef.__init__(self, constant=frozenset, source_ref=source_ref)

class ExpressionConstantTypeListRef(ExpressionConstantConcreteTypeMixin, ExpressionConstantTypeSubscriptableMixin, ExpressionConstantTypeRef):
    kind = 'EXPRESSION_CONSTANT_TYPE_LIST_REF'

    def __init__(self, source_ref):
        if False:
            print('Hello World!')
        ExpressionConstantTypeRef.__init__(self, constant=list, source_ref=source_ref)

class ExpressionConstantTypeTupleRef(ExpressionConstantConcreteTypeMixin, ExpressionConstantTypeSubscriptableMixin, ExpressionConstantTypeRef):
    kind = 'EXPRESSION_CONSTANT_TYPE_TUPLE_REF'

    def __init__(self, source_ref):
        if False:
            return 10
        ExpressionConstantTypeRef.__init__(self, constant=tuple, source_ref=source_ref)

class ExpressionConstantTypeTypeRef(ExpressionConstantConcreteTypeMixin, ExpressionConstantTypeSubscriptableMixin, ExpressionConstantTypeRef):
    kind = 'EXPRESSION_CONSTANT_TYPE_TYPE_REF'

    def __init__(self, source_ref):
        if False:
            return 10
        ExpressionConstantTypeRef.__init__(self, constant=type, source_ref=source_ref)

def makeConstantRefNode(constant, source_ref, user_provided=False):
    if False:
        while True:
            i = 10
    if constant is None:
        return ExpressionConstantNoneRef(source_ref=source_ref)
    elif constant is True:
        return ExpressionConstantTrueRef(source_ref=source_ref)
    elif constant is False:
        return ExpressionConstantFalseRef(source_ref=source_ref)
    elif constant is Ellipsis:
        return ExpressionConstantEllipsisRef(source_ref=source_ref)
    constant_type = type(constant)
    if constant_type is int:
        return ExpressionConstantIntRef(constant=constant, source_ref=source_ref)
    elif constant_type is str:
        if constant:
            return ExpressionConstantStrRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantStrEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is float:
        return ExpressionConstantFloatRef(constant=constant, source_ref=source_ref)
    elif constant_type is long:
        return ExpressionConstantLongRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
    elif constant_type is unicode:
        if constant:
            return ExpressionConstantUnicodeRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantUnicodeEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is bytes:
        if constant:
            return ExpressionConstantBytesRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantBytesEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is dict:
        if constant:
            assert isConstant(constant), repr(constant)
            return ExpressionConstantDictRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantDictEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is tuple:
        if constant:
            assert isConstant(constant), repr(constant)
            if isMutable(constant):
                return ExpressionConstantTupleMutableRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
            else:
                return ExpressionConstantTupleRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantTupleEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is list:
        if constant:
            assert isConstant(constant), repr(constant)
            return ExpressionConstantListRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantListEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is set:
        if constant:
            assert isConstant(constant), repr(constant)
            return ExpressionConstantSetRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantSetEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is frozenset:
        if constant:
            assert isConstant(constant), repr(constant)
            return ExpressionConstantFrozensetRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
        else:
            return ExpressionConstantFrozensetEmptyRef(user_provided=user_provided, source_ref=source_ref)
    elif constant_type is complex:
        return ExpressionConstantComplexRef(constant=constant, source_ref=source_ref)
    elif constant_type is slice:
        return ExpressionConstantSliceRef(constant=constant, source_ref=source_ref)
    elif constant_type is type:
        if constant is dict:
            return ExpressionConstantTypeDictRef(source_ref=source_ref)
        if constant is set:
            return ExpressionConstantTypeSetRef(source_ref=source_ref)
        if constant is frozenset:
            return ExpressionConstantTypeFrozensetRef(source_ref=source_ref)
        if constant is tuple:
            return ExpressionConstantTypeTupleRef(source_ref=source_ref)
        if constant is list:
            return ExpressionConstantTypeListRef(source_ref=source_ref)
        if constant is type:
            return ExpressionConstantTypeTypeRef(source_ref=source_ref)
        return ExpressionConstantTypeRef(constant=constant, source_ref=source_ref)
    elif constant_type is xrange:
        return ExpressionConstantXrangeRef(constant=constant, source_ref=source_ref)
    elif constant_type is bytearray:
        return ExpressionConstantBytearrayRef(constant=constant, user_provided=user_provided, source_ref=source_ref)
    elif constant_type is GenericAlias:
        from .BuiltinTypeNodes import ExpressionConstantGenericAlias
        return ExpressionConstantGenericAlias(generic_alias=constant, source_ref=source_ref)
    elif constant_type is UnionType:
        from .BuiltinTypeNodes import ExpressionConstantUnionType
        return ExpressionConstantUnionType(union_type=constant, source_ref=source_ref)
    elif constant is sys.version_info:
        return ExpressionConstantSysVersionInfoRef(source_ref=source_ref)
    elif constant in builtin_anon_values:
        from .BuiltinRefNodes import ExpressionBuiltinAnonymousRef
        return ExpressionBuiltinAnonymousRef(builtin_name=builtin_anon_values[constant], source_ref=source_ref)
    elif constant in builtin_named_values:
        from .BuiltinRefNodes import ExpressionBuiltinRef
        return ExpressionBuiltinRef(builtin_name=builtin_named_values[constant], source_ref=source_ref)
    elif constant in builtin_exception_values_list:
        from .BuiltinRefNodes import ExpressionBuiltinExceptionRef
        if constant is NotImplemented:
            exception_name = 'NotImplemented'
        else:
            exception_name = constant.__name__
        return ExpressionBuiltinExceptionRef(exception_name=exception_name, source_ref=source_ref)
    else:
        assert False, (constant, constant_type)

class ExpressionConstantSysVersionInfoRef(ExpressionConstantUntrackedRefBase):
    kind = 'EXPRESSION_CONSTANT_SYS_VERSION_INFO_REF'
    __slots__ = ()

    def __init__(self, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionConstantUntrackedRefBase.__init__(self, constant=sys.version_info, source_ref=source_ref)

    @staticmethod
    def getDetails():
        if False:
            for i in range(10):
                print('nop')
        return {}

    @staticmethod
    def getTypeShape():
        if False:
            for i in range(10):
                print('nop')
        return tshape_namedtuple

    @staticmethod
    def isMutable():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def isKnownToBeHashable():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def isIterableConstant():
        if False:
            return 10
        return True

    def getIterationHandle(self):
        if False:
            i = 10
            return i + 15
        return ConstantTupleIterationHandle(self)

    def getIterationLength(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.constant)

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            i = 10
            return i + 15
        result = makeConstantRefNode(constant=tuple(self.constant), user_provided=True, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        self.finalize()
        return (iter_node, 'new_constant', "Iteration over constant 'sys.version_info' lowered to tuple.")

    @staticmethod
    def getTruthValue():
        if False:
            while True:
                i = 10
        return True