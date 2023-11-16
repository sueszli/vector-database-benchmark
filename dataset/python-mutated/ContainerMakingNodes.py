""" Nodes that build containers.

"""
import functools
from abc import abstractmethod
from nuitka.PythonVersions import needsSetLiteralReverseInsertion
from .ChildrenHavingMixins import ChildHavingElementsTupleMixin
from .ConstantRefNodes import ExpressionConstantListEmptyRef, ExpressionConstantSetEmptyRef, ExpressionConstantTupleEmptyRef, makeConstantRefNode
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionListShapeExactMixin, ExpressionSetShapeExactMixin, ExpressionTupleShapeExactMixin
from .IterationHandles import ListAndTupleContainerMakingIterationHandle
from .NodeBases import SideEffectsFromChildrenMixin
from .NodeMakingHelpers import makeStatementOnlyNodesFromExpressions

class ExpressionMakeSequenceMixin(object):
    __slots__ = ()

    def isKnownToBeIterable(self, count):
        if False:
            return 10
        return count is None or count == len(self.subnode_elements)

    def isKnownToBeIterableAtMin(self, count):
        if False:
            while True:
                i = 10
        return count <= len(self.subnode_elements)

    def getIterationValue(self, count):
        if False:
            while True:
                i = 10
        return self.subnode_elements[count]

    def getIterationValueRange(self, start, stop):
        if False:
            while True:
                i = 10
        return self.subnode_elements[start:stop]

    @staticmethod
    def canPredictIterationValues():
        if False:
            while True:
                i = 10
        return True

    def getIterationValues(self):
        if False:
            i = 10
            return i + 15
        return self.subnode_elements

    def getIterationHandle(self):
        if False:
            while True:
                i = 10
        return ListAndTupleContainerMakingIterationHandle(self.subnode_elements)

    @staticmethod
    def getTruthValue():
        if False:
            return 10
        return True

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        for element in self.subnode_elements:
            if element.mayRaiseException(exception_type):
                return True
        return False

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        result = makeStatementOnlyNodesFromExpressions(expressions=self.subnode_elements)
        del self.parent
        return (result, 'new_statements', 'Removed %s creation for unused sequence.' % self.getSequenceName())

    def onContentEscapes(self, trace_collection):
        if False:
            while True:
                i = 10
        for element in self.subnode_elements:
            element.onContentEscapes(trace_collection)

    @abstractmethod
    def getSequenceName(self):
        if False:
            for i in range(10):
                print('nop')
        'Get name for use in traces'

class ExpressionMakeSequenceBase(SideEffectsFromChildrenMixin, ExpressionMakeSequenceMixin, ChildHavingElementsTupleMixin, ExpressionBase):
    named_children = ('elements|tuple',)

    def __init__(self, elements, source_ref):
        if False:
            for i in range(10):
                print('nop')
        assert elements
        ChildHavingElementsTupleMixin.__init__(self, elements=elements)
        ExpressionBase.__init__(self, source_ref)

    def getSequenceName(self):
        if False:
            print('Hello World!')
        'Get name for use in traces'
        simulator = self.getSimulator()
        return simulator.__name__.capitalize()

    @staticmethod
    def isExpressionMakeSequence():
        if False:
            while True:
                i = 10
        return True

    @abstractmethod
    def getSimulator(self):
        if False:
            print('Hello World!')
        'The simulator for the container making, for overload.'

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        for element in self.subnode_elements:
            if not element.isCompileTimeConstant():
                return (self, None, None)
        simulator = self.getSimulator()
        assert simulator is not None
        return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : simulator((element.getCompileTimeConstant() for element in self.subnode_elements)), description='%s with constant arguments.' % simulator.__name__.capitalize(), user_provided=True)

def makeExpressionMakeTuple(elements, source_ref):
    if False:
        for i in range(10):
            print('nop')
    if elements:
        return ExpressionMakeTuple(elements, source_ref)
    else:
        return ExpressionConstantTupleEmptyRef(user_provided=False, source_ref=source_ref)

def makeExpressionMakeTupleOrConstant(elements, user_provided, source_ref):
    if False:
        return 10
    for element in elements:
        if not element.isExpressionConstantRef():
            result = makeExpressionMakeTuple(elements, source_ref)
            break
    else:
        result = makeConstantRefNode(constant=tuple((element.getCompileTimeConstant() for element in elements)), user_provided=user_provided, source_ref=source_ref)
    if elements:
        result.setCompatibleSourceReference(source_ref=elements[-1].getCompatibleSourceReference())
    return result

class ExpressionMakeTuple(ExpressionTupleShapeExactMixin, ExpressionMakeSequenceBase):
    kind = 'EXPRESSION_MAKE_TUPLE'

    def __init__(self, elements, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ExpressionMakeSequenceBase.__init__(self, elements=elements, source_ref=source_ref)

    @staticmethod
    def getSimulator():
        if False:
            while True:
                i = 10
        return tuple

    def getIterationLength(self):
        if False:
            print('Hello World!')
        return len(self.subnode_elements)

def makeExpressionMakeList(elements, source_ref):
    if False:
        while True:
            i = 10
    if elements:
        return ExpressionMakeList(elements, source_ref)
    else:
        return ExpressionConstantListEmptyRef(user_provided=False, source_ref=source_ref)

def makeExpressionMakeListOrConstant(elements, user_provided, source_ref):
    if False:
        for i in range(10):
            print('nop')
    for element in elements:
        if not element.isExpressionConstantRef():
            result = makeExpressionMakeList(elements, source_ref)
            break
    else:
        result = makeConstantRefNode(constant=[element.getCompileTimeConstant() for element in elements], user_provided=user_provided, source_ref=source_ref)
    if elements:
        result.setCompatibleSourceReference(source_ref=elements[-1].getCompatibleSourceReference())
    return result

class ExpressionMakeListMixin(object):
    __slots__ = ()

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        result = ExpressionMakeTuple(elements=self.subnode_elements, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        del self.parent
        return (iter_node, 'new_expression', 'Iteration over list lowered to iteration over tuple.')

class ExpressionMakeList(ExpressionListShapeExactMixin, ExpressionMakeListMixin, ExpressionMakeSequenceBase):
    kind = 'EXPRESSION_MAKE_LIST'

    def __init__(self, elements, source_ref):
        if False:
            while True:
                i = 10
        ExpressionMakeSequenceBase.__init__(self, elements=elements, source_ref=source_ref)

    @staticmethod
    def getSimulator():
        if False:
            while True:
                i = 10
        return list

    def getIterationLength(self):
        if False:
            return 10
        return len(self.subnode_elements)

class ExpressionMakeSet(ExpressionSetShapeExactMixin, ExpressionMakeSequenceBase):
    kind = 'EXPRESSION_MAKE_SET'

    def __init__(self, elements, source_ref):
        if False:
            i = 10
            return i + 15
        ExpressionMakeSequenceBase.__init__(self, elements=elements, source_ref=source_ref)

    @staticmethod
    def getSimulator():
        if False:
            return 10
        return set

    def getIterationLength(self):
        if False:
            for i in range(10):
                print('nop')
        element_count = len(self.subnode_elements)
        if element_count >= 2:
            return None
        else:
            return element_count

    @staticmethod
    def getIterationMinLength():
        if False:
            for i in range(10):
                print('nop')
        return 1

    def computeExpression(self, trace_collection):
        if False:
            return 10
        are_constants = True
        are_hashable = True
        for element in self.subnode_elements:
            if are_constants and (not element.isCompileTimeConstant()):
                are_constants = False
            if are_hashable and (not element.isKnownToBeHashable()):
                are_hashable = False
            if not are_hashable and (not are_constants):
                break
        if not are_constants:
            if not are_hashable:
                trace_collection.onExceptionRaiseExit(BaseException)
            return (self, None, None)
        simulator = self.getSimulator()
        assert simulator is not None
        return trace_collection.getCompileTimeComputationResult(node=self, computation=lambda : simulator((element.getCompileTimeConstant() for element in self.subnode_elements)), description='%s with constant arguments.' % simulator.__name__.capitalize(), user_provided=True)

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        for element in self.subnode_elements:
            if not element.isKnownToBeHashable():
                return True
            if element.mayRaiseException(exception_type):
                return True
        return False

    def computeExpressionIter1(self, iter_node, trace_collection):
        if False:
            return 10
        result = ExpressionMakeTuple(elements=self.subnode_elements, source_ref=self.source_ref)
        self.parent.replaceChild(self, result)
        del self.parent
        return (iter_node, 'new_expression', 'Iteration over set lowered to iteration over tuple.')
needs_set_literal_reverse = needsSetLiteralReverseInsertion()

def makeExpressionMakeSetLiteral(elements, source_ref):
    if False:
        print('Hello World!')
    if elements:
        if needs_set_literal_reverse:
            return ExpressionMakeSetLiteral(elements, source_ref)
        else:
            return ExpressionMakeSet(elements, source_ref)
    else:
        return ExpressionConstantSetEmptyRef(user_provided=False, source_ref=source_ref)

@functools.wraps(set)
def reversed_set(value):
    if False:
        while True:
            i = 10
    return set(reversed(tuple(value)))

def makeExpressionMakeSetLiteralOrConstant(elements, user_provided, source_ref):
    if False:
        return 10
    for element in elements:
        if not element.isExpressionConstantRef():
            result = makeExpressionMakeSetLiteral(elements, source_ref)
            break
    else:
        if needs_set_literal_reverse:
            elements = tuple(reversed(elements))
        result = makeConstantRefNode(constant=set((element.getCompileTimeConstant() for element in elements)), user_provided=user_provided, source_ref=source_ref)
    if elements:
        result.setCompatibleSourceReference(source_ref=elements[-1].getCompatibleSourceReference())
    return result

class ExpressionMakeSetLiteral(ExpressionMakeSet):
    kind = 'EXPRESSION_MAKE_SET_LITERAL'

    @staticmethod
    def getSimulator():
        if False:
            while True:
                i = 10
        return reversed_set