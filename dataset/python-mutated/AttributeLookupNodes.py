""" Attribute lookup nodes, generic one and base for generated ones.

See AttributeNodes otherwise.
"""
from .ChildrenHavingMixins import ChildHavingExpressionMixin
from .ExpressionBases import ExpressionBase
from .ExpressionBasesGenerated import ExpressionAttributeLookupBase

class ExpressionAttributeLookup(ExpressionAttributeLookupBase):
    """Looking up an attribute of an object.

    Typically code like: source.attribute_name
    """
    kind = 'EXPRESSION_ATTRIBUTE_LOOKUP'
    named_children = ('expression',)
    node_attributes = ('attribute_name',)

    def getAttributeName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.attribute_name

    def computeExpression(self, trace_collection):
        if False:
            return 10
        return self.subnode_expression.computeExpressionAttribute(lookup_node=self, attribute_name=self.attribute_name, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return self.subnode_expression.mayRaiseException(exception_type) or self.subnode_expression.mayRaiseExceptionAttributeLookup(exception_type=exception_type, attribute_name=self.attribute_name)

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            i = 10
            return i + 15
        return None

class ExpressionAttributeLookupSpecial(ExpressionAttributeLookup):
    """Special lookup up an attribute of an object.

    Typically from code like this: with source: pass

    These directly go to slots, and are performed for with statements
    of Python2.7 or higher.
    """
    kind = 'EXPRESSION_ATTRIBUTE_LOOKUP_SPECIAL'

    def computeExpression(self, trace_collection):
        if False:
            i = 10
            return i + 15
        return self.subnode_expression.computeExpressionAttributeSpecial(lookup_node=self, attribute_name=self.attribute_name, trace_collection=trace_collection)

class ExpressionAttributeLookupFixedBase(ChildHavingExpressionMixin, ExpressionBase):
    """Looking up an attribute of an object.

    Typically code like: source.attribute_name
    """
    attribute_name = None
    named_children = ('expression',)

    def __init__(self, expression, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildHavingExpressionMixin.__init__(self, expression=expression)
        ExpressionBase.__init__(self, source_ref)

    def getAttributeName(self):
        if False:
            i = 10
            return i + 15
        return self.attribute_name

    @staticmethod
    def getDetails():
        if False:
            for i in range(10):
                print('nop')
        return {}

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_expression.computeExpressionAttribute(lookup_node=self, attribute_name=self.attribute_name, trace_collection=trace_collection)

    def mayRaiseException(self, exception_type):
        if False:
            for i in range(10):
                print('nop')
        return self.subnode_expression.mayRaiseException(exception_type) or self.subnode_expression.mayRaiseExceptionAttributeLookup(exception_type=exception_type, attribute_name=self.attribute_name)

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            i = 10
            return i + 15
        return None