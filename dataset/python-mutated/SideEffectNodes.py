""" Node that models side effects.

Sometimes, the effect of an expression needs to be had, but the value itself
does not matter at all.
"""
from .ChildrenHavingMixins import ChildrenHavingSideEffectsTupleExpressionMixin
from .ExpressionBases import ExpressionBase
from .NodeMakingHelpers import makeStatementOnlyNodesFromExpressions

class ExpressionSideEffects(ChildrenHavingSideEffectsTupleExpressionMixin, ExpressionBase):
    kind = 'EXPRESSION_SIDE_EFFECTS'
    named_children = ('side_effects|tuple+setter', 'expression|setter')

    def __init__(self, side_effects, expression, source_ref):
        if False:
            return 10
        ChildrenHavingSideEffectsTupleExpressionMixin.__init__(self, side_effects=side_effects, expression=expression)
        ExpressionBase.__init__(self, source_ref)

    @staticmethod
    def isExpressionSideEffects():
        if False:
            while True:
                i = 10
        return True

    def getTypeShape(self):
        if False:
            while True:
                i = 10
        return self.subnode_expression.getTypeShape()

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        new_side_effects = []
        side_effects = self.subnode_side_effects
        for (count, side_effect) in enumerate(side_effects):
            side_effect = trace_collection.onExpression(side_effect)
            if side_effect.willRaiseAnyException():
                for c in side_effects[count + 1:]:
                    c.finalize()
                if new_side_effects:
                    expression = self.subnode_expression
                    expression.finalize()
                    self.setChildExpression(side_effect)
                    return (self, 'new_expression', 'Side effects caused exception raise.')
                else:
                    del self.parent
                    del self.subnode_side_effects
                    return (side_effect, 'new_expression', 'Side effects caused exception raise.')
            if side_effect.isExpressionSideEffects():
                new_side_effects.extend(side_effect.subnode_side_effects)
                del side_effect.parent
                del side_effect.subnode_side_effects
            elif side_effect is not None and side_effect.mayHaveSideEffects():
                new_side_effects.append(side_effect)
        self.setChildSideEffects(tuple(new_side_effects))
        trace_collection.onExpression(self.subnode_expression)
        if not new_side_effects:
            return (self.subnode_expression, 'new_expression', 'Removed now empty side effects.')
        return (self, None, None)

    def computeExpression(self, trace_collection):
        if False:
            while True:
                i = 10
        return (self, None, None)

    def getTruthValue(self):
        if False:
            while True:
                i = 10
        return self.subnode_expression.getTruthValue()

    def mayRaiseException(self, exception_type):
        if False:
            print('Hello World!')
        return self.subnode_expression.mayRaiseException(exception_type) or any((side_effect.mayRaiseException(exception_type) for side_effect in self.subnode_side_effects))

    def computeExpressionDrop(self, statement, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        expressions = self.subnode_side_effects + (self.subnode_expression,)
        result = makeStatementOnlyNodesFromExpressions(expressions=expressions)
        return (result, 'new_statements', 'Turned side effects of expression only statement into statements.')

    @staticmethod
    def canPredictIterationValues():
        if False:
            for i in range(10):
                print('nop')
        return False

    def willRaiseAnyException(self):
        if False:
            print('Hello World!')
        return self.subnode_expression.willRaiseAnyException()