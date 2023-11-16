""" Print nodes.

Right now there is only the print statement, but in principle, there should
also be the print function here. These perform output, which can be combined
if possible, and could be detected to fail, which would be perfect.

Predicting the behavior of 'print' is not trivial at all, due to many special
cases.
"""
from .NodeMakingHelpers import makeStatementExpressionOnlyReplacementNode, makeStatementsSequenceReplacementNode
from .StatementBasesGenerated import StatementPrintNewlineBase, StatementPrintValueBase

class StatementPrintValue(StatementPrintValueBase):
    kind = 'STATEMENT_PRINT_VALUE'
    named_children = ('dest|optional', 'value|setter')
    auto_compute_handling = 'operation'
    python_version_spec = '< 0x300'

    def computeStatementOperation(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        if self.subnode_dest is None and self.subnode_value.isExpressionSideEffects():
            self.setChildValue(self.subnode_value.value.subnode_expression)
            statements = [makeStatementExpressionOnlyReplacementNode(side_effect, self) for side_effect in self.subnode_value.value.subnode_side_effects]
            statements.append(self)
            result = makeStatementsSequenceReplacementNode(statements=statements, node=self)
            return (result, 'new_statements', 'Side effects printed item promoted to statements.')
        if self.subnode_value.isCompileTimeConstant() and (not self.subnode_value.hasShapeStrOrUnicodeExact()):
            new_value = self.subnode_value.getStrValue()
            assert new_value is not None, self.subnode_value
            self.setChildValue(new_value)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            i = 10
            return i + 15
        return True

class StatementPrintNewline(StatementPrintNewlineBase):
    kind = 'STATEMENT_PRINT_NEWLINE'
    named_children = ('dest|optional',)
    python_version_spec = '< 0x300'
    auto_compute_handling = 'operation'

    def computeStatementOperation(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            i = 10
            return i + 15
        return True