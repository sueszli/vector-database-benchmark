""" Nodes for statements.

"""
from nuitka.PythonVersions import python_version
from .NodeBases import StatementBase
from .StatementBasesGenerated import StatementExpressionOnlyBase, StatementsSequenceBase

class StatementsSequenceMixin(object):
    __slots__ = ()

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent
        for s in self.subnode_statements:
            s.finalize()

    @staticmethod
    def isStatementsSequence():
        if False:
            for i in range(10):
                print('nop')
        return True

    def trimStatements(self, statement):
        if False:
            i = 10
            return i + 15
        assert statement.parent is self
        old_statements = list(self.subnode_statements)
        assert statement in old_statements, (statement, self)
        new_statements = old_statements[:old_statements.index(statement) + 1]
        self.setChildStatements(new_statements)

    def removeStatement(self, statement):
        if False:
            for i in range(10):
                print('nop')
        assert statement.parent is self
        statements = list(self.subnode_statements)
        statements.remove(statement)
        self.setChildStatements(tuple(statements))
        if statements:
            return self
        else:
            return None

    def replaceStatement(self, statement, statements):
        if False:
            i = 10
            return i + 15
        old_statements = list(self.subnode_statements)
        merge_index = old_statements.index(statement)
        new_statements = tuple(old_statements[:merge_index]) + tuple(statements) + tuple(old_statements[merge_index + 1:])
        self.setChildStatements(new_statements)

    def mayHaveSideEffects(self):
        if False:
            while True:
                i = 10
        for statement in self.subnode_statements:
            if statement.mayHaveSideEffects():
                return True
        return False

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        for statement in self.subnode_statements:
            if statement.mayRaiseException(exception_type):
                return True
        return False

    def needsFrame(self):
        if False:
            i = 10
            return i + 15
        for statement in self.subnode_statements:
            if statement.needsFrame():
                return True
        return False

    def mayReturn(self):
        if False:
            for i in range(10):
                print('nop')
        for statement in self.subnode_statements:
            if statement.mayReturn():
                return True
        return False

    def mayBreak(self):
        if False:
            while True:
                i = 10
        for statement in self.subnode_statements:
            if statement.mayBreak():
                return True
        return False

    def mayContinue(self):
        if False:
            print('Hello World!')
        for statement in self.subnode_statements:
            if statement.mayContinue():
                return True
        return False

    def mayRaiseExceptionOrAbort(self, exception_type):
        if False:
            while True:
                i = 10
        return self.mayRaiseException(exception_type) or self.mayReturn() or self.mayBreak() or self.mayContinue()

    def isStatementAborting(self):
        if False:
            return 10
        return self.subnode_statements[-1].isStatementAborting()

    def willRaiseAnyException(self):
        if False:
            print('Hello World!')
        return self.subnode_statements[-1].willRaiseAnyException()

class StatementsSequence(StatementsSequenceMixin, StatementsSequenceBase):
    kind = 'STATEMENTS_SEQUENCE'
    named_children = ('statements|tuple+setter',)

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        assert False, self

    def computeStatementsSequence(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        new_statements = []
        statements = self.subnode_statements
        assert statements, self
        for (count, statement) in enumerate(statements):
            if statement.isStatementsFrame():
                new_statement = statement.computeStatementsSequence(trace_collection)
            else:
                new_statement = trace_collection.onStatement(statement=statement)
            if new_statement is not None:
                if new_statement.isStatementsSequence() and (not new_statement.isStatementsFrame()):
                    new_statements.extend(new_statement.subnode_statements)
                else:
                    new_statements.append(new_statement)
                if statement is not statements[-1] and new_statement.isStatementAborting():
                    trace_collection.signalChange('new_statements', statements[count + 1].getSourceReference(), 'Removed dead statements.')
                    for s in statements[statements.index(statement) + 1:]:
                        s.finalize()
                    break
        new_statements = tuple(new_statements)
        if statements != new_statements:
            if new_statements:
                self.setChildStatements(new_statements)
                return self
            else:
                return None
        else:
            return self

    @staticmethod
    def getStatementNiceName():
        if False:
            while True:
                i = 10
        return 'statements sequence'

class StatementExpressionOnly(StatementExpressionOnlyBase):
    kind = 'STATEMENT_EXPRESSION_ONLY'
    named_children = ('expression',)

    def mayHaveSideEffects(self):
        if False:
            while True:
                i = 10
        return self.subnode_expression.mayHaveSideEffects()

    def mayRaiseException(self, exception_type):
        if False:
            return 10
        return self.subnode_expression.mayRaiseException(exception_type)

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        expression = trace_collection.onExpression(self.subnode_expression)
        return expression.computeExpressionDrop(statement=self, trace_collection=trace_collection)

    @staticmethod
    def getStatementNiceName():
        if False:
            while True:
                i = 10
        return 'expression only statement'

    def getDetailsForDisplay(self):
        if False:
            i = 10
            return i + 15
        return {'expression': self.subnode_expression.kind}

class StatementPreserveFrameException(StatementBase):
    kind = 'STATEMENT_PRESERVE_FRAME_EXCEPTION'
    __slots__ = ('preserver_id',)

    def __init__(self, preserver_id, source_ref):
        if False:
            for i in range(10):
                print('nop')
        StatementBase.__init__(self, source_ref=source_ref)
        self.preserver_id = preserver_id

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

    def getDetails(self):
        if False:
            while True:
                i = 10
        return {'preserver_id': self.preserver_id}

    def getPreserverId(self):
        if False:
            while True:
                i = 10
        return self.preserver_id
    if python_version < 768:

        def computeStatement(self, trace_collection):
            if False:
                print('Hello World!')
            if self.getParentStatementsFrame().needsExceptionFramePreservation():
                return (self, None, None)
            else:
                return (None, 'new_statements', 'Removed frame preservation for generators.')
    else:

        def computeStatement(self, trace_collection):
            if False:
                print('Hello World!')
            return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def needsFrame():
        if False:
            i = 10
            return i + 15
        return True

class StatementRestoreFrameException(StatementBase):
    kind = 'STATEMENT_RESTORE_FRAME_EXCEPTION'
    __slots__ = ('preserver_id',)

    def __init__(self, preserver_id, source_ref):
        if False:
            i = 10
            return i + 15
        StatementBase.__init__(self, source_ref=source_ref)
        self.preserver_id = preserver_id

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'preserver_id': self.preserver_id}

    def getPreserverId(self):
        if False:
            print('Hello World!')
        return self.preserver_id

    def computeStatement(self, trace_collection):
        if False:
            while True:
                i = 10
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return False

class StatementPublishException(StatementBase):
    kind = 'STATEMENT_PUBLISH_EXCEPTION'

    def __init__(self, source_ref):
        if False:
            for i in range(10):
                print('nop')
        StatementBase.__init__(self, source_ref=source_ref)

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

    def computeStatement(self, trace_collection):
        if False:
            return 10
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            i = 10
            return i + 15
        return False