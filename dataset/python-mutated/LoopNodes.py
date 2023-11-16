""" Loop nodes.

There are for and loop nodes, but both are reduced to loops with break/continue
statements for it. These re-formulations require that optimization of loops has
to be very general, yet the node type for loop, becomes very simple.
"""
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.optimizations.TraceCollections import TraceCollectionBranch
from .NodeBases import StatementBase
from .shapes.StandardShapes import tshape_unknown, tshape_unknown_loop
from .StatementBasesGenerated import StatementLoopBase
tshape_unknown_set = frozenset([tshape_unknown])

def minimizeShapes(shapes):
    if False:
        i = 10
        return i + 15
    if tshape_unknown in shapes:
        return tshape_unknown_set
    return shapes

class StatementLoop(StatementLoopBase):
    kind = 'STATEMENT_LOOP'
    named_children = ('loop_body|statements_or_none+setter',)
    auto_compute_handling = 'post_init'
    __slots__ = ('loop_variables', 'loop_start', 'loop_resume', 'loop_previous_resume', 'incomplete_count')

    def postInitNode(self):
        if False:
            while True:
                i = 10
        self.loop_variables = None
        self.loop_start = {}
        self.loop_resume = {}
        self.loop_previous_resume = {}
        self.incomplete_count = 0

    def mayReturn(self):
        if False:
            print('Hello World!')
        loop_body = self.subnode_loop_body
        if loop_body is not None and loop_body.mayReturn():
            return True
        return False

    @staticmethod
    def mayBreak():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def mayContinue():
        if False:
            for i in range(10):
                print('nop')
        return False

    def isStatementAborting(self):
        if False:
            return 10
        loop_body = self.subnode_loop_body
        if loop_body is None:
            return True
        else:
            return not loop_body.mayBreak()

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return True

    def _computeLoopBody(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        loop_body = self.subnode_loop_body
        if loop_body is None:
            return (None, None, None)
        if self.loop_variables is None:
            self.loop_variables = OrderedSet()
            loop_body.collectVariableAccesses(self.loop_variables.add, self.loop_variables.add)
            all_first_pass = True
        else:
            all_first_pass = False
        incomplete_variables = None
        loop_entry_traces = set()
        for loop_variable in self.loop_variables:
            current = trace_collection.getVariableCurrentTrace(loop_variable)
            if all_first_pass:
                first_pass = True
                self.loop_start[loop_variable] = current
            elif not self.loop_start[loop_variable].compareValueTrace(current):
                first_pass = True
                self.loop_start[loop_variable] = current
            else:
                first_pass = False
            if first_pass:
                incomplete = True
                self.loop_previous_resume[loop_variable] = None
                self.loop_resume[loop_variable] = set()
                current.getTypeShape().emitAlternatives(self.loop_resume[loop_variable].add)
            elif self.loop_resume[loop_variable] != self.loop_previous_resume[loop_variable]:
                incomplete = True
                if incomplete_variables is None:
                    incomplete_variables = set()
                incomplete_variables.add(loop_variable)
            else:
                incomplete = False
            loop_entry_traces.add((loop_variable, trace_collection.markActiveVariableAsLoopMerge(loop_node=self, current=current, variable=loop_variable, shapes=self.loop_resume[loop_variable], incomplete=incomplete)))
        abort_context = trace_collection.makeAbortStackContext(catch_breaks=True, catch_continues=True, catch_returns=False, catch_exceptions=False)
        with abort_context:
            result = loop_body.computeStatementsSequence(trace_collection=trace_collection)
            if result is not loop_body:
                self.setChildLoopBody(result)
                loop_body = result
            if loop_body is not None:
                if not loop_body.isStatementAborting():
                    trace_collection.onLoopContinue()
            continue_collections = trace_collection.getLoopContinueCollections()
            self.loop_variables = []
            for (loop_variable, loop_entry_trace) in loop_entry_traces:
                if self.incomplete_count >= 20:
                    self.loop_previous_resume[loop_variable] = self.loop_resume[loop_variable] = set((tshape_unknown_loop,))
                    continue
                self.loop_previous_resume[loop_variable] = self.loop_resume[loop_variable]
                self.loop_resume[loop_variable] = set()
                loop_resume_traces = set((continue_collection.getVariableCurrentTrace(loop_variable) for continue_collection in continue_collections))
                if not loop_resume_traces or (len(loop_resume_traces) == 1 and loop_entry_trace.compareValueTrace(next(iter(loop_resume_traces)))):
                    del self.loop_resume[loop_variable]
                    del self.loop_previous_resume[loop_variable]
                    del self.loop_start[loop_variable]
                    continue
                self.loop_variables.append(loop_variable)
                loop_entry_trace.addLoopContinueTraces(loop_resume_traces)
                loop_resume_traces.add(self.loop_start[loop_variable])
                shapes = set()
                for loop_resume_trace in loop_resume_traces:
                    loop_resume_trace.getTypeShape().emitAlternatives(shapes.add)
                self.loop_resume[loop_variable] = minimizeShapes(shapes)
            break_collections = trace_collection.getLoopBreakCollections()
        if incomplete_variables:
            self.incomplete_count += 1
            trace_collection.signalChange('loop_analysis', self.source_ref, lambda : "Loop has incomplete variable types after %d attempts for '%s'." % (self.incomplete_count, ','.join((variable.getName() for variable in incomplete_variables))))
        elif self.incomplete_count:
            trace_collection.signalChange('loop_analysis', self.source_ref, lambda : 'Loop has complete variable types after %d attempts.' % self.incomplete_count)
            self.incomplete_count = 0
        return (loop_body, break_collections, continue_collections)

    def computeStatement(self, trace_collection):
        if False:
            while True:
                i = 10
        outer_trace_collection = trace_collection
        trace_collection = TraceCollectionBranch(parent=trace_collection, name='loop')
        (loop_body, break_collections, continue_collections) = self._computeLoopBody(trace_collection)
        if break_collections:
            outer_trace_collection.mergeMultipleBranches(break_collections)
        if loop_body is not None:
            assert loop_body.isStatementsSequence()
            statements = loop_body.subnode_statements
            assert statements
            last_statement = statements[-1]
            if last_statement.isStatementLoopContinue():
                if len(statements) == 1:
                    self.subnode_body.finalize()
                    self.clearChild('loop_body')
                    loop_body = None
                else:
                    last_statement.parent.replaceChild(last_statement, None)
                    last_statement.finalize()
                trace_collection.signalChange('new_statements', last_statement.getSourceReference(), "Removed useless terminal 'continue' as last statement of loop.")
            elif last_statement.isStatementLoopBreak():
                if not continue_collections and len(break_collections) == 1:
                    loop_body = loop_body.removeStatement(last_statement)
                    return (loop_body, 'new_statements', 'Removed useless loop with only a break at the end.')
        outer_trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    @staticmethod
    def getStatementNiceName():
        if False:
            i = 10
            return i + 15
        return 'loop statement'

class StatementLoopContinue(StatementBase):
    kind = 'STATEMENT_LOOP_CONTINUE'

    def __init__(self, source_ref):
        if False:
            return 10
        StatementBase.__init__(self, source_ref=source_ref)

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent

    @staticmethod
    def isStatementAborting():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def mayContinue():
        if False:
            while True:
                i = 10
        return True

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onLoopContinue()
        return (self, None, None)

    @staticmethod
    def getStatementNiceName():
        if False:
            print('Hello World!')
        return 'loop continue statement'

class StatementLoopBreak(StatementBase):
    kind = 'STATEMENT_LOOP_BREAK'

    def __init__(self, source_ref):
        if False:
            return 10
        StatementBase.__init__(self, source_ref=source_ref)

    def finalize(self):
        if False:
            while True:
                i = 10
        del self.parent

    @staticmethod
    def isStatementAborting():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def mayBreak():
        if False:
            for i in range(10):
                print('nop')
        return True

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onLoopBreak()
        return (self, None, None)

    @staticmethod
    def getStatementNiceName():
        if False:
            i = 10
            return i + 15
        return 'loop break statement'