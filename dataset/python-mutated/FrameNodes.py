""" Frame nodes.

The frame attaches name and other frame properties to a scope, where it is
optional. For use in tracebacks, their created frame objects, potentially
cached are essential.

Otherwise, they are similar to statement sequences, so they inherit from
them.

"""
from abc import abstractmethod
from nuitka.PythonVersions import python_version
from .CodeObjectSpecs import CodeObjectSpec
from .FutureSpecs import fromFlags
from .StatementNodes import StatementsSequence

def checkFrameStatements(value):
    if False:
        while True:
            i = 10
    'Check that frames statements list value proper.\n\n    Must not be None, must not contain None, may be empty though.\n    '
    assert value is not None
    assert None not in value
    return tuple(value)

class StatementsFrameBase(StatementsSequence):
    checkers = {'statements': checkFrameStatements}
    __slots__ = ('code_object', 'needs_frame_exception_preserve')

    def __init__(self, statements, code_object, source_ref):
        if False:
            i = 10
            return i + 15
        StatementsSequence.__init__(self, statements=statements, source_ref=source_ref)
        self.code_object = code_object
        self.needs_frame_exception_preserve = False

    def isStatementsFrame(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def getDetails(self):
        if False:
            print('Hello World!')
        result = {'code_object': self.code_object}
        result.update(StatementsSequence.getDetails(self))
        return result

    def getDetailsForDisplay(self):
        if False:
            for i in range(10):
                print('nop')
        result = StatementsSequence.getDetails(self)
        result.update()
        result.update(self.code_object.getDetails())
        return result

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            for i in range(10):
                print('nop')
        code_object_args = {}
        other_args = {}
        for (key, value) in args.items():
            if key.startswith('co_'):
                code_object_args[key] = value
            elif key == 'code_flags':
                code_object_args['future_spec'] = fromFlags(args['code_flags'])
            else:
                other_args[key] = value
        code_object = CodeObjectSpec(**code_object_args)
        return cls(code_object=code_object, source_ref=source_ref, **other_args)

    def getGuardMode(self):
        if False:
            for i in range(10):
                print('nop')
        provider = self.getParentVariableProvider()
        while provider.isExpressionClassBodyBase():
            provider = provider.getParentVariableProvider()
        if provider.isCompiledPythonModule():
            return 'once'
        else:
            return 'full'
        return self.guard_mode

    @staticmethod
    def needsExceptionFramePreservation():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getVarNames(self):
        if False:
            while True:
                i = 10
        return self.code_object.getVarNames()

    def updateLocalNames(self):
        if False:
            i = 10
            return i + 15
        'For use during variable closure phase. Finalize attributes.'
        provider = self.getParentVariableProvider()
        if not provider.isCompiledPythonModule():
            if provider.isExpressionGeneratorObjectBody() or provider.isExpressionCoroutineObjectBody() or provider.isExpressionAsyncgenObjectBody():
                closure_provider = provider.getParentVariableProvider()
            else:
                closure_provider = provider
            if closure_provider.isExpressionFunctionBody():
                closure_variables = closure_provider.getClosureVariables()
            else:
                closure_variables = ()
            self.code_object.updateLocalNames([variable.getName() for variable in provider.getLocalVariables()], [variable.getName() for variable in closure_variables if variable.getOwner() is not closure_provider])
        entry_point = provider.getEntryPoint()
        is_optimized = not entry_point.isCompiledPythonModule() and (not entry_point.isExpressionClassBodyBase()) and (not entry_point.isUnoptimized())
        self.code_object.setFlagIsOptimizedValue(is_optimized)
        new_locals = not provider.isCompiledPythonModule() and (python_version < 832 or (not provider.isExpressionClassBodyBase() and (not provider.isUnoptimized())))
        self.code_object.setFlagNewLocalsValue(new_locals)

    def markAsFrameExceptionPreserving(self):
        if False:
            i = 10
            return i + 15
        self.needs_frame_exception_preserve = True

    def needsFrameExceptionPreserving(self):
        if False:
            i = 10
            return i + 15
        return self.needs_frame_exception_preserve

    def getCodeObject(self):
        if False:
            i = 10
            return i + 15
        return self.code_object

    def computeStatementsSequence(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        new_statements = []
        statements = self.subnode_statements
        for (count, statement) in enumerate(statements):
            if statement.isStatementsFrame():
                new_statement = statement.computeStatementsSequence(trace_collection=trace_collection)
            else:
                new_statement = trace_collection.onStatement(statement=statement)
            if new_statement is not None:
                if new_statement.isStatementsSequence() and (not new_statement.isStatementsFrame()):
                    new_statements.extend(new_statement.subnode_statements)
                else:
                    new_statements.append(new_statement)
                if statement is not statements[-1] and new_statement.isStatementAborting():
                    trace_collection.signalChange('new_statements', statements[count + 1].getSourceReference(), 'Removed dead statements.')
                    break
        if not new_statements:
            trace_collection.signalChange('new_statements', self.source_ref, "Removed empty frame object of '%s'." % self.code_object.getCodeObjectName())
            return None
        new_statements_tuple = tuple(new_statements)
        if statements != new_statements_tuple:
            self.setChildStatements(new_statements_tuple)
            return self
        outside_pre = []
        while new_statements and (not new_statements[0].needsFrame()):
            outside_pre.append(new_statements[0])
            del new_statements[0]
        outside_post = []
        while new_statements and (not new_statements[-1].needsFrame()):
            outside_post.insert(0, new_statements[-1])
            del new_statements[-1]
        if outside_pre or outside_post:
            from .NodeMakingHelpers import makeStatementsSequenceReplacementNode
            if new_statements:
                self.setChildStatements(tuple(new_statements))
                return makeStatementsSequenceReplacementNode(statements=outside_pre + [self] + outside_post, node=self)
            else:
                trace_collection.signalChange('new_statements', self.source_ref, "Removed useless frame object of '%s'." % self.code_object.getCodeObjectName())
                return makeStatementsSequenceReplacementNode(statements=outside_pre + outside_post, node=self)
        else:
            if statements != new_statements:
                self.setChildStatements(tuple(new_statements))
            return self

    @abstractmethod
    def hasStructureMember(self):
        if False:
            print('Hello World!')
        'Does the frame have a structure associated, like e.g. generator objects need.'

    def getStructureMember(self):
        if False:
            i = 10
            return i + 15
        'Get the frame structure member code name, generator, coroutine, asyncgen.'
        assert not self.hasStructureMember()
        return None

class StatementsFrameModule(StatementsFrameBase):
    kind = 'STATEMENTS_FRAME_MODULE'

    def __init__(self, statements, code_object, source_ref):
        if False:
            return 10
        StatementsFrameBase.__init__(self, statements=statements, code_object=code_object, source_ref=source_ref)

    @staticmethod
    def hasStructureMember():
        if False:
            return 10
        return False

class StatementsFrameFunction(StatementsFrameBase):
    kind = 'STATEMENTS_FRAME_FUNCTION'

    def __init__(self, statements, code_object, source_ref):
        if False:
            i = 10
            return i + 15
        StatementsFrameBase.__init__(self, statements=statements, code_object=code_object, source_ref=source_ref)

    @staticmethod
    def hasStructureMember():
        if False:
            while True:
                i = 10
        return False

class StatementsFrameClass(StatementsFrameBase):
    kind = 'STATEMENTS_FRAME_CLASS'
    __slots__ = ('locals_scope',)

    def __init__(self, statements, code_object, locals_scope, source_ref):
        if False:
            print('Hello World!')
        StatementsFrameBase.__init__(self, statements=statements, code_object=code_object, source_ref=source_ref)
        self.locals_scope = locals_scope

    @staticmethod
    def hasStructureMember():
        if False:
            for i in range(10):
                print('nop')
        return False

    def getLocalsScope(self):
        if False:
            print('Hello World!')
        return self.locals_scope

class StatementsFrameGeneratorBase(StatementsFrameBase):

    def __init__(self, statements, code_object, source_ref):
        if False:
            print('Hello World!')
        StatementsFrameBase.__init__(self, statements=statements, code_object=code_object, source_ref=source_ref)

    @staticmethod
    def getGuardMode():
        if False:
            while True:
                i = 10
        return 'generator'

    @staticmethod
    def hasStructureMember():
        if False:
            while True:
                i = 10
        return True

class StatementsFrameGenerator(StatementsFrameGeneratorBase):
    kind = 'STATEMENTS_FRAME_GENERATOR'
    if python_version < 768:

        @staticmethod
        def needsExceptionFramePreservation():
            if False:
                print('Hello World!')
            return False

    @staticmethod
    def getStructureMember():
        if False:
            return 10
        return 'generator'

class StatementsFrameCoroutine(StatementsFrameGeneratorBase):
    kind = 'STATEMENTS_FRAME_COROUTINE'
    python_version_spec = '>= 0x350'

    @staticmethod
    def getStructureMember():
        if False:
            print('Hello World!')
        return 'coroutine'

class StatementsFrameAsyncgen(StatementsFrameGeneratorBase):
    kind = 'STATEMENTS_FRAME_ASYNCGEN'
    python_version_spec = '>= 0x360'

    @staticmethod
    def getStructureMember():
        if False:
            return 10
        return 'asyncgen'