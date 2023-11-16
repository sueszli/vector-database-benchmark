""" Tree nodes for built-in references.

There is 2 major types of built-in references. One is the values from
built-ins, the other is built-in exceptions. They work differently and
mean different things, but they have similar origin, that is, access
to variables only ever read.

"""
from nuitka.Builtins import builtin_anon_names, builtin_exception_names, builtin_exception_values, builtin_names, builtin_type_names
from nuitka.Options import hasPythonFlagNoAsserts
from nuitka.PythonVersions import python_version
from nuitka.specs import BuiltinParameterSpecs
from .ConstantRefNodes import makeConstantRefNode
from .ExceptionNodes import ExpressionBuiltinMakeException, ExpressionBuiltinMakeExceptionImportError
from .ExpressionBases import CompileTimeConstantExpressionBase
from .shapes.BuiltinTypeShapes import tshape_exception_class

class ExpressionBuiltinRefBase(CompileTimeConstantExpressionBase):
    __slots__ = ('builtin_name',)

    def __init__(self, builtin_name, source_ref):
        if False:
            print('Hello World!')
        CompileTimeConstantExpressionBase.__init__(self, source_ref)
        self.builtin_name = builtin_name

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.parent

    def getDetails(self):
        if False:
            print('Hello World!')
        return {'builtin_name': self.builtin_name}

    def getBuiltinName(self):
        if False:
            i = 10
            return i + 15
        return self.builtin_name

    @staticmethod
    def isKnownToBeHashable():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getStrValue(self):
        if False:
            while True:
                i = 10
        return makeConstantRefNode(constant=str(self.getCompileTimeConstant()), user_provided=True, source_ref=self.source_ref)

def makeExpressionBuiltinTypeRef(builtin_name, source_ref):
    if False:
        while True:
            i = 10
    return makeConstantRefNode(constant=__builtins__[builtin_name], source_ref=source_ref)
quick_names = {'None': None, 'True': True, 'False': False}

def makeExpressionBuiltinRef(builtin_name, locals_scope, source_ref):
    if False:
        return 10
    assert builtin_name in builtin_names, builtin_name
    if builtin_name in quick_names:
        return makeConstantRefNode(constant=quick_names[builtin_name], source_ref=source_ref)
    elif builtin_name == '__debug__':
        return makeConstantRefNode(constant=not hasPythonFlagNoAsserts(), source_ref=source_ref)
    elif builtin_name in builtin_type_names:
        return makeExpressionBuiltinTypeRef(builtin_name=builtin_name, source_ref=source_ref)
    elif builtin_name in ('dir', 'eval', 'exec', 'execfile', 'locals', 'vars'):
        return ExpressionBuiltinWithContextRef(builtin_name=builtin_name, locals_scope=locals_scope, source_ref=source_ref)
    else:
        return ExpressionBuiltinRef(builtin_name=builtin_name, source_ref=source_ref)

class ExpressionBuiltinRef(ExpressionBuiltinRefBase):
    kind = 'EXPRESSION_BUILTIN_REF'
    __slots__ = ()
    locals_scope = None

    @staticmethod
    def isExpressionBuiltinRef():
        if False:
            print('Hello World!')
        return True

    def __init__(self, builtin_name, source_ref):
        if False:
            while True:
                i = 10
        ExpressionBuiltinRefBase.__init__(self, builtin_name=builtin_name, source_ref=source_ref)

    def getCompileTimeConstant(self):
        if False:
            while True:
                i = 10
        return __builtins__[self.builtin_name]

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            i = 10
            return i + 15
        from nuitka.optimizations.OptimizeBuiltinCalls import computeBuiltinCall
        trace_collection.onExceptionRaiseExit(BaseException)
        (new_node, tags, message) = computeBuiltinCall(builtin_name=self.builtin_name, call_node=call_node)
        if self.builtin_name in ('dir', 'eval', 'exec', 'execfile', 'locals', 'vars'):
            trace_collection.onLocalsUsage(locals_scope=self.getLocalsScope())
        return (new_node, tags, message)

    def computeExpressionCallViaVariable(self, call_node, variable_ref_node, call_args, call_kw, trace_collection):
        if False:
            print('Hello World!')
        return self.computeExpressionCall(call_node=call_node, call_args=call_args, call_kw=call_kw, trace_collection=trace_collection)

    @staticmethod
    def isKnownToBeIterable(count):
        if False:
            i = 10
            return i + 15
        return None

class ExpressionBuiltinWithContextRef(ExpressionBuiltinRef):
    """Same as ExpressionBuiltinRef, but with a context it refers to."""
    kind = 'EXPRESSION_BUILTIN_WITH_CONTEXT_REF'
    __slots__ = ('locals_scope',)

    def __init__(self, builtin_name, locals_scope, source_ref):
        if False:
            print('Hello World!')
        ExpressionBuiltinRef.__init__(self, builtin_name=builtin_name, source_ref=source_ref)
        self.locals_scope = locals_scope

    def getDetails(self):
        if False:
            i = 10
            return i + 15
        return {'builtin_name': self.builtin_name, 'locals_scope': self.locals_scope}

    def getLocalsScope(self):
        if False:
            for i in range(10):
                print('nop')
        return self.locals_scope

class ExpressionBuiltinAnonymousRef(ExpressionBuiltinRefBase):
    kind = 'EXPRESSION_BUILTIN_ANONYMOUS_REF'
    __slots__ = ()

    def __init__(self, builtin_name, source_ref):
        if False:
            return 10
        assert builtin_name in builtin_anon_names, (builtin_name, source_ref)
        ExpressionBuiltinRefBase.__init__(self, builtin_name=builtin_name, source_ref=source_ref)

    def getCompileTimeConstant(self):
        if False:
            for i in range(10):
                print('nop')
        return builtin_anon_names[self.builtin_name]

    def computeExpressionRaw(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        return (self, None, None)

class ExpressionBuiltinExceptionRef(ExpressionBuiltinRefBase):
    kind = 'EXPRESSION_BUILTIN_EXCEPTION_REF'
    __slots__ = ()

    def __init__(self, exception_name, source_ref):
        if False:
            i = 10
            return i + 15
        assert exception_name in builtin_exception_names, exception_name
        ExpressionBuiltinRefBase.__init__(self, builtin_name=exception_name, source_ref=source_ref)

    def getDetails(self):
        if False:
            while True:
                i = 10
        return {'exception_name': self.builtin_name}
    getExceptionName = ExpressionBuiltinRefBase.getBuiltinName

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_exception_class

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            return 10
        return False

    def getCompileTimeConstant(self):
        if False:
            while True:
                i = 10
        return builtin_exception_values[self.builtin_name]

    def computeExpressionRaw(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

    def computeExpressionCall(self, call_node, call_args, call_kw, trace_collection):
        if False:
            i = 10
            return i + 15
        exception_name = self.getExceptionName()

        def createBuiltinMakeException(args, name=None, path=None, source_ref=None):
            if False:
                for i in range(10):
                    print('nop')
            if exception_name == 'ImportError' and python_version >= 768:
                return ExpressionBuiltinMakeExceptionImportError(args=args, name=name, path=path, source_ref=source_ref)
            else:
                assert name is None
                assert path is None
                return ExpressionBuiltinMakeException(exception_name=exception_name, args=args, source_ref=source_ref)
        new_node = BuiltinParameterSpecs.extractBuiltinArgs(node=call_node, builtin_class=createBuiltinMakeException, builtin_spec=BuiltinParameterSpecs.makeBuiltinExceptionParameterSpec(exception_name=exception_name))
        assert new_node is not None
        return (new_node, 'new_expression', 'Detected built-in exception making.')

    def computeExpressionCallViaVariable(self, call_node, variable_ref_node, call_args, call_kw, trace_collection):
        if False:
            while True:
                i = 10
        return self.computeExpressionCall(call_node=call_node, call_args=call_args, call_kw=call_kw, trace_collection=trace_collection)