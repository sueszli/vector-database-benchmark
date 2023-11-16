""" Nodes for classes and their creations.

The classes are are at the core of the language and have their complexities.

"""
from nuitka.PythonVersions import python_version
from .ChildrenHavingMixins import ChildrenExpressionBuiltinType3Mixin, ChildrenHavingMetaclassBasesMixin
from .ExpressionBases import ExpressionBase
from .ExpressionShapeMixins import ExpressionDictShapeExactMixin
from .IndicatorMixins import MarkNeedsAnnotationsMixin
from .LocalsScopes import getLocalsDictHandle
from .OutlineNodes import ExpressionOutlineFunctionBase

class ExpressionClassBodyBase(ExpressionOutlineFunctionBase):
    kind = 'EXPRESSION_CLASS_BODY'
    __slots__ = ('doc',)

    def __init__(self, provider, name, doc, source_ref):
        if False:
            while True:
                i = 10
        ExpressionOutlineFunctionBase.__init__(self, provider=provider, name=name, body=None, code_prefix='class', source_ref=source_ref)
        self.doc = doc
        self.locals_scope = getLocalsDictHandle('locals_%s_%d' % (self.getCodeName(), source_ref.getLineNumber()), self.locals_kind, self)

    @staticmethod
    def isExpressionClassBodyBase():
        if False:
            while True:
                i = 10
        return True

    def getDetails(self):
        if False:
            for i in range(10):
                print('nop')
        return {'name': self.getFunctionName(), 'provider': self.provider.getCodeName(), 'doc': self.doc, 'flags': self.flags}

    def getDetailsForDisplay(self):
        if False:
            return 10
        result = {'name': self.getFunctionName(), 'provider': self.provider.getCodeName(), 'flags': '' if self.flags is None else ','.join(sorted(self.flags))}
        if self.doc is not None:
            result['doc'] = self.doc
        return result

    @classmethod
    def fromXML(cls, provider, source_ref, **args):
        if False:
            for i in range(10):
                print('nop')
        return cls(provider=provider, source_ref=source_ref, **args)

    def getDoc(self):
        if False:
            for i in range(10):
                print('nop')
        return self.doc

    @staticmethod
    def isEarlyClosure():
        if False:
            for i in range(10):
                print('nop')
        return True

    def getVariableForClosure(self, variable_name):
        if False:
            while True:
                i = 10
        if variable_name == '__class__':
            if python_version < 768:
                return self.provider.getVariableForClosure('__class__')
            else:
                return ExpressionOutlineFunctionBase.getVariableForClosure(self, variable_name='__class__')
        else:
            result = self.provider.getVariableForClosure(variable_name)
            self.taken.add(result)
            return result

    @staticmethod
    def markAsDirectlyCalled():
        if False:
            return 10
        pass

    def getChildQualname(self, function_name):
        if False:
            print('Hello World!')
        return self.getFunctionQualname() + '.' + function_name

    @staticmethod
    def mayHaveSideEffects():
        if False:
            return 10
        return False

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return self.subnode_body.mayRaiseException(exception_type)

    def isUnoptimized(self):
        if False:
            print('Hello World!')
        return True

class ExpressionClassBodyP3(MarkNeedsAnnotationsMixin, ExpressionClassBodyBase):
    kind = 'EXPRESSION_CLASS_BODY_P3'
    __slots__ = ('needs_annotations_dict',)
    if python_version >= 832:
        __slots__ += ('qualname_setup',)
    locals_kind = 'python3_class'

    def __init__(self, provider, name, doc, source_ref):
        if False:
            return 10
        ExpressionClassBodyBase.__init__(self, provider=provider, name=name, doc=doc, source_ref=source_ref)
        MarkNeedsAnnotationsMixin.__init__(self)
        if python_version >= 832:
            self.qualname_setup = None

class ExpressionClassBodyP2(ExpressionDictShapeExactMixin, ExpressionClassBodyBase):
    kind = 'EXPRESSION_CLASS_BODY_P2'
    __slots__ = ()
    locals_kind = 'python2_class'

    def __init__(self, provider, name, doc, source_ref):
        if False:
            while True:
                i = 10
        ExpressionClassBodyBase.__init__(self, provider=provider, name=name, doc=doc, source_ref=source_ref)

class ExpressionSelectMetaclass(ChildrenHavingMetaclassBasesMixin, ExpressionBase):
    kind = 'EXPRESSION_SELECT_METACLASS'
    named_children = ('metaclass', 'bases')

    def __init__(self, metaclass, bases, source_ref):
        if False:
            i = 10
            return i + 15
        ChildrenHavingMetaclassBasesMixin.__init__(self, metaclass=metaclass, bases=bases)
        ExpressionBase.__init__(self, source_ref)

    def computeExpression(self, trace_collection):
        if False:
            return 10
        if self.subnode_bases.isExpressionConstantTupleEmptyRef():
            return (self.subnode_metaclass, 'new_expression', 'Metaclass selection without bases is trivial.')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

    def mayRaiseException(self, exception_type):
        if False:
            i = 10
            return i + 15
        return not self.subnode_bases.isExpressionConstantTupleEmptyRef()

class ExpressionBuiltinType3(ChildrenExpressionBuiltinType3Mixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_TYPE3'
    named_children = ('type_name', 'bases', 'dict_arg')

    def __init__(self, type_name, bases, dict_arg, source_ref):
        if False:
            while True:
                i = 10
        ChildrenExpressionBuiltinType3Mixin.__init__(self, type_name=type_name, bases=bases, dict_arg=dict_arg)
        ExpressionBase.__init__(self, source_ref)

    def _calculateMetaClass(self):
        if False:
            print('Hello World!')
        if not self.subnode_bases.isCompileTimeConstant():
            return None
        import ctypes
        ctypes.pythonapi._PyType_CalculateMetaclass.argtypes = (ctypes.py_object, ctypes.py_object)
        ctypes.pythonapi._PyType_CalculateMetaclass.restype = ctypes.py_object
        bases = self.subnode_bases.getCompileTimeConstant()
        return ctypes.pythonapi._PyType_CalculateMetaclass(type, bases)

    def mayRaiseException(self, exception_type):
        if False:
            while True:
                i = 10
        return True

    def computeExpression(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        if self.mayRaiseException(BaseException):
            trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)