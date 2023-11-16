""" Node the calls to the 'open' built-in.

This is a rather two sided beast, as it may be read or write. And we would like to be able
to track it, so we can include files into the executable, or write more efficiently.
"""
from .ChildrenHavingMixins import ChildrenExpressionBuiltinOpenP2Mixin, ChildrenExpressionBuiltinOpenP3Mixin
from .ExpressionBases import ExpressionBase
from .shapes.BuiltinTypeShapes import tshape_file

class ExpressionBuiltinOpenMixin(object):
    __slots__ = ()

    @staticmethod
    def getTypeShape():
        if False:
            i = 10
            return i + 15
        return tshape_file

    def computeExpression(self, trace_collection):
        if False:
            return 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionBuiltinOpenP2(ExpressionBuiltinOpenMixin, ChildrenExpressionBuiltinOpenP2Mixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_OPEN_P2'
    python_version_spec = '< 0x300'
    named_children = ('filename', 'mode|optional', 'buffering|optional')

    def __init__(self, filename, mode, buffering, source_ref):
        if False:
            for i in range(10):
                print('nop')
        ChildrenExpressionBuiltinOpenP2Mixin.__init__(self, filename=filename, mode=mode, buffering=buffering)
        ExpressionBase.__init__(self, source_ref)

class ExpressionBuiltinOpenP3(ExpressionBuiltinOpenMixin, ChildrenExpressionBuiltinOpenP3Mixin, ExpressionBase):
    kind = 'EXPRESSION_BUILTIN_OPEN_P3'
    python_version_spec = '>= 0x300'
    named_children = ('filename', 'mode|optional', 'buffering|optional', 'encoding|optional', 'errors|optional', 'newline|optional', 'closefd|optional', 'opener|optional')

    def __init__(self, filename, mode, buffering, encoding, errors, newline, closefd, opener, source_ref):
        if False:
            return 10
        ChildrenExpressionBuiltinOpenP3Mixin.__init__(self, filename=filename, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener)
        ExpressionBase.__init__(self, source_ref)

def makeExpressionBuiltinsOpenCall(filename, mode, buffering, encoding, errors, newline, closefd, opener, source_ref):
    if False:
        for i in range(10):
            print('nop')
    'Function reference ctypes.CDLL'
    assert str is not bytes
    return ExpressionBuiltinOpenP3(filename=filename, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline, closefd=closefd, opener=opener, source_ref=source_ref)