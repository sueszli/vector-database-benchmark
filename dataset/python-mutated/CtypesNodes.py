""" Nodes for all things "ctypes" stdlib module.

"""
from .HardImportNodesGenerated import ExpressionCtypesCdllBefore38CallBase, ExpressionCtypesCdllSince38CallBase

class ExpressionCtypesCdllSince38Call(ExpressionCtypesCdllSince38CallBase):
    """Function reference ctypes.CDLL"""
    kind = 'EXPRESSION_CTYPES_CDLL_SINCE38_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionCtypesCdllBefore38Call(ExpressionCtypesCdllBefore38CallBase):
    """Function reference ctypes.CDLL"""
    kind = 'EXPRESSION_CTYPES_CDLL_BEFORE38_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)