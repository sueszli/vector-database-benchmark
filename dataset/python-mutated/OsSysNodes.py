""" Nodes the represent ways to access os and sys functions. """
import os
from .ConstantRefNodes import makeConstantRefNode
from .ExpressionBases import ExpressionNoSideEffectsMixin
from .HardImportNodesGenerated import ExpressionOsListdirCallBase, ExpressionOsPathAbspathCallBase, ExpressionOsPathBasenameCallBase, ExpressionOsPathExistsCallBase, ExpressionOsPathIsabsCallBase, ExpressionOsPathIsdirCallBase, ExpressionOsPathIsfileCallBase, ExpressionOsUnameCallBase

class ExpressionOsUnameCall(ExpressionNoSideEffectsMixin, ExpressionOsUnameCallBase):
    kind = 'EXPRESSION_OS_UNAME_CALL'

    @staticmethod
    def mayRaiseExceptionOperation():
        if False:
            for i in range(10):
                print('nop')
        return False

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionOsPathExistsCall(ExpressionOsPathExistsCallBase):
    kind = 'EXPRESSION_OS_PATH_EXISTS_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            i = 10
            return i + 15
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionOsPathIsfileCall(ExpressionOsPathIsfileCallBase):
    kind = 'EXPRESSION_OS_PATH_ISFILE_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            for i in range(10):
                print('nop')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionOsPathIsdirCall(ExpressionOsPathIsdirCallBase):
    kind = 'EXPRESSION_OS_PATH_ISDIR_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionOsPathBasenameCall(ExpressionOsPathBasenameCallBase):
    kind = 'EXPRESSION_OS_PATH_BASENAME_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            print('Hello World!')
        result = makeConstantRefNode(constant=os.path.basename(self.subnode_p.getCompileTimeConstant()), source_ref=self.source_ref)
        return (result, 'new_expression', "Compile time resolved 'os.path.basename' call.")

class ExpressionOsPathAbspathCall(ExpressionOsPathAbspathCallBase):
    kind = 'EXPRESSION_OS_PATH_ABSPATH_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            while True:
                i = 10
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)

class ExpressionOsPathIsabsCall(ExpressionOsPathIsabsCallBase):
    kind = 'EXPRESSION_OS_PATH_ISABS_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            i = 10
            return i + 15
        result = makeConstantRefNode(constant=os.path.isabs(self.subnode_s.getCompileTimeConstant()), source_ref=self.source_ref)
        return (result, 'new_expression', "Compile time resolved 'os.path.isabs' call.")

class ExpressionOsListdirCall(ExpressionOsListdirCallBase):
    kind = 'EXPRESSION_OS_LISTDIR_CALL'

    def replaceWithCompileTimeValue(self, trace_collection):
        if False:
            print('Hello World!')
        trace_collection.onExceptionRaiseExit(BaseException)
        return (self, None, None)