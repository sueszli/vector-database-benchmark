""" Nodes to inject C code into generated code. """
from .NodeBases import StatementBase

class StatementInjectCBase(StatementBase):
    __slots__ = ('c_code',)

    def __init__(self, c_code, source_ref):
        if False:
            for i in range(10):
                print('nop')
        StatementBase.__init__(self, source_ref=source_ref)
        self.c_code = c_code

    def finalize(self):
        if False:
            i = 10
            return i + 15
        del self.c_code

    def computeStatement(self, trace_collection):
        if False:
            print('Hello World!')
        return (self, None, None)

    @staticmethod
    def mayRaiseException(exception_type):
        if False:
            for i in range(10):
                print('nop')
        return False

class StatementInjectCCode(StatementInjectCBase):
    kind = 'STATEMENT_INJECT_C_CODE'

class StatementInjectCDecl(StatementInjectCBase):
    kind = 'STATEMENT_INJECT_C_DECL'
    __slots__ = ('c_code',)