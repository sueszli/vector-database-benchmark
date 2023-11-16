from ut_helpers_ir import exec_instruction
from miasm.expression.expression import ExprId, ExprInt, ExprCond, ExprOp

class TestLdz(object):

    def test_ldz(self):
        if False:
            for i in range(10):
                print('nop')
        'Test LDZ execution'
        exec_instruction('LDZ R0, R1', [(ExprId('R1', 32), ExprInt(2147483648, 32))], [(ExprId('R0', 32), ExprInt(0, 32))])
        exec_instruction('LDZ R0, R1', [(ExprId('R1', 32), ExprInt(1073741824, 32))], [(ExprId('R0', 32), ExprInt(1, 32))])
        exec_instruction('LDZ R0, R1', [(ExprId('R1', 32), ExprInt(15, 32))], [(ExprId('R0', 32), ExprInt(28, 32))])
        exec_instruction('LDZ R0, R1', [(ExprId('R1', 32), ExprInt(4, 32))], [(ExprId('R0', 32), ExprInt(29, 32))])
        exec_instruction('LDZ R0, R1', [(ExprId('R1', 32), ExprInt(2, 32))], [(ExprId('R0', 32), ExprInt(30, 32))])
        exec_instruction('LDZ R0, R1', [(ExprId('R1', 32), ExprInt(1, 32))], [(ExprId('R0', 32), ExprInt(31, 32))])
        exec_instruction('LDZ R0, R1', [(ExprId('R1', 32), ExprInt(0, 32))], [(ExprId('R0', 32), ExprInt(32, 32))])