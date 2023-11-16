from ut_helpers_ir import exec_instruction
from miasm.expression.expression import ExprId, ExprInt, ExprCond, ExprOp
from miasm.jitter.csts import EXCEPT_DIV_BY_ZERO

class TestDivide(object):

    def test_div(self):
        if False:
            while True:
                i = 10
        'Test DIV execution'
        exec_instruction('DIV R0, R1', [(ExprId('R0', 32), ExprInt(128, 32)), (ExprId('R1', 32), ExprInt(0, 32)), (ExprId('HI', 32), ExprInt(0, 32)), (ExprId('LO', 32), ExprInt(0, 32))], [(ExprId('HI', 32), ExprInt(0, 32)), (ExprId('LO', 32), ExprInt(0, 32)), (ExprId('exception_flags', 32), ExprInt(EXCEPT_DIV_BY_ZERO, 32))])
        exec_instruction('DIV R0, R1', [(ExprId('R0', 32), ExprInt(-4, 32)), (ExprId('R1', 32), ExprInt(-2, 32))], [(ExprId('HI', 32), ExprInt(0, 32)), (ExprId('LO', 32), ExprInt(2, 32))])
        exec_instruction('DIV R0, R1', [(ExprId('R0', 32), ExprInt(-5, 32)), (ExprId('R1', 32), ExprInt(-2, 32))], [(ExprId('HI', 32), ExprInt(1, 32)), (ExprId('LO', 32), ExprInt(2, 32))])
        exec_instruction('DIV R0, R1', [(ExprId('R0', 32), ExprInt(4, 32)), (ExprId('R1', 32), ExprInt(2, 32))], [(ExprId('HI', 32), ExprInt(4294967292, 32)), (ExprId('LO', 32), ExprInt(0, 32))])
        exec_instruction('DIV R0, R1', [(ExprId('R0', 32), ExprInt(-5, 32)), (ExprId('R1', 32), ExprInt(2, 32))], [(ExprId('HI', 32), ExprInt(4294967295, 32)), (ExprId('LO', 32), ExprInt(4294967294, 32))])
        exec_instruction('DIV R0, R1', [(ExprId('R0', 32), ExprInt(5, 32)), (ExprId('R1', 32), ExprInt(-2, 32))], [(ExprId('HI', 32), ExprInt(4294967295, 32)), (ExprId('LO', 32), ExprInt(4294967294, 32))])

    def test_divu(self):
        if False:
            for i in range(10):
                print('nop')
        'Test DIVU execution'
        exec_instruction('DIVU R0, R1', [(ExprId('R0', 32), ExprInt(128, 32)), (ExprId('R1', 32), ExprInt(0, 32)), (ExprId('HI', 32), ExprInt(0, 32)), (ExprId('LO', 32), ExprInt(0, 32))], [(ExprId('HI', 32), ExprInt(0, 32)), (ExprId('LO', 32), ExprInt(0, 32)), (ExprId('exception_flags', 32), ExprInt(EXCEPT_DIV_BY_ZERO, 32))])
        exec_instruction('DIVU R0, R1', [(ExprId('R0', 32), ExprInt(128, 32)), (ExprId('R1', 32), ExprInt(2, 32))], [(ExprId('HI', 32), ExprInt(0, 32)), (ExprId('LO', 32), ExprInt(64, 32))])
        exec_instruction('DIVU R0, R1', [(ExprId('R0', 32), ExprInt(131, 32)), (ExprId('R1', 32), ExprInt(2, 32))], [(ExprId('HI', 32), ExprInt(1, 32)), (ExprId('LO', 32), ExprInt(65, 32))])
        exec_instruction('DIVU R0, R1', [(ExprId('R0', 32), ExprInt(2147483648, 32)), (ExprId('R1', 32), ExprInt(-1, 32))], [(ExprId('HI', 32), ExprInt(2147483648, 32)), (ExprId('LO', 32), ExprInt(0, 32))])