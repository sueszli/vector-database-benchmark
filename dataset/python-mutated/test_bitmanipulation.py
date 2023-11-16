from ut_helpers_ir import exec_instruction
from miasm.expression.expression import ExprId, ExprInt, ExprMem

class TestBitManipulation(object):

    def test_bsetm(self):
        if False:
            while True:
                i = 10
        'Test BSETM execution'
        exec_instruction('BSETM (R1), 1', [(ExprId('R1', 32), ExprInt(40, 32)), (ExprMem(ExprInt(40, 32), 8), ExprInt(1, 8))], [(ExprMem(ExprInt(40, 32), 8), ExprInt(3, 8))])

    def test_bclrm(self):
        if False:
            while True:
                i = 10
        'Test BCLRM execution'
        exec_instruction('BCLRM (R1), 1', [(ExprId('R1', 32), ExprInt(40, 32)), (ExprMem(ExprInt(40, 32), 8), ExprInt(3, 8))], [(ExprMem(ExprInt(40, 32), 8), ExprInt(1, 8))])

    def test_bnotm(self):
        if False:
            return 10
        'Test BNOTM execution'
        exec_instruction('BNOTM (R1), 1', [(ExprId('R1', 32), ExprInt(40, 32)), (ExprMem(ExprInt(40, 32), 8), ExprInt(1, 8))], [(ExprMem(ExprInt(40, 32), 8), ExprInt(3, 8))])

    def test_btstm(self):
        if False:
            while True:
                i = 10
        'Test BTSTM execution'
        exec_instruction('BTSTM R0, (R1), 1', [(ExprId('R1', 32), ExprInt(40, 32)), (ExprMem(ExprInt(40, 32), 8), ExprInt(2, 8))], [(ExprId('R0', 32), ExprInt(2, 32))])

    def test_tas(self):
        if False:
            i = 10
            return i + 15
        'Test TAS execution'
        exec_instruction('TAS R0, (R1)', [(ExprId('R1', 32), ExprInt(40, 32)), (ExprMem(ExprInt(40, 32), 8), ExprInt(2, 8))], [(ExprId('R0', 32), ExprInt(2, 32)), (ExprMem(ExprInt(40, 32), 8), ExprInt(1, 8))])