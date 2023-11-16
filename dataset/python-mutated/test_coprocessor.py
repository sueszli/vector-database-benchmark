from ut_helpers_ir import exec_instruction
from miasm.expression.expression import ExprId, ExprMem, ExprInt

class TestCoprocessor(object):

    def test_swcp(self):
        if False:
            print('Hello World!')
        'Test SWCP execution'
        exec_instruction('SWCP C1, (R2)', [(ExprId('C1', 32), ExprInt(671551504, 32)), (ExprId('R2', 32), ExprInt(17, 32))], [(ExprMem(ExprInt(16, 32), 32), ExprInt(671551504, 32))])
        exec_instruction('SWCP C10, 0xF800(R2)', [(ExprId('C10', 32), ExprInt(43975, 32)), (ExprId('R2', 32), ExprInt(17, 32))], [(ExprMem(ExprInt(4294965264, 32), 32), ExprInt(43975, 32))])

    def test_lwcp(self):
        if False:
            return 10
        'Test LWCP execution'
        exec_instruction('LWCP C1, (R2)', [(ExprId('R2', 32), ExprInt(17, 32)), (ExprMem(ExprInt(16, 32), 32), ExprInt(43981, 32))], [(ExprId('C1', 32), ExprInt(43981, 32))])
        exec_instruction('LWCP C9, 0xF000(R2)', [(ExprId('R2', 32), ExprInt(66, 32)), (ExprMem(ExprInt(4294963264, 32), 32), ExprInt(16, 32))], [(ExprId('C9', 32), ExprInt(16, 32))])

    def test_smcp(self):
        if False:
            while True:
                i = 10
        'Test SMCP execution'
        exec_instruction('SMCP C1, (R2)', [(ExprId('C1', 32), ExprInt(671551504, 32)), (ExprId('R2', 32), ExprInt(23, 32))], [(ExprMem(ExprInt(16, 32), 32), ExprInt(671551504, 32))])
        exec_instruction('SMCP C10, 0xF800(R2)', [(ExprId('C10', 32), ExprInt(43975, 32)), (ExprId('R2', 32), ExprInt(23, 32))], [(ExprMem(ExprInt(4294965264, 32), 32), ExprInt(43975, 32))])

    def test_lmcp(self):
        if False:
            print('Hello World!')
        'Test LMCP execution'
        exec_instruction('LMCP C1, (R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(16, 32), 32), ExprInt(43981, 32))], [(ExprId('C1', 32), ExprInt(43981, 32))])
        exec_instruction('LMCP C9, 0xF000(R2)', [(ExprId('R2', 32), ExprInt(23, 32)), (ExprMem(ExprInt(4294963216, 32), 32), ExprInt(16, 32))], [(ExprId('C9', 32), ExprInt(16, 32))])

    def test_swcpi(self):
        if False:
            for i in range(10):
                print('nop')
        'Test SWCPI execution'
        exec_instruction('SWCPI C1, (R2+)', [(ExprId('C1', 32), ExprInt(671551504, 32)), (ExprId('R2', 32), ExprInt(17, 32))], [(ExprMem(ExprInt(16, 32), 32), ExprInt(671551504, 32)), (ExprId('R2', 32), ExprInt(21, 32))])

    def test_lwcpi(self):
        if False:
            while True:
                i = 10
        'Test LWCPI execution'
        exec_instruction('LWCPI C1, (R2+)', [(ExprId('R2', 32), ExprInt(17, 32)), (ExprMem(ExprInt(16, 32), 32), ExprInt(43981, 32))], [(ExprId('C1', 32), ExprInt(43981, 32)), (ExprId('R2', 32), ExprInt(21, 32))])

    def test_smcpi(self):
        if False:
            while True:
                i = 10
        'Test SMCPI execution'
        exec_instruction('SMCPI C1, (R2+)', [(ExprId('C1', 32), ExprInt(671551504, 32)), (ExprId('R2', 32), ExprInt(23, 32))], [(ExprMem(ExprInt(16, 32), 32), ExprInt(671551504, 32)), (ExprId('R2', 32), ExprInt(31, 32))])

    def test_lmcpi(self):
        if False:
            while True:
                i = 10
        'Test LMCPI execution'
        exec_instruction('LMCPI C1, (R2+)', [(ExprId('R2', 32), ExprInt(17, 32)), (ExprMem(ExprInt(16, 32), 32), ExprInt(43981, 32))], [(ExprId('C1', 32), ExprInt(43981, 32)), (ExprId('R2', 32), ExprInt(25, 32))])