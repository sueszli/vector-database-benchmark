from ut_helpers_ir import exec_instruction
from miasm.expression.expression import ExprId, ExprMem, ExprInt

class TestLoadStore(object):

    def test_sb(self):
        if False:
            i = 10
            return i + 15
        'Test SB execution'
        exec_instruction('SB R1, (R2)', [(ExprId('R1', 32), ExprInt(40, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(16, 32), 8), ExprInt(40, 8))])
        exec_instruction('SB R1, 0x18(R2)', [(ExprId('R1', 32), ExprInt(43975, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(40, 32), 8), ExprInt(199, 8))])
        exec_instruction('SB R10, 0xF800(R2)', [(ExprId('R10', 32), ExprInt(43975, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(4294965264, 32), 8), ExprInt(199, 8))])

    def test_sh(self):
        if False:
            while True:
                i = 10
        'Test SH execution'
        exec_instruction('SH R1, (R2)', [(ExprId('R1', 32), ExprInt(10247, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(16, 32), 16), ExprInt(10247, 16))])
        exec_instruction('SH R1, 0x18(R2)', [(ExprId('R1', 32), ExprInt(43975, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(40, 32), 16), ExprInt(43975, 16))])
        exec_instruction('SH R10, 0xF800(R2)', [(ExprId('R10', 32), ExprInt(43975, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(4294965264, 32), 16), ExprInt(43975, 16))])

    def test_sw(self):
        if False:
            for i in range(10):
                print('nop')
        'Test SW execution'
        exec_instruction('SW R1, (R2)', [(ExprId('R1', 32), ExprInt(671551504, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(16, 32), 32), ExprInt(671551504, 32))])
        exec_instruction('SW R1, 4(SP)', [(ExprId('R1', 32), ExprInt(671551504, 32)), (ExprId('SP', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(20, 32), 32), ExprInt(671551504, 32))])
        exec_instruction('SW R1, 12(TP)', [(ExprId('R1', 32), ExprInt(671551504, 32)), (ExprId('TP', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(28, 32), 32), ExprInt(671551504, 32))])
        exec_instruction('SW R10, 0xF800(R2)', [(ExprId('R10', 32), ExprInt(43975, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprMem(ExprInt(4294965264, 32), 32), ExprInt(43975, 32))])
        exec_instruction('SW R10, (0x1010)', [(ExprId('R10', 32), ExprInt(43975, 32))], [(ExprMem(ExprInt(4112, 32), 32), ExprInt(43975, 32))])

    def test_lb(self):
        if False:
            print('Hello World!')
        'Test LB executon'
        exec_instruction('LB R1, (R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(16, 32), 8), ExprInt(240, 8))], [(ExprId('R1', 32), ExprInt(4294967280, 32))])
        exec_instruction('LB R7, 0x3(TP)', [(ExprId('TP', 32), ExprInt(16, 32)), (ExprMem(ExprInt(19, 32), 8), ExprInt(240, 8))], [(ExprId('R7', 32), ExprInt(4294967280, 32))])
        exec_instruction('LB R10, 0xF800(R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(4294965264, 32), 8), ExprInt(4, 8))], [(ExprId('R10', 32), ExprInt(4, 32))])
        exec_instruction('LB R10, 0xF800(R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(4294965264, 32), 8), ExprInt(254, 8))], [(ExprId('R10', 32), ExprInt(4294967294, 32))])

    def test_lh(self):
        if False:
            i = 10
            return i + 15
        'Test lh execution'
        exec_instruction('LH R1, (R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(16, 32), 16), ExprInt(62743, 16))], [(ExprId('R1', 32), ExprInt(4294964503, 32))])
        exec_instruction('LH R1, 0x18(R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(40, 32), 16), ExprInt(62743, 16))], [(ExprId('R1', 32), ExprInt(4294964503, 32))])
        exec_instruction('LH R9, 0xF000(R2)', [(ExprId('R2', 32), ExprInt(66, 32)), (ExprMem(ExprInt(4294963266, 32), 16), ExprInt(16, 16))], [(ExprId('R9', 32), ExprInt(16, 32))])
        exec_instruction('LH R9, 0xF000(R2)', [(ExprId('R2', 32), ExprInt(66, 32)), (ExprMem(ExprInt(4294963266, 32), 16), ExprInt(43981, 16))], [(ExprId('R9', 32), ExprInt(4294945741, 32))])

    def test_lw(self):
        if False:
            while True:
                i = 10
        'Test SW execution'
        exec_instruction('LW R1, (R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(16, 32), 32), ExprInt(43981, 32))], [(ExprId('R1', 32), ExprInt(43981, 32))])
        exec_instruction('LW R1, 0x18(SP)', [(ExprId('SP', 32), ExprInt(16, 32)), (ExprMem(ExprInt(40, 32), 32), ExprInt(19088743, 32))], [(ExprId('R1', 32), ExprInt(19088743, 32))])
        exec_instruction('LW R1, 0x18(TP)', [(ExprId('TP', 32), ExprInt(16, 32)), (ExprMem(ExprInt(40, 32), 32), ExprInt(4112, 32))], [(ExprId('R1', 32), ExprInt(4112, 32))])
        exec_instruction('LW R9, 0xF000(R2)', [(ExprId('R2', 32), ExprInt(66, 32)), (ExprMem(ExprInt(4294963264, 32), 32), ExprInt(16, 32))], [(ExprId('R9', 32), ExprInt(16, 32))])
        exec_instruction('LW R10, (0x1010)', [(ExprMem(ExprInt(4112, 32), 32), ExprInt(43975, 32))], [(ExprId('R10', 32), ExprInt(43975, 32))])

    def test_lbu(self):
        if False:
            print('Hello World!')
        'Test LBU execution'
        exec_instruction('LBU R1, (R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(16, 32), 8), ExprInt(10, 8))], [(ExprId('R1', 32), ExprInt(10, 32))])
        exec_instruction('LBU R1, 0x22(R3)', [(ExprId('R3', 32), ExprInt(16, 32)), (ExprMem(ExprInt(50, 32), 8), ExprInt(10, 8))], [(ExprId('R1', 32), ExprInt(10, 32))])
        exec_instruction('LBU R10, 0xF000(R2)', [(ExprId('R2', 32), ExprInt(66, 32)), (ExprMem(ExprInt(4294963266, 32), 32), ExprInt(16, 32))], [(ExprId('R10', 32), ExprInt(16, 32))])

    def test_lhu(self):
        if False:
            print('Hello World!')
        'Test LHU execution'
        exec_instruction('LHU R1, (R2)', [(ExprId('R2', 32), ExprInt(16, 32)), (ExprMem(ExprInt(16, 32), 16), ExprInt(239, 16))], [(ExprId('R1', 32), ExprInt(239, 32))])
        exec_instruction('LHU R1, 0x22(R3)', [(ExprId('R3', 32), ExprInt(16, 32)), (ExprMem(ExprInt(50, 32), 16), ExprInt(65244, 16))], [(ExprId('R1', 32), ExprInt(65244, 32))])
        exec_instruction('LHU R10, 0xF000(R2)', [(ExprId('R2', 32), ExprInt(66, 32)), (ExprMem(ExprInt(4294963266, 32), 16), ExprInt(4660, 16))], [(ExprId('R10', 32), ExprInt(4660, 32))])