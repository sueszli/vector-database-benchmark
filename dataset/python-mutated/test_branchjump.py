from ut_helpers_ir import exec_instruction
from miasm.expression.expression import ExprId, ExprInt

class TestBranchJump(object):

    def test_bra(self):
        if False:
            while True:
                i = 10
        'Test BRA execution'
        exec_instruction('BRA 0x28', [], [(ExprId('PC', 32), ExprInt(40, 32))])
        exec_instruction('BRA 0x800', [], [(ExprId('PC', 32), ExprInt(4294965248, 32))])
        exec_instruction('BRA 0x28', [], [(ExprId('PC', 32), ExprInt(4136, 32))], offset=4096)

    def test_beqz(self):
        if False:
            print('Hello World!')
        'Test BEQZ execution'
        exec_instruction('BEQZ R1, 0x10', [(ExprId('R1', 32), ExprInt(0, 32))], [(ExprId('PC', 32), ExprInt(32, 32))], offset=16)
        exec_instruction('BEQZ R1, 0x10', [(ExprId('R1', 32), ExprInt(1, 32))], [(ExprId('PC', 32), ExprInt(2, 32))])
        exec_instruction('BEQZ R1, 0x80', [(ExprId('R1', 32), ExprInt(0, 32))], [(ExprId('PC', 32), ExprInt(4294967184, 32))], offset=16)

    def test_bnez(self):
        if False:
            while True:
                i = 10
        'Test BNEZ execution'
        exec_instruction('BNEZ R1, 0x10', [(ExprId('R1', 32), ExprInt(0, 32))], [(ExprId('PC', 32), ExprInt(2, 32))])
        exec_instruction('BNEZ R1, 0x10', [(ExprId('R1', 32), ExprInt(1, 32))], [(ExprId('PC', 32), ExprInt(32, 32))], offset=16)
        exec_instruction('BNEZ R1, 0x80', [(ExprId('R1', 32), ExprInt(0, 32))], [(ExprId('PC', 32), ExprInt(2, 32))])

    def test_beqi(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BEQI execution'
        exec_instruction('BEQI R1, 0x8, 0x28', [(ExprId('R1', 32), ExprInt(0, 32))], [(ExprId('PC', 32), ExprInt(4, 32))])
        exec_instruction('BEQI R1, 0x1, 0x28', [(ExprId('R1', 32), ExprInt(1, 32))], [(ExprId('PC', 32), ExprInt(56, 32))], offset=16)
        exec_instruction('BEQI R1, 0x6, 0x10000', [(ExprId('R1', 32), ExprInt(6, 32))], [(ExprId('PC', 32), ExprInt(4294901776, 32))], offset=16)

    def test_bnei(self):
        if False:
            return 10
        'Test BNEI execution'
        exec_instruction('BNEI R1, 0x5, 0x28', [(ExprId('R1', 32), ExprInt(0, 32))], [(ExprId('PC', 32), ExprInt(56, 32))], offset=16)
        exec_instruction('BNEI R1, 0x7, 0xFF00', [(ExprId('R1', 32), ExprInt(7, 32)), (ExprId('PC', 32), ExprInt(1, 32))], [(ExprId('PC', 32), ExprInt(4, 32))])

    def test_blti(self):
        if False:
            print('Hello World!')
        'Test BLTI execution'
        exec_instruction('BLTI R1, 0x5, 0x10000', [(ExprId('R1', 32), ExprInt(16, 32))], [(ExprId('PC', 32), ExprInt(20, 32))], offset=16)
        exec_instruction('BLTI R1, 0x5, 0x10000', [(ExprId('R1', 32), ExprInt(1, 32))], [(ExprId('PC', 32), ExprInt(4294901776, 32))], offset=16)

    def test_bgei(self):
        if False:
            i = 10
            return i + 15
        'Test BGEI execution'
        exec_instruction('BGEI R1, 0x5, 0x10000', [(ExprId('R1', 32), ExprInt(16, 32))], [(ExprId('PC', 32), ExprInt(4294901776, 32))], offset=16)
        exec_instruction('BGEI R1, 0x5, 0x10000', [(ExprId('R1', 32), ExprInt(1, 32))], [(ExprId('PC', 32), ExprInt(20, 32))], offset=16)
        exec_instruction('BGEI R1, 0x5, 0x10000', [(ExprId('R1', 32), ExprInt(5, 32))], [(ExprId('PC', 32), ExprInt(4294901776, 32))], offset=16)

    def test_beq(self):
        if False:
            return 10
        'Test BEQ execution'
        exec_instruction('BEQ R1, R2, 0x10000', [(ExprId('R1', 32), ExprInt(16, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprId('PC', 32), ExprInt(4294901776, 32))], offset=16)
        exec_instruction('BEQ R1, R2, 0x8000', [(ExprId('R1', 32), ExprInt(9, 32)), (ExprId('R2', 32), ExprInt(16, 32)), (ExprId('PC', 32), ExprInt(16, 32))], [(ExprId('PC', 32), ExprInt(4, 32))])

    def test_bne(self):
        if False:
            return 10
        'Test BNE execution'
        exec_instruction('BNE R1, R2, 0x8000', [(ExprId('R1', 32), ExprInt(16, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprId('PC', 32), ExprInt(4, 32))])
        exec_instruction('BNE R1, R2, 0x8000', [(ExprId('R1', 32), ExprInt(9, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprId('PC', 32), ExprInt(32784, 32))], offset=16)
        exec_instruction('BNE R1, R2, 0x10000', [(ExprId('R1', 32), ExprInt(9, 32)), (ExprId('R2', 32), ExprInt(16, 32))], [(ExprId('PC', 32), ExprInt(4294901776, 32))], offset=16)

    def test_bsr(self):
        if False:
            while True:
                i = 10
        'Test BSR execution'
        exec_instruction('BSR 0x800', [(ExprId('PC', 32), ExprInt(2, 32))], [(ExprId('PC', 32), ExprInt(4294965248, 32)), (ExprId('LP', 32), ExprInt(2, 32))], index=0)
        exec_instruction('BSR 0x101015', [(ExprId('PC', 32), ExprInt(4, 32))], [(ExprId('PC', 32), ExprInt(1052692, 32)), (ExprId('LP', 32), ExprInt(4, 32))], index=1)

    def test_jmp(self):
        if False:
            i = 10
            return i + 15
        'Test JMP execution'
        exec_instruction('JMP R1', [(ExprId('R1', 32), ExprInt(1052693, 32))], [(ExprId('PC', 32), ExprInt(1052693, 32))])
        exec_instruction('JMP 0x2807', [(ExprId('PC', 32), ExprInt(0, 32))], [(ExprId('PC', 32), ExprInt(10246, 32))], offset=66)
        exec_instruction('JMP 0x2807', [(ExprId('PC', 32), ExprInt(2952790016, 32))], [(ExprId('PC', 32), ExprInt(2952800262, 32))], offset=2952790016)

    def test_jsr(self):
        if False:
            for i in range(10):
                print('nop')
        'Test JSR execution'
        exec_instruction('JSR R1', [(ExprId('R1', 32), ExprInt(10247, 32))], [(ExprId('PC', 32), ExprInt(10247, 32)), (ExprId('LP', 32), ExprInt(2, 32))])

    def test_ret(self):
        if False:
            while True:
                i = 10
        'Test RET execution'
        exec_instruction('RET', [(ExprId('LP', 32), ExprInt(40, 32))], [(ExprId('PC', 32), ExprInt(40, 32))])