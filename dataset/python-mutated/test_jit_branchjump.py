from ut_helpers_jit import jit_instructions

class TestBranchJump(object):

    def test_blti(self):
        if False:
            i = 10
            return i + 15
        'Test BLTI jit'
        instructions = 'MOV R0, 1\n'
        instructions += 'BLTI R0, 0x2, 0x6\n'
        instructions += 'MOV R0, 0\n'
        instructions += 'MOV R1, 1'
        jitter = jit_instructions(instructions)
        assert jitter.cpu.R0 == 1
        assert jitter.cpu.R1 == 1