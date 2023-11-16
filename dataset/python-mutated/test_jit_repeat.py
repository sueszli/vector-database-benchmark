from ut_helpers_jit import jit_instructions

class TestRepeat(object):

    def test_repeat(self):
        if False:
            for i in range(10):
                print('nop')
        'Test REPEAT jit'
        instructions = 'MOV R0, 8\n'
        instructions += 'REPEAT R0, 0x6\n'
        instructions += 'ADD R1, 1\n'
        instructions += 'ADD R2, 1\n'
        instructions += 'ADD R3, 1'
        jitter = jit_instructions(instructions)
        assert jitter.cpu.R0 == 8
        assert jitter.cpu.R1 == 8
        assert jitter.cpu.R2 == 8
        assert jitter.cpu.R3 == 8

    def test_erepeat_0(self):
        if False:
            return 10
        'Test EREPEAT jit'
        instructions = 'EREPEAT 0xA\n'
        instructions += 'ADD R1, 1\n'
        instructions += 'BEQI R1, 0x6, 0x8\n'
        instructions += 'ADD R2, 1\n'
        instructions += 'ADD R3, 1'
        jitter = jit_instructions(instructions)
        assert jitter.cpu.R1 == 6
        assert jitter.cpu.R2 == 5
        assert jitter.cpu.R3 == 5

    def test_erepeat_1(self):
        if False:
            while True:
                i = 10
        'Test EREPEAT jit'
        instructions = 'EREPEAT 0x8\n'
        instructions += 'ADD R1, 1\n'
        instructions += 'ADD R2, 1\n'
        instructions += 'ADD R3, 1\n'
        instructions += 'BEQI R1, 0x6, 0x4\n'
        instructions += 'ADD R2, 1\n'
        instructions += 'ADD R3, 1'
        jitter = jit_instructions(instructions)
        assert jitter.cpu.R1 == 6
        assert jitter.cpu.R2 == 7
        assert jitter.cpu.R3 == 7

    def test_erepeat_2(self):
        if False:
            print('Hello World!')
        'Test EREPEAT jit'
        instructions = 'EREPEAT 0x8\n'
        instructions += 'ADD R1, 1\n'
        instructions += 'ADD R2, 1\n'
        instructions += 'ADD R3, 1\n'
        instructions += 'BEQI R3, 0x6, 0x4'
        jitter = jit_instructions(instructions)
        assert jitter.cpu.R1 == 6
        assert jitter.cpu.R2 == 6
        assert jitter.cpu.R3 == 6

    def test_erepeat_3(self):
        if False:
            i = 10
            return i + 15
        'Test EREPEAT jit'
        instructions = 'EREPEAT 0x8\n'
        instructions += 'ADD R1, 1\n'
        instructions += 'ADD R2, 1\n'
        instructions += 'BEQI R1, 0x6, 0x6\n'
        instructions += 'ADD R3, 1'
        jitter = jit_instructions(instructions)
        assert jitter.cpu.R1 == 6
        assert jitter.cpu.R2 == 6
        assert jitter.cpu.R3 == 5