import sys
from manticore.native import Manticore
'\nDemonstrates the ability to set a basic hook on a specific program counter and\nthe ability to read from memory.\n'
if __name__ == '__main__':
    path = sys.argv[1]
    pc = int(sys.argv[2], 0)
    m = Manticore(path)

    @m.hook(pc)
    def reached_goal(state):
        if False:
            for i in range(10):
                print('nop')
        cpu = state.cpu
        assert cpu.PC == pc
        instruction = cpu.read_int(cpu.PC)
        print('Execution goal reached.')
        print(f'Instruction bytes: {instruction:08x}')
    m.run()