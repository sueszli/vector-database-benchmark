import sys
from manticore.native import Manticore
'\nSolves modified version of baby-re, compiled for arm.\n'
if __name__ == '__main__':
    path = sys.argv[1]
    m = Manticore(path)

    @m.hook(68080)
    def myhook(state):
        if False:
            print('Hello World!')
        flag = ''
        cpu = state.cpu
        arraytop = cpu.R11
        base = arraytop - 24
        for i in range(4):
            symbolic_input = cpu.read_int(base + i * 4)
            concrete_input = state.solve_one(symbolic_input)
            flag += chr(concrete_input & 255)
        print('flag is:', flag)
        m.terminate()
    m.run()