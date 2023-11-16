import os
from manticore.native import Manticore
DIR = os.path.dirname(__file__)
FILE = os.path.join(DIR, 'hello42')
if __name__ == '__main__':
    m = Manticore(FILE)
    with m.locked_context() as context:
        context['count'] = 0

    @m.hook(None)
    def explore(state):
        if False:
            for i in range(10):
                print('nop')
        with m.locked_context() as context:
            context['count'] += 1
            if state.cpu.PC == 4222736:
                s = state.cpu.read_string(state.cpu.X0)
                assert s == 'hello'
                print(f'puts argument: {s}')
            elif state.cpu.PC == 4223084:
                result = state.cpu.X0
                assert result >= 0
                print(f'puts result: {result}')
            elif state.cpu.PC == 4283984:
                status = state.cpu.X0
                syscall = state.cpu.X8
                assert syscall == 94
                print(f'exit status: {status}')

    def execute_instruction(self, insn, msg):
        if False:
            i = 10
            return i + 15
        print(f'{msg}: 0x{insn.address:x}: {insn.mnemonic} {insn.op_str}')
    m.subscribe('will_execute_instruction', lambda self, state, pc, insn: execute_instruction(self, insn, 'next'))
    m.subscribe('did_execute_instruction', lambda self, state, last_pc, pc, insn: execute_instruction(self, insn, 'done'))
    m.run()
    print(f"Executed {m.context['count']} instructions")