import sys
from manticore import issymbolic
from manticore.native import Manticore
"\nReplaces a variable that controls program flow with a tainted symbolic value. This\nin turn explores all possible states under that variable's influence, and reports the\nspecific cmp/test instructions can be influenced by tainted data.\n\nUsage:\n\n $ gcc -static -g src/state_explore.c -o state_explore # -static is optional\n $ ADDRESS=0x$(objdump -S state_explore | grep -A 1 '((value & 0xff) != 0)' |\n         tail -n 1 | sed 's|^\\s*||g' | cut -f1 -d:)\n $ python ./introduce_symbolic_bytes.py state_explore $ADDRESS\n Tainted Control Flow:\n introducing symbolic value to 7ffffffffd44\n 400a0e: test eax, eax\n 400a19: cmp eax, 0x3f\n 400b17: test eax, eax\n 400b1e: cmp eax, 0x1000\n 400b63: test eax, eax\n 400a3e: cmp eax, 0x41\n 400a64: cmp eax, 0x42\n 400a8a: cmp eax, 0x43\n 400ab0: cmp eax, 0x44\n 400b6a: cmp eax, 0xf0000\n Analysis finished. See ./mcore_cz3Jzp for results.\n"
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write(f'Usage: {sys.argv[0]} [binary] [address]\n')
        sys.exit(2)
    m = Manticore(sys.argv[1], ['anything'])
    introduce_at = int(sys.argv[2], 0)
    taint_id = 'taint_A'

    @m.hook(introduce_at)
    def introduce_sym(state):
        if False:
            for i in range(10):
                print('nop')
        print(f'introducing symbolic value to {state.cpu.RBP - 12:x}')
        val = state.new_symbolic_value(32, taint=(taint_id,))
        state.cpu.write_int(state.cpu.RBP - 12, val, 32)

    def has_tainted_operands(operands, taint_id):
        if False:
            return 10
        for operand in operands:
            op = operand.read()
            if issymbolic(op) and taint_id in op.taint:
                return True
        return False
    every_instruction = None

    @m.hook(every_instruction)
    def check_taint(state):
        if False:
            return 10
        insn = state.cpu.instruction
        if insn is None:
            return
        if insn.mnemonic in ('cmp', 'test'):
            if has_tainted_operands(insn.operands, taint_id):
                print(f'{insn.address:x}: {insn.mnemonic} {insn.op_str}')
    print('Tainted Control Flow:')
    m.run()
    print(f'Analysis finished. See {m.workspace} for results.')