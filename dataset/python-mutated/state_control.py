import sys
from manticore.native import Manticore
"\nDemonstrates the ability to guide Manticore's state exploration. In this case,\nabandoning a state we're no longer interested in.\n\nUsage:\n\n $ gcc -static -g src/state_explore.c -o state_explore # -static is optional\n $ ADDRESS=0x$(objdump -S state_explore | grep -A 1 'value == 0x41' | tail -n 1 | sed 's|^\\s*||g' | cut -f1 -d:)\n $ python ./state_control.py state_explore $ADDRESS\n\n"
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write(f'Usage: {sys.argv[0]} [binary] [address]\n')
        sys.exit(2)
    m = Manticore(sys.argv[1])
    to_abandon = int(sys.argv[2], 0)

    @m.hook(to_abandon)
    def explore(state):
        if False:
            print('Hello World!')
        print(f'Abandoning state at PC: {state.cpu.PC:x}')
        state.abandon()
    print(f'Adding hook to: {to_abandon:x}')
    m.run()