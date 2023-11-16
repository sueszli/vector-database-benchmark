from __future__ import print_function
from argparse import ArgumentParser
from miasm.analysis import debugging
from miasm.jitter.csts import *
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
parser = ArgumentParser(description='Sandbox raw binary with msp430 engine\n(ex: jit_msp430.py example/msp430_sc.bin 0)')
parser.add_argument('-t', '--trace', help='Log instructions/registers values', action='store_true')
parser.add_argument('-n', '--log-newbloc', help='Log basic blocks processed by the Jitter', action='store_true')
parser.add_argument('-j', '--jitter', help="Jitter engine (default is 'gcc')", default='gcc')
parser.add_argument('-d', '--debugging', help='Attach a CLI debugguer to the sandboxed program', action='store_true')
parser.add_argument('binary', help='binary to run')
parser.add_argument('addr', help='start exec on addr')
machine = Machine('msp430')

def jit_msp430_binary(args):
    if False:
        for i in range(10):
            print('nop')
    loc_db = LocationDB()
    (filepath, entryp) = (args.binary, int(args.addr, 0))
    myjit = machine.jitter(loc_db, jit_type=args.jitter)
    myjit.set_trace_log(trace_instr=args.trace, trace_regs=args.trace, trace_new_blocks=args.log_newbloc)
    myjit.vm.add_memory_page(0, PAGE_READ | PAGE_WRITE, open(filepath, 'rb').read())
    myjit.add_breakpoint(4919, lambda _: exit(0))
    myjit.vm.add_memory_page(61440, PAGE_READ | PAGE_WRITE, b'\x00' * 4096)
    myjit.cpu.SP = 63488
    myjit.push_uint16_t(4919)
    myjit.init_run(entryp)
    if args.debugging is True:
        dbg = debugging.Debugguer(myjit)
        cmd = debugging.DebugCmd(dbg)
        cmd.cmdloop()
    else:
        print(myjit.continue_run())
if __name__ == '__main__':
    args = parser.parse_args()
    jit_msp430_binary(args)