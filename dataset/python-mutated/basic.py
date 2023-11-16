import os
import struct
import sys
from manticore.native import Manticore
DIR = os.path.dirname(__file__)
FILE = os.path.join(DIR, 'basic')
STDIN = sys.stdin.readline()
m = Manticore(FILE, concrete_start='', stdin_size=0)

@m.init
def init(state):
    if False:
        return 10
    state.platform.input.write(state.symbolicate_buffer(STDIN, label='STDIN'))

@m.hook(4196028)
def hook_if(state):
    if False:
        return 10
    print('hook if')
    state.abandon()

@m.hook(4196044)
def hook_else(state):
    if False:
        while True:
            i = 10
    print('hook else')
    print_constraints(state, 6)
    w0 = state.cpu.W0
    if isinstance(w0, int):
        print(hex(w0))
    else:
        print(w0)
    solved = state.solve_one(w0)
    print(struct.pack('<I', solved))

@m.hook(4196052)
def hook_puts(state):
    if False:
        for i in range(10):
            print('nop')
    print('hook puts')
    cpu = state.cpu
    print(cpu.read_string(cpu.X0))

def print_constraints(state, nlines):
    if False:
        return 10
    i = 0
    for c in str(state.constraints).split('\n'):
        if i >= nlines:
            break
        print(c)
        i += 1
m.run()