from __future__ import print_function
import os
import sys
from miasm.core.utils import decode_hex
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
from pdb import pm
data = decode_hex('b810000000bb0100000083e8010f44cb75f8c3')
run_addr = 1073741824
loc_db = LocationDB()

def code_sentinelle(jitter):
    if False:
        while True:
            i = 10
    jitter.running = False
    jitter.pc = 0
    return True

def init_jitter(loc_db):
    if False:
        while True:
            i = 10
    global data, run_addr
    myjit = Machine('x86_32').jitter(loc_db, sys.argv[1])
    myjit.vm.add_memory_page(run_addr, PAGE_READ | PAGE_WRITE, data)
    myjit.init_stack()
    myjit.set_trace_log()
    myjit.push_uint32_t(322420463)
    myjit.add_breakpoint(322420463, code_sentinelle)
    return myjit
print('[+] First run, to jit blocks')
myjit = init_jitter(loc_db)
myjit.init_run(run_addr)
myjit.continue_run()
assert myjit.running is False
assert myjit.cpu.EAX == 0
myjit.jit.options['max_exec_per_call'] = 5
first_call = True

def cb(jitter):
    if False:
        i = 10
        return i + 15
    global first_call
    if first_call:
        first_call = False
        return True
    return False
print('[+] Second run')
myjit.push_uint32_t(322420463)
myjit.cpu.EAX = 0
myjit.init_run(run_addr)
myjit.exec_cb = cb
myjit.continue_run()
assert myjit.running is True
assert myjit.cpu.EAX >= 10
print('[+] Run instr one by one')
myjit = init_jitter(loc_db)
myjit.jit.options['jit_maxline'] = 1
myjit.jit.options['max_exec_per_call'] = 1
counter = 0

def cb(jitter):
    if False:
        for i in range(10):
            print('nop')
    global counter
    counter += 1
    return True
myjit.init_run(run_addr)
myjit.exec_cb = cb
myjit.continue_run()
assert myjit.running is False
assert myjit.cpu.EAX == 0
assert counter == 52