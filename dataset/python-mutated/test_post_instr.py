from __future__ import print_function
import sys
from miasm.core.utils import decode_hex
from miasm.analysis.machine import Machine
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE, EXCEPT_BREAKPOINT_MEMORY, EXCEPT_ACCESS_VIOL
from miasm.core.locationdb import LocationDB
from miasm.jitter.jitload import JitterException
machine = Machine('x86_32')
loc_db = LocationDB()
jitter = machine.jitter(loc_db, sys.argv[1])
jitter.vm.add_memory_page(65536, PAGE_READ | PAGE_WRITE, b'\x00' * 4096, 'stack')
print(jitter.vm)
jitter.cpu.ESP = 65536 + 4096
jitter.push_uint32_t(0)
jitter.push_uint32_t(322420463)
jitter.vm.reset_memory_access()
print(hex(jitter.vm.get_exception()))
jitter.vm.add_memory_page(4096, PAGE_READ | PAGE_WRITE, b'\x00' * 4096, 'code page')
jitter.vm.set_mem(4096, decode_hex('B844332211C3'))
jitter.set_trace_log()

def do_not_raise_me(jitter):
    if False:
        for i in range(10):
            print('nop')
    raise ValueError('Should not be here')
jitter.add_exception_handler(EXCEPT_BREAKPOINT_MEMORY, do_not_raise_me)
jitter.vm.add_memory_breakpoint(69632 - 4, 4, PAGE_READ | PAGE_WRITE)
jitter.init_run(4096)
try:
    jitter.continue_run()
except JitterException:
    assert jitter.vm.get_exception() == EXCEPT_ACCESS_VIOL