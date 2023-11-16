import sys
from miasm.core.utils import decode_hex
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE, EXCEPT_ACCESS_VIOL
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB

def code_sentinelle(jitter):
    if False:
        print('Hello World!')
    jitter.running = False
    jitter.pc = 0
    return True
machine = Machine('x86_32')
loc_db = LocationDB()
jitter = machine.jitter(loc_db, sys.argv[1])
jitter.init_stack()
data = decode_hex('90b842000000eb20')
error_raised = False

def raise_me(jitter):
    if False:
        i = 10
        return i + 15
    global error_raised
    error_raised = True
    assert jitter.pc == 1073741864
    return False
jitter.add_exception_handler(EXCEPT_ACCESS_VIOL, raise_me)
run_addr = 1073741824
jitter.vm.add_memory_page(run_addr, PAGE_READ | PAGE_WRITE, data)
jitter.set_trace_log()
jitter.push_uint32_t(322420463)
jitter.add_breakpoint(322420463, code_sentinelle)
jitter.init_run(run_addr)
jitter.continue_run()
assert error_raised is True