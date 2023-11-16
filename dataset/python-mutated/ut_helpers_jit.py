from __future__ import print_function
from miasm.analysis.machine import Machine
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE
from miasm.core.locationdb import LocationDB

def jit_instructions(mn_str):
    if False:
        i = 10
        return i + 15
    'JIT instructions and return the jitter object.'
    machine = Machine('mepb')
    mn_mep = machine.mn()
    loc_db = LocationDB()
    asm = b''
    for instr_str in mn_str.split('\n'):
        instr = mn_mep.fromstring(instr_str, 'b')
        instr.mode = 'b'
        asm += mn_mep.asm(instr)[0]
    jitter = machine.jitter(loc_db, jit_type='gcc')
    jitter.vm.add_memory_page(0, PAGE_READ | PAGE_WRITE, asm)
    jitter.add_breakpoint(len(asm), lambda x: False)
    jitter.init_run(0)
    jitter.continue_run()
    return jitter

def launch_tests(obj):
    if False:
        while True:
            i = 10
    'Call test methods by name'
    test_methods = [name for name in dir(obj) if name.startswith('test')]
    for method in test_methods:
        print(method)
        try:
            getattr(obj, method)()
        except AttributeError as e:
            print('Method not found: %s' % method)
            assert False
        print('-' * 42)