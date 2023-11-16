from __future__ import print_function
import logging
from miasm.analysis.sandbox import Sandbox_Win_x86_32
from miasm.core.locationdb import LocationDB
from miasm.jitter.csts import PAGE_WRITE, PAGE_READ, EXCEPT_BREAKPOINT_MEMORY
parser = Sandbox_Win_x86_32.parser(description='Displays accesses to a specified memory space')
parser.add_argument('filename', help='PE Filename')
parser.add_argument('memory_address', help='Starting address of the memory space')
parser.add_argument('size', help='Size of the address space')
parser.add_argument('--access', help='Access type', choices=['r', 'w', 'rw'], default='rw')
options = parser.parse_args()
loc_db = LocationDB()
sb = Sandbox_Win_x86_32(loc_db, options.filename, options, globals())
address = int(options.memory_address, 0)
size = int(options.size, 0)
access_type = 0
if 'r' in options.access:
    access_type |= PAGE_WRITE
if 'w' in options.access:
    access_type |= PAGE_READ
sb.jitter.vm.add_memory_breakpoint(address, size, access_type)

def memory_breakpoint_handler(jitter):
    if False:
        for i in range(10):
            print('nop')
    memory_read = jitter.vm.get_memory_read()
    if len(memory_read) > 0:
        print('Read at instruction 0x%s:' % jitter.pc)
        for (start_address, end_address) in memory_read:
            print('- from %s to %s' % (hex(start_address), hex(end_address)))
    memory_write = jitter.vm.get_memory_write()
    if len(memory_write) > 0:
        print('Write at instruction 0x%s:' % jitter.pc)
        for (start_address, end_address) in memory_write:
            print('- from %s to %s' % (hex(start_address), hex(end_address)))
    jitter.vm.set_exception(jitter.vm.get_exception() ^ EXCEPT_BREAKPOINT_MEMORY)
    jitter.vm.reset_memory_access()
    return False
sb.jitter.add_exception_handler(EXCEPT_BREAKPOINT_MEMORY, memory_breakpoint_handler)
sb.run()