from argparse import ArgumentParser
from pdb import pm
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE, EXCEPT_SYSCALL
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
SYSCALL = {0: 'read', 1: 'write', 2: 'open', 9: 'mmap', 39: 'getpid', 41: 'socket', 42: 'connect', 43: 'accept', 44: 'sendto', 45: 'recvfrom', 49: 'bind', 50: 'listen', 51: 'getsockname', 52: 'getpeername', 59: 'execve', 60: 'exit', 61: 'wait4', 62: 'kill', 87: 'unlink', 90: 'chmod', 91: 'fchmod', 92: 'chown'}

def code_sentinelle(jitter):
    if False:
        for i in range(10):
            print('nop')
    jitter.running = False
    jitter.pc = 0
    return True

def log_syscalls(jitter):
    if False:
        return 10
    if jitter.cpu.EAX == 1:
        size_t = jitter.cpu.RDX
        print('write(fd: {}, buf: {}, size_t: {})'.format(jitter.cpu.RDI, jitter.vm.get_mem(jitter.cpu.RSI, size_t), size_t))
        jitter.cpu.EAX = size_t
    elif jitter.cpu.EAX == 60:
        print('Exit syscall - stopping the machine')
        return False
    elif jitter.cpu.EAX in SYSCALL:
        print('syscall {} - {} : Not Implemented'.format(jitter.cpu.EAX, SYSCALL[jitter.cpu.EAX]))
    else:
        print('Unknown syscall {} : NotImplemented'.format(jitter.cpu.EAX))
    jitter.cpu.set_exception(0)
    jitter.cpu.EAX = 0
    return True
if __name__ == '__main__':
    parser = ArgumentParser(description='x86 64 basic Jitter')
    parser.add_argument('filename', help='x86 64 shellcode filename')
    parser.add_argument('-j', '--jitter', help="Jitter engine (default is 'gcc')", default='gcc')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose mode')
    args = parser.parse_args()
    loc_db = LocationDB()
    myjit = Machine('x86_64').jitter(loc_db, args.jitter)
    myjit.init_stack()
    with open(args.filename, 'rb') as f:
        data = f.read()
    run_addr = 1073741824
    myjit.vm.add_memory_page(run_addr, PAGE_READ | PAGE_WRITE, data)
    if args.verbose:
        myjit.set_trace_log()
    myjit.push_uint64_t(322420463)
    myjit.add_breakpoint(322420463, code_sentinelle)
    myjit.add_exception_handler(EXCEPT_SYSCALL, log_syscalls)
    myjit.run(run_addr)