from . import cgcrandom
from ..native.cpu.x86 import I386Cpu
from ..native.cpu.abstractcpu import Interruption, ConcretizeRegister, ConcretizeArgument
from ..native.memory import SMemory32, Memory32
from ..core.smtlib import *
from ..core.state import TerminateState
from ..binary import CGCElf
from ..platforms.platform import Platform
import logging
import random
logger = logging.getLogger(__name__)

class RestartSyscall(Exception):
    pass

class Deadlock(Exception):
    pass

class SymbolicSyscallArgument(ConcretizeRegister):

    def __init__(self, cpu, number, message='Concretizing syscall argument', policy='SAMPLED'):
        if False:
            for i in range(10):
                print('nop')
        reg_name = ['EBX', 'ECX', 'EDX', 'ESI', 'EDI', 'EBP'][number]
        super().__init__(cpu, reg_name, message, policy)

class Socket:

    @staticmethod
    def pair():
        if False:
            for i in range(10):
                print('nop')
        a = Socket()
        b = Socket()
        a.connect(b)
        return (a, b)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.buffer = []
        self.peer = None

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'SOCKET(%x, %r, %x)' % (hash(self), self.buffer, hash(self.peer))

    def is_connected(self):
        if False:
            for i in range(10):
                print('nop')
        return self.peer is not None

    def is_empty(self):
        if False:
            print('Hello World!')
        return len(self.buffer) == 0

    def is_full(self):
        if False:
            while True:
                i = 10
        return len(self.buffer) > 2 * 1024

    def connect(self, peer):
        if False:
            while True:
                i = 10
        assert not self.is_connected()
        assert not peer.is_connected()
        self.peer = peer
        if peer.peer is None:
            peer.peer = self

    def receive(self, size):
        if False:
            return 10
        rx_bytes = min(size, len(self.buffer))
        ret = []
        for i in range(rx_bytes):
            ret.append(self.buffer.pop())
        return ret

    def transmit(self, buf):
        if False:
            for i in range(10):
                print('nop')
        assert self.is_connected()
        return self.peer._transmit(buf)

    def _transmit(self, buf):
        if False:
            for i in range(10):
                print('nop')
        for c in buf:
            self.buffer.insert(0, c)
        return len(buf)

class Decree(Platform):
    """
    A simple Decree Operating System.
    This class emulates the most common Decree system calls
    """
    CGC_EBADF = 1
    CGC_EFAULT = 2
    CGC_EINVAL = 3
    CGC_ENOMEM = 4
    CGC_ENOSYS = 5
    CGC_EPIPE = 6
    CGC_SSIZE_MAX = 2147483647
    CGC_SIZE_MAX = 4294967295
    CGC_FD_SETSIZE = 32

    def __init__(self, programs, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Builds a Decree OS\n        :param cpus: CPU for this platform\n        :param mem: memory for this platform\n        :todo: generalize for more CPUs\n        :todo: fix deps?\n        '
        programs = programs.split(',')
        super().__init__(path=programs[0], **kwargs)
        self.program = programs[0]
        self.clocks = 0
        self.files = []
        self.syscall_trace = []
        self.files = []
        logger.info('Opening file descriptors (0,1,2)')
        self.input = Socket()
        self.output = Socket()
        stdin = Socket()
        stdout = Socket()
        stderr = Socket()
        stdin.peer = self.output
        stdout.peer = self.output
        stderr.peer = self.output
        self.input.peer = stdin
        assert self._open(stdin) == 0
        assert self._open(stdout) == 1
        assert self._open(stderr) == 2
        self.procs = []
        for program in programs:
            self.procs += self.load(program)
            (socka, sockb) = Socket.pair()
            self._open(socka)
            self._open(sockb)
        nprocs = len(self.procs)
        nfiles = len(self.files)
        assert nprocs > 0
        self.running = list(range(nprocs))
        self._current = 0
        self.timers = [None] * nprocs
        self.rwait = [set() for _ in range(nfiles)]
        self.twait = [set() for _ in range(nfiles)]
        for proc in self.procs:
            self.forward_events_from(proc)

    @property
    def PC(self):
        if False:
            while True:
                i = 10
        return (self._current, self.procs[self._current].PC)

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        return self

    def _mk_proc(self):
        if False:
            i = 10
            return i + 15
        return I386Cpu(Memory32())

    @property
    def current(self):
        if False:
            print('Hello World!')
        return self.procs[self._current]

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        state = super().__getstate__()
        state['clocks'] = self.clocks
        state['input'] = self.input.buffer
        state['output'] = self.output.buffer
        state['files'] = [x.buffer for x in self.files]
        state['procs'] = self.procs
        state['current'] = self._current
        state['running'] = self.running
        state['rwait'] = self.rwait
        state['twait'] = self.twait
        state['timers'] = self.timers
        state['syscall_trace'] = self.syscall_trace
        return state

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        '\n        :todo: some asserts\n        :todo: fix deps? (last line)\n        '
        super().__setstate__(state)
        self.input = Socket()
        self.input.buffer = state['input']
        self.output = Socket()
        self.output.buffer = state['output']
        self.files = []
        for buf in state['files']:
            f = Socket()
            f.buffer = buf
            self.files.append(f)
        for fd in range(len(self.files)):
            if self.connections(fd) is not None:
                self.files[fd].peer = self.files[self.connections(fd)]
        self.files[0].peer = self.output
        self.files[1].peer = self.output
        self.files[2].peer = self.output
        self.input.peer = self.files[0]
        self.procs = state['procs']
        self._current = state['current']
        self.running = state['running']
        self.rwait = state['rwait']
        self.twait = state['twait']
        self.timers = state['timers']
        self.clocks = state['clocks']
        self.syscall_trace = state['syscall_trace']
        for proc in self.procs:
            self.forward_events_from(proc)

    def _read_string(self, cpu, buf):
        if False:
            print('Hello World!')
        '\n        Reads a null terminated concrete buffer form memory\n        :todo: FIX. move to cpu or memory\n        '
        filename = ''
        for i in range(0, 1024):
            c = Operators.CHR(cpu.read_int(buf + i, 8))
            if c == '\x00':
                break
            filename += c
        return filename

    def load(self, filename):
        if False:
            while True:
                i = 10
        '\n        Loads a CGC-ELF program in memory and prepares the initial CPU state\n        and the stack.\n\n        :param filename: pathname of the file to be executed.\n        '
        CGC_MIN_PAGE_SIZE = 4096
        CGC_MIN_ALIGN = CGC_MIN_PAGE_SIZE
        TASK_SIZE = 2147483648

        def CGC_PAGESTART(_v):
            if False:
                print('Hello World!')
            return _v & ~(CGC_MIN_ALIGN - 1)

        def CGC_PAGEOFFSET(_v):
            if False:
                while True:
                    i = 10
            return _v & CGC_MIN_ALIGN - 1

        def CGC_PAGEALIGN(_v):
            if False:
                i = 10
                return i + 15
            return _v + CGC_MIN_ALIGN - 1 & ~(CGC_MIN_ALIGN - 1)

        def BAD_ADDR(x):
            if False:
                print('Hello World!')
            return x >= TASK_SIZE
        cgc = CGCElf(filename)
        logger.info('Loading %s as a %s elf' % (filename, cgc.arch))
        cpu = self._mk_proc()
        bss = brk = 0
        start_code = 4294967295
        end_code = start_data = end_data = 0
        for (vaddr, memsz, perms, name, offset, filesz) in cgc.maps():
            if vaddr < start_code:
                start_code = vaddr
            if start_data < vaddr:
                start_data = vaddr
            if vaddr > TASK_SIZE or filesz > memsz or memsz > TASK_SIZE or (TASK_SIZE - memsz < vaddr):
                raise Exception('Set_brk can never work. avoid overflows')
            addr = None
            if filesz > 0:
                hint = CGC_PAGESTART(vaddr)
                size = CGC_PAGEALIGN(filesz + CGC_PAGEOFFSET(vaddr))
                offset = CGC_PAGESTART(offset)
                addr = cpu.memory.mmapFile(hint, size, perms, name, offset)
                assert not BAD_ADDR(addr)
                lo = CGC_PAGEALIGN(vaddr + filesz)
                hi = CGC_PAGEALIGN(vaddr + memsz)
            else:
                lo = CGC_PAGESTART(vaddr + filesz)
                hi = CGC_PAGEALIGN(vaddr + memsz)
            if hi - lo > 0:
                zaddr = cpu.memory.mmap(lo, hi - lo, perms)
                assert not BAD_ADDR(zaddr)
            lo = vaddr + filesz
            hi = CGC_PAGEALIGN(vaddr + memsz)
            if hi - lo > 0:
                old_perms = cpu.memory.perms(lo)
                cpu.memory.mprotect(lo, hi - lo, 'rw')
                try:
                    cpu.memory[lo:hi] = '\x00' * (hi - lo)
                except Exception as e:
                    logger.debug('Exception zeroing main elf fractional pages: %s' % str(e))
                cpu.memory.mprotect(lo, hi, old_perms)
            if addr is None:
                addr = zaddr
            assert addr is not None
            k = vaddr + filesz
            if k > bss:
                bss = k
            if 'x' in perms and end_code < k:
                end_code = k
            if end_data < k:
                end_data = k
            k = vaddr + memsz
            if k > brk:
                brk = k
        bss = brk
        stack_base = 3131748348
        stack_size = 8388608
        stack = cpu.memory.mmap(3131748352 - stack_size, stack_size, 'rwx') + stack_size - 4
        assert stack_base in cpu.memory and stack_base - stack_size + 4 in cpu.memory
        (status, thread) = next(cgc.threads())
        assert status == 'Running'
        logger.info('Setting initial cpu state')
        cpu.write_register('EAX', 0)
        cpu.write_register('ECX', cpu.memory.mmap(CGC_PAGESTART(1128775680), CGC_PAGEALIGN(4096 + CGC_PAGEOFFSET(1128775680)), 'rwx'))
        cpu.write_register('EDX', 0)
        cpu.write_register('EBX', 0)
        cpu.write_register('ESP', stack)
        cpu.write_register('EBP', 0)
        cpu.write_register('ESI', 0)
        cpu.write_register('EDI', 0)
        cpu.write_register('EIP', thread['EIP'])
        cpu.write_register('RFLAGS', 514)
        cpu.write_register('CS', 0)
        cpu.write_register('SS', 0)
        cpu.write_register('DS', 0)
        cpu.write_register('ES', 0)
        cpu.write_register('FS', 0)
        cpu.write_register('GS', 0)
        cpu.memory.mmap(1128775680, 4096, 'r')
        logger.info('Entry point: %016x', cpu.EIP)
        logger.info('Stack start: %016x', cpu.ESP)
        logger.info('Brk: %016x', brk)
        logger.info('Mappings:')
        for m in str(cpu.memory).split('\n'):
            logger.info('  %s', m)
        return [cpu]

    def _open(self, f):
        if False:
            i = 10
            return i + 15
        if None in self.files:
            fd = self.files.index(None)
            self.files[fd] = f
        else:
            fd = len(self.files)
            self.files.append(f)
        return fd

    def _close(self, fd):
        if False:
            i = 10
            return i + 15
        '\n        Closes a file descriptor\n        :rtype: int\n        :param fd: the file descriptor to close.\n        :return: C{0} on success.\n        '
        self.files[fd] = None

    def _dup(self, fd):
        if False:
            for i in range(10):
                print('nop')
        '\n        Duplicates a file descriptor\n        :rtype: int\n        :param fd: the file descriptor to close.\n        :return: C{0} on success.\n        '
        return self._open(self.files[fd])

    def _is_open(self, fd):
        if False:
            for i in range(10):
                print('nop')
        return fd >= 0 and fd < len(self.files) and (self.files[fd] is not None)

    def sys_allocate(self, cpu, length, isX, addr):
        if False:
            i = 10
            return i + 15
        "allocate - allocate virtual memory\n\n        The  allocate  system call creates a new allocation in the virtual address\n        space of the calling process.  The length argument specifies the length of\n        the allocation in bytes which will be rounded up to the hardware page size.\n\n        The kernel chooses the address at which to create the allocation; the\n        address of the new allocation is returned in *addr as the result of the call.\n\n        All newly allocated memory is readable and writeable. In addition, the\n        is_X argument is a boolean that allows newly allocated memory to be marked\n        as executable (non-zero) or non-executable (zero).\n\n        The allocate function is invoked through system call number 5.\n\n        :param cpu: current CPU\n        :param length: the length of the allocation in bytes\n        :param isX: boolean that allows newly allocated memory to be marked as executable\n        :param addr: the address of the new allocation is returned in *addr\n\n        :return: On success, allocate returns zero and a pointer to the allocated area\n                            is returned in *addr.  Otherwise, an error code is returned\n                            and *addr is undefined.\n                EINVAL   length is zero.\n                EINVAL   length is too large.\n                EFAULT   addr points to an invalid address.\n                ENOMEM   No memory is available or the process' maximum number of allocations\n                         would have been exceeded.\n        "
        if addr not in cpu.memory:
            logger.info('ALLOCATE: addr points to invalid address. Returning EFAULT')
            return Decree.CGC_EFAULT
        perms = ['rw ', 'rwx'][bool(isX)]
        try:
            result = cpu.memory.mmap(None, length, perms)
        except Exception as e:
            logger.info('ALLOCATE exception %s. Returning ENOMEM %r', str(e), length)
            return Decree.CGC_ENOMEM
        cpu.write_int(addr, result, 32)
        logger.info('ALLOCATE(%d, %s, 0x%08x) -> 0x%08x' % (length, perms, addr, result))
        self.syscall_trace.append(('_allocate', -1, length))
        return 0

    def sys_random(self, cpu, buf, count, rnd_bytes):
        if False:
            print('Hello World!')
        'random - fill a buffer with random data\n\n        The  random  system call populates the buffer referenced by buf with up to\n        count bytes of random data. If count is zero, random returns 0 and optionally\n        sets *rx_bytes to zero. If count is greater than SSIZE_MAX, the result is unspecified.\n\n        :param cpu: current CPU\n        :param buf: a memory buffer\n        :param count: max number of bytes to receive\n        :param rnd_bytes: if valid, points to the actual number of random bytes\n\n        :return:  0        On success\n                  EINVAL   count is invalid.\n                  EFAULT   buf or rnd_bytes points to an invalid address.\n        '
        ret = 0
        if count != 0:
            if count > Decree.CGC_SSIZE_MAX or count < 0:
                ret = Decree.CGC_EINVAL
            else:
                if buf not in cpu.memory or buf + count not in cpu.memory:
                    logger.info('RANDOM: buf points to invalid address. Returning EFAULT')
                    return Decree.CGC_EFAULT
                with open('/dev/urandom', 'rb') as f:
                    data = f.read(count)
                self.syscall_trace.append(('_random', -1, data))
                cpu.write_bytes(buf, data)
        if rnd_bytes:
            if rnd_bytes not in cpu.memory:
                logger.info('RANDOM: Not valid rnd_bytes. Returning EFAULT')
                return Decree.CGC_EFAULT
            cpu.write_int(rnd_bytes, len(data), 32)
        logger.info('RANDOM(0x%08x, %d, 0x%08x) -> <%s>)' % (buf, count, rnd_bytes, repr(data[:10])))
        return ret

    def sys_receive(self, cpu, fd, buf, count, rx_bytes):
        if False:
            i = 10
            return i + 15
        'receive - receive bytes from a file descriptor\n\n        The receive system call reads up to count bytes from file descriptor fd to the\n        buffer pointed to by buf. If count is zero, receive returns 0 and optionally\n        sets *rx_bytes to zero.\n\n        :param cpu: current CPU.\n        :param fd: a valid file descriptor\n        :param buf: a memory buffer\n        :param count: max number of bytes to receive\n        :param rx_bytes: if valid, points to the actual number of bytes received\n        :return: 0            Success\n                 EBADF        fd is not a valid file descriptor or is not open\n                 EFAULT       buf or rx_bytes points to an invalid address.\n        '
        data = ''
        if count != 0:
            if not self._is_open(fd):
                logger.info('RECEIVE: Not valid file descriptor on receive. Returning EBADF')
                return Decree.CGC_EBADF
            if buf not in cpu.memory:
                logger.info('RECEIVE: buf points to invalid address. Returning EFAULT')
                return Decree.CGC_EFAULT
            if fd > 2 and self.files[fd].is_empty():
                cpu.PC -= cpu.instruction.size
                self.wait([fd], [], None)
                raise RestartSyscall()
            data = self.files[fd].receive(count)
            self.syscall_trace.append(('_receive', fd, data))
            cpu.write_bytes(buf, data)
            self.signal_receive(fd)
        if rx_bytes:
            if rx_bytes not in cpu.memory:
                logger.info('RECEIVE: Not valid file descriptor on receive. Returning EFAULT')
                return Decree.CGC_EFAULT
            cpu.write_int(rx_bytes, len(data), 32)
        logger.info('RECEIVE(%d, 0x%08x, %d, 0x%08x) -> <%s> (size:%d)' % (fd, buf, count, rx_bytes, repr(data)[:min(count, 10)], len(data)))
        return 0

    def sys_transmit(self, cpu, fd, buf, count, tx_bytes):
        if False:
            for i in range(10):
                print('nop')
        'transmit - send bytes through a file descriptor\n        The  transmit system call writes up to count bytes from the buffer pointed\n        to by buf to the file descriptor fd. If count is zero, transmit returns 0\n        and optionally sets *tx_bytes to zero.\n\n        :param cpu           current CPU\n        :param fd            a valid file descriptor\n        :param buf           a memory buffer\n        :param count         number of bytes to send\n        :param tx_bytes      if valid, points to the actual number of bytes transmitted\n        :return: 0            Success\n                 EBADF        fd is not a valid file descriptor or is not open.\n                 EFAULT       buf or tx_bytes points to an invalid address.\n        '
        data = []
        if count != 0:
            if not self._is_open(fd):
                logger.error('TRANSMIT: Not valid file descriptor. Returning EBADFD %d', fd)
                return Decree.CGC_EBADF
            if buf not in cpu.memory or buf + count not in cpu.memory:
                logger.debug('TRANSMIT: buf points to invalid address. Rerurning EFAULT')
                return Decree.CGC_EFAULT
            if fd > 2 and self.files[fd].is_full():
                cpu.PC -= cpu.instruction.size
                self.wait([], [fd], None)
                raise RestartSyscall()
            for i in range(0, count):
                value = Operators.CHR(cpu.read_int(buf + i, 8))
                if not isinstance(value, str):
                    logger.debug('TRANSMIT: Writing symbolic values to file %d', fd)
                data.append(value)
            self.files[fd].transmit(data)
            logger.info('TRANSMIT(%d, 0x%08x, %d, 0x%08x) -> <%.24r>' % (fd, buf, count, tx_bytes, ''.join([str(x) for x in data])))
            self.syscall_trace.append(('_transmit', fd, data))
            self.signal_transmit(fd)
        if tx_bytes:
            if tx_bytes not in cpu.memory:
                logger.debug('TRANSMIT: Not valid tx_bytes pointer on transmit. Returning EFAULT')
                return Decree.CGC_EFAULT
            cpu.write_int(tx_bytes, len(data), 32)
        return 0

    def sys_terminate(self, cpu, error_code):
        if False:
            while True:
                i = 10
        "\n        Exits all threads in a process\n        :param cpu: current CPU.\n        :raises Exception: 'Finished'\n        "
        procid = self.procs.index(cpu)
        self.sched()
        self.running.remove(procid)
        if issymbolic(error_code):
            logger.info('TERMINATE PROC_%02d with symbolic exit code [%d,%d]', procid, solver.minmax(self.constraints, error_code))
        else:
            logger.info('TERMINATE PROC_%02d %x', procid, error_code)
        if len(self.running) == 0:
            raise TerminateState(f'Process exited correctly. Code: {error_code}')
        return error_code

    def sys_deallocate(self, cpu, addr, size):
        if False:
            while True:
                i = 10
        'deallocate - remove allocations\n        The  deallocate  system call deletes the allocations for the specified\n        address range, and causes further references to the addresses within the\n        range to generate invalid memory accesses. The region is also\n        automatically deallocated when the process is terminated.\n\n        The address addr must be a multiple of the page size.  The length parameter\n        specifies the size of the region to be deallocated in bytes.  All pages\n        containing a part of the indicated range are deallocated, and subsequent\n        references will terminate the process.  It is not an error if the indicated\n        range does not contain any allocated pages.\n\n        The deallocate function is invoked through system call number 6.\n\n        :param cpu: current CPU\n        :param addr: the starting address to unmap.\n        :param size: the size of the portion to unmap.\n        :return 0        On success\n                EINVAL   addr is not page aligned.\n                EINVAL   length is zero.\n                EINVAL   any  part  of  the  region  being  deallocated  is outside the valid\n                         address range of the process.\n\n        :param cpu: current CPU.\n        :return: C{0} on success.\n        '
        logger.info('DEALLOCATE(0x%08x, %d)' % (addr, size))
        if addr & 4095 != 0:
            logger.info('DEALLOCATE: addr is not page aligned')
            return Decree.CGC_EINVAL
        if size == 0:
            logger.info('DEALLOCATE:length is zero')
            return Decree.CGC_EINVAL
        cpu.memory.munmap(addr, size)
        self.syscall_trace.append(('_deallocate', -1, size))
        return 0

    def sys_fdwait(self, cpu, nfds, readfds, writefds, timeout, readyfds):
        if False:
            while True:
                i = 10
        'fdwait - wait for file descriptors to become ready'
        logger.debug('FDWAIT(%d, 0x%08x, 0x%08x, 0x%08x, 0x%08x)' % (nfds, readfds, writefds, timeout, readyfds))
        if timeout:
            if timeout not in cpu.memory:
                logger.info('FDWAIT: timeout is pointing to invalid memory. Returning EFAULT')
                return Decree.CGC_EFAULT
        if readyfds:
            if readyfds not in cpu.memory:
                logger.info('FDWAIT: readyfds pointing to invalid memory. Returning EFAULT')
                return Decree.CGC_EFAULT
        writefds_wait = set()
        writefds_ready = set()
        fds_bitsize = nfds + 7 & ~7
        if writefds:
            if writefds not in cpu.memory:
                logger.info('FDWAIT: writefds pointing to invalid memory. Returning EFAULT')
                return Decree.CGC_EFAULT
            bits = cpu.read_int(writefds, fds_bitsize)
            for fd in range(nfds):
                if bits & 1 << fd:
                    if self.files[fd].is_full():
                        writefds_wait.add(fd)
                    else:
                        writefds_ready.add(fd)
        readfds_wait = set()
        readfds_ready = set()
        if readfds:
            if readfds not in cpu.memory:
                logger.info('FDWAIT: readfds pointing to invalid memory. Returning EFAULT')
                return Decree.CGC_EFAULT
            bits = cpu.read_int(readfds, fds_bitsize)
            for fd in range(nfds):
                if bits & 1 << fd:
                    if self.files[fd].is_empty():
                        readfds_wait.add(fd)
                    else:
                        readfds_ready.add(fd)
        n = len(readfds_ready) + len(writefds_ready)
        if n == 0:
            if timeout != 0:
                seconds = cpu.read_int(timeout, 32)
                microseconds = cpu.read_int(timeout + 4, 32)
                logger.info('FDWAIT: waiting for read on fds: {%s} and write to: {%s} timeout: %d', repr(list(readfds_wait)), repr(list(writefds_wait)), microseconds + 1000 * seconds)
                to = microseconds + 1000 * seconds
            else:
                to = None
                logger.info('FDWAIT: waiting for read on fds: {%s} and write to: {%s} timeout: INDIFENITELY', repr(list(readfds_wait)), repr(list(writefds_wait)))
            cpu.PC -= cpu.instruction.size
            self.wait(readfds_wait, writefds_wait, to)
            raise RestartSyscall()
        if readfds:
            bits = 0
            for fd in readfds_ready:
                bits |= 1 << fd
            for byte in range(0, nfds, 8):
                cpu.write_int(readfds, bits >> byte & 255, 8)
        if writefds:
            bits = 0
            for fd in writefds_ready:
                bits |= 1 << fd
            for byte in range(0, nfds, 8):
                cpu.write_int(writefds, bits >> byte & 255, 8)
        logger.info('FDWAIT: continuing. Some file is ready Readyfds: %08x', readyfds)
        if readyfds:
            cpu.write_int(readyfds, n, 32)
        self.syscall_trace.append(('_fdwait', -1, None))
        return 0

    def int80(self, cpu):
        if False:
            print('Hello World!')
        '\n        32 bit dispatcher.\n        :param cpu: current CPU.\n        _terminate, transmit, receive, fdwait, allocate, deallocate and random\n        '
        syscalls = {1: self.sys_terminate, 2: self.sys_transmit, 3: self.sys_receive, 4: self.sys_fdwait, 5: self.sys_allocate, 6: self.sys_deallocate, 7: self.sys_random}
        if cpu.EAX not in syscalls.keys():
            raise TerminateState(f'32 bit DECREE system call number {cpu.EAX} Not Implemented')
        func = syscalls[cpu.EAX]
        logger.debug('SYSCALL32: %s (nargs: %d)', func.__name__, func.__code__.co_argcount)
        nargs = func.__code__.co_argcount
        args = [cpu, cpu.EBX, cpu.ECX, cpu.EDX, cpu.ESI, cpu.EDI, cpu.EBP]
        cpu.EAX = func(*args[:nargs - 1])

    def sched(self):
        if False:
            for i in range(10):
                print('nop')
        'Yield CPU.\n        This will choose another process from the RUNNNIG list and change\n        current running process. May give the same cpu if only one running\n        process.\n        '
        if len(self.procs) > 1:
            logger.info('SCHED:')
            logger.info('\tProcess: %r', self.procs)
            logger.info('\tRunning: %r', self.running)
            logger.info('\tRWait: %r', self.rwait)
            logger.info('\tTWait: %r', self.twait)
            logger.info('\tTimers: %r', self.timers)
            logger.info('\tCurrent clock: %d', self.clocks)
            logger.info('\tCurrent cpu: %d', self._current)
        if len(self.running) == 0:
            logger.info('None running checking if there is some process waiting for a timeout')
            if all([x is None for x in self.timers]):
                raise Deadlock()
            self.clocks = min([x for x in self.timers if x is not None]) + 1
            self.check_timers()
            assert len(self.running) != 0, 'DEADLOCK!'
            self._current = self.running[0]
            return
        next_index = (self.running.index(self._current) + 1) % len(self.running)
        next = self.running[next_index]
        if len(self.procs) > 1:
            logger.info('\tTransfer control from process %d to %d', self._current, next)
        self._current = next

    def wait(self, readfds, writefds, timeout):
        if False:
            for i in range(10):
                print('nop')
        'Wait for filedescriptors or timeout.\n        Adds the current process to the corresponding waiting list and\n        yields the cpu to another running process.\n        '
        logger.info('WAIT:')
        logger.info('\tProcess %d is going to wait for [ %r %r %r ]', self._current, readfds, writefds, timeout)
        logger.info('\tProcess: %r', self.procs)
        logger.info('\tRunning: %r', self.running)
        logger.info('\tRWait: %r', self.rwait)
        logger.info('\tTWait: %r', self.twait)
        logger.info('\tTimers: %r', self.timers)
        for fd in readfds:
            self.rwait[fd].add(self._current)
        for fd in writefds:
            self.twait[fd].add(self._current)
        if timeout is not None:
            self.timers[self._current] = self.clocks + timeout
        else:
            self.timers[self._current] = None
        procid = self._current
        next_index = (self.running.index(procid) + 1) % len(self.running)
        self._current = self.running[next_index]
        logger.info('\tTransfer control from process %d to %d', procid, self._current)
        logger.info('\tREMOVING %r from %r. Current: %r', procid, self.running, self._current)
        self.running.remove(procid)
        if self._current not in self.running:
            logger.info('\tCurrent not running. Checking for timers...')
            self._current = None
            if all([x is None for x in self.timers]):
                raise Deadlock()
            self.check_timers()

    def awake(self, procid):
        if False:
            print('Hello World!')
        'Remove procid from waitlists and reestablish it in the running list'
        logger.info('Remove procid:%d from waitlists and reestablish it in the running list', procid)
        for wait_list in self.rwait:
            if procid in wait_list:
                wait_list.remove(procid)
        for wait_list in self.twait:
            if procid in wait_list:
                wait_list.remove(procid)
        self.timers[procid] = None
        self.running.append(procid)
        if self._current is None:
            self._current = procid

    def connections(self, fd):
        if False:
            i = 10
            return i + 15
        if fd in [0, 1, 2]:
            return None
        if fd % 2:
            return fd + 1
        else:
            return fd - 1

    def signal_receive(self, fd):
        if False:
            i = 10
            return i + 15
        'Awake one process waiting to receive data on fd'
        connections = self.connections
        if connections(fd) and self.twait[connections(fd)]:
            procid = random.sample(self.twait[connections(fd)], 1)[0]
            self.awake(procid)

    def signal_transmit(self, fd):
        if False:
            while True:
                i = 10
        'Awake one process waiting to transmit data on fd'
        connections = self.connections
        if connections(fd) and self.rwait[connections(fd)]:
            procid = random.sample(self.rwait[connections(fd)], 1)[0]
            self.awake(procid)

    def check_timers(self):
        if False:
            i = 10
            return i + 15
        'Awake process if timer has expired'
        if self._current is None:
            advance = min([x for x in self.timers if x is not None]) + 1
            logger.info('Advancing the clock from %d to %d', self.clocks, advance)
            self.clocks = advance
        for procid in range(len(self.timers)):
            if self.timers[procid] is not None:
                if self.clocks > self.timers[procid]:
                    self.procs[procid].PC += self.procs[procid].instruction.size
                    self.awake(procid)

    def execute(self):
        if False:
            while True:
                i = 10
        '\n        Execute one cpu instruction in the current thread (only one supported).\n        :rtype: bool\n        :return: C{True}\n\n        :todo: This is where we could implement a simple schedule.\n        '
        try:
            self.current.execute()
            self.clocks += 1
            if self.clocks % 10000 == 0:
                self.check_timers()
                self.sched()
        except Interruption as e:
            if e.N != 128:
                raise
            try:
                self.int80(self.current)
            except RestartSyscall:
                pass
        return True

class SDecree(Decree):
    """
    A symbolic extension of a Decree Operating System .
    """

    def __init__(self, constraints, programs, symbolic_random=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a symbolic extension of a Decree OS\n        :param constraints: a constraint set\n        :param cpus: CPU for this platform\n        :param mem: memory for this platform\n        '
        self.random = 0
        self._constraints = constraints
        super().__init__(programs)

    def _mk_proc(self):
        if False:
            print('Hello World!')
        return I386Cpu(SMemory32(self.constraints))

    @property
    def constraints(self):
        if False:
            i = 10
            return i + 15
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        if False:
            for i in range(10):
                print('nop')
        self._constraints = constraints
        for proc in self.procs:
            proc.memory.constraints = constraints

    def __getstate__(self):
        if False:
            return 10
        state = super().__getstate__()
        state['constraints'] = self.constraints
        state['random'] = self.random
        return state

    def __setstate__(self, state):
        if False:
            return 10
        self._constraints = state['constraints']
        self.random = state['random']
        super().__setstate__(state)

    def sys_receive(self, cpu, fd, buf, count, rx_bytes):
        if False:
            while True:
                i = 10
        '\n        Symbolic version of Decree.sys_receive\n        '
        if issymbolic(fd):
            logger.info('Ask to read from a symbolic file descriptor!!')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 0)
        if issymbolic(buf):
            logger.info('Ask to read to a symbolic buffer')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 1)
        if issymbolic(count):
            logger.info('Ask to read a symbolic number of bytes ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 2)
        if issymbolic(rx_bytes):
            logger.info('Ask to return size to a symbolic address ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 3)
        return super().sys_receive(cpu, fd, buf, count, rx_bytes)

    def sys_transmit(self, cpu, fd, buf, count, tx_bytes):
        if False:
            i = 10
            return i + 15
        '\n        Symbolic version of Decree.sys_transmit\n        '
        if issymbolic(fd):
            logger.info('Ask to write to a symbolic file descriptor!!')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 0)
        if issymbolic(buf):
            logger.info('Ask to write to a symbolic buffer')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 1)
        if issymbolic(count):
            logger.info('Ask to write a symbolic number of bytes ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 2)
        if issymbolic(tx_bytes):
            logger.info('Ask to return size to a symbolic address ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 3)
        return super().sys_transmit(cpu, fd, buf, count, tx_bytes)

    def sys_allocate(self, cpu, length, isX, address_p):
        if False:
            print('Hello World!')
        if issymbolic(length):
            logger.info('Ask to ALLOCATE a symbolic number of bytes ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 0)
        if issymbolic(isX):
            logger.info('Ask to ALLOCATE potentially executable or not executable memory')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 1)
        if issymbolic(address_p):
            logger.info('Ask to return ALLOCATE result to a symbolic reference ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 2)
        return super().sys_allocate(cpu, length, isX, address_p)

    def sys_deallocate(self, cpu, addr, size):
        if False:
            return 10
        if issymbolic(addr):
            logger.info('Ask to DEALLOCATE a symbolic pointer?!')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 0)
        if issymbolic(size):
            logger.info('Ask to DEALLOCATE a symbolic size?!')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 1)
        return super().sys_deallocate(cpu, addr, size)

    def sys_random(self, cpu, buf, count, rnd_bytes):
        if False:
            return 10
        if issymbolic(buf):
            logger.info('Ask to write random bytes to a symbolic buffer')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 0)
        if issymbolic(count):
            logger.info('Ask to read a symbolic number of random bytes ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 1)
        if issymbolic(rnd_bytes):
            logger.info('Ask to return rnd size to a symbolic address ')
            cpu.PC = cpu.PC - cpu.instruction.size
            raise SymbolicSyscallArgument(cpu, 2)
        data = []
        for i in range(count):
            value = cgcrandom.stream[self.random]
            data.append(value)
            self.random += 1
        cpu.write_bytes(buf, data)
        if rnd_bytes:
            cpu.write_int(rnd_bytes, len(data), 32)
        logger.info('RANDOM(0x%08x, %d, 0x%08x) -> %d', buf, count, rnd_bytes, len(data))
        return 0

class DecreeEmu:
    RANDOM = 0

    @staticmethod
    def cgc_initialize_secret_page(platform):
        if False:
            return 10
        logger.info('Skipping: cgc_initialize_secret_page()')
        return 0

    @staticmethod
    def cgc_random(platform, buf, count, rnd_bytes):
        if False:
            print('Hello World!')
        from . import cgcrandom
        if issymbolic(buf):
            logger.info('Ask to write random bytes to a symbolic buffer')
            raise ConcretizeArgument(platform.current, 0)
        if issymbolic(count):
            logger.info('Ask to read a symbolic number of random bytes ')
            raise ConcretizeArgument(platform.current, 1)
        if issymbolic(rnd_bytes):
            logger.info('Ask to return rnd size to a symbolic address ')
            raise ConcretizeArgument(platform.current, 2)
        data = []
        for i in range(count):
            value = cgcrandom.stream[DecreeEmu.RANDOM]
            data.append(value)
            DecreeEmu.random += 1
        cpu = platform.current
        cpu.write(buf, data)
        if rnd_bytes:
            cpu.store(rnd_bytes, len(data), 32)
        logger.info('RANDOM(0x%08x, %d, 0x%08x) -> %d', buf, count, rnd_bytes, len(data))
        return 0