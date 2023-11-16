from __future__ import print_function
from manticore.native import Manticore
from manticore.platforms import linux_syscalls
import logging
from sys import argv, exit
import struct
import qemu
import gdb
logger = logging.getLogger('TRACE')
stack_top = 3221225472
stack_size = 131072
initialized = False
last_instruction = None
in_helper = False

def init_logging():
    if False:
        print('Hello World!')

    class ContextFilter(logging.Filter):

        def filter(self, record):
            if False:
                i = 10
                return i + 15
            record.stateid = ''
            return True
    logger.addFilter(ContextFilter())

def dump_gdb(cpu, addr, count):
    if False:
        i = 10
        return i + 15
    for offset in range(addr, addr + count, 4):
        val = int(gdb.getM(offset) & 4294967295)
        val2 = int(cpu.read_int(offset))
        print(f'{offset:x}: g{val:08x} m{val2:08x}')

def cmp_regs(cpu, should_print=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compare registers from a remote gdb session to current mcore.\n\n    :param manticore.core.cpu Cpu: Current cpu\n    :param bool should_print: Whether to print values to stdout\n    :return: Whether or not any differences were detected\n    :rtype: bool\n    '
    differing = False
    gdb_regs = gdb.getCanonicalRegisters()
    for name in sorted(gdb_regs):
        vg = gdb_regs[name]
        if name.endswith('psr'):
            name = 'apsr'
        v = cpu.read_register(name.upper())
        if should_print:
            logger.debug(f'{name} gdb:{vg:x} mcore:{v:x}')
        if vg != v:
            if should_print:
                logger.warning('^^ unequal')
            differing = True
    if differing:
        logger.debug(qemu.correspond(None))
    return differing

def pre_mcore(state):
    if False:
        print('Hello World!')
    if state.cpu.instruction.mnemonic.lower() == 'svc':
        state.cpu.memory.push_record_writes()

def post_mcore(state, last_instruction):
    if False:
        return 10
    '\n    Handle syscalls (import memory) and bail if we diverge\n    '
    global in_helper
    if last_instruction.mnemonic.lower() == 'svc':
        writes = state.cpu.memory.pop_record_writes()
        if writes:
            logger.debug('Got %d writes', len(writes))
        for (addr, val) in writes:
            gdb.setByte(addr, val[0])
        gdb_r0 = gdb.getR('R0')
        if gdb_r0 != state.cpu.R0:
            logger.debug(f"Writing 0x{state.cpu.R0:x} to R0 (overwriting 0x{gdb.getR('R0'):x})")
        for reg in state.cpu.canonical_registers:
            if reg.endswith('PSR') or reg in ('R15', 'PC'):
                continue
            val = state.cpu.read_register(reg)
            gdb.setR(reg, val)
    if state.cpu.PC >> 16 == 65535:
        in_helper = True
        return
    if in_helper:
        for reg in state.cpu.canonical_registers:
            if reg.endswith('PSR'):
                continue
            if reg == 'R15':
                continue
            gdb.setR(reg, state.cpu.read_register(reg))
        in_helper = False
    if cmp_regs(state.cpu):
        cmp_regs(state.cpu, should_print=True)
        state.abandon()

def pre_qemu(state):
    if False:
        for i in range(10):
            print('nop')
    pass

def post_qemu(state, last_mnemonic):
    if False:
        while True:
            i = 10
    if last_mnemonic.lower() == 'svc':
        sync_svc(state)

def sync_svc(state):
    if False:
        for i in range(10):
            print('nop')
    '\n    Mirror some service calls in manticore. Happens after qemu executed a SVC\n    instruction, but before manticore did.\n    '
    syscall = state.cpu.R7
    name = linux_syscalls.armv7[syscall]
    logger.debug(f'Syncing syscall: {name}')
    try:
        if 'mmap' in name:
            returned = gdb.getR('R0')
            logger.debug(f'Syncing mmap ({returned:x})')
            state.cpu.write_register('R0', returned)
        if 'exit' in name:
            return
    except ValueError:
        for reg in state.cpu.canonical_registers:
            print(f'{reg}: {state.cpu.read_register(reg):x}')
        raise

def initialize(state):
    if False:
        i = 10
        return i + 15
    '\n    Synchronize the stack and register state (manticore->qemu)\n    '
    logger.debug(f'Copying {stack_top - state.cpu.SP} bytes in the stack..')
    stack_bottom = min(state.cpu.SP, gdb.getR('SP'))
    for address in range(stack_bottom, stack_top):
        b = state.cpu.read_int(address, 8)
        gdb.setByte(address, chr(b))
    logger.debug('Done')
    mcore_stdout = state.platform.files[1]
    state.platform.files.append(mcore_stdout)
    state.platform.files.append(mcore_stdout)
    for gdb_reg in gdb.getCanonicalRegisters():
        if gdb_reg.endswith('psr'):
            mcore_reg = 'APSR'
        else:
            mcore_reg = gdb_reg.upper()
        value = state.cpu.read_register(mcore_reg)
        gdb.setR(gdb_reg, value)

def verify(argv):
    if False:
        return 10
    logger.debug(f'Verifying program "{argv}"')
    qemu.start('arm', argv, va_size=stack_top, stack_size=stack_size)
    gdb.start('arm', argv)
    m = Manticore(argv[0], argv[1:])
    m.verbosity = 2
    init_logging()
    logger.setLevel(logging.DEBUG)

    @m.hook(None)
    def on_instruction(state):
        if False:
            while True:
                i = 10
        '\n        Handle all the hooks for each instruction executed. Ordered as:\n\n        pre_qemu\n         * qemu exec *\n        post_qemu\n\n        // svc synchronization happens here (mmap specifically)\n\n        pre_mcore\n         * mcore exec *\n        post_mcore\n\n        // all memory written in a mcore syscall gets moved to qemu here\n        '
        global initialized, last_instruction
        if not initialized:
            initialize(state)
            initialized = True
        if last_instruction:
            post_mcore(state, last_instruction)
        if state.cpu.PC >> 16 == 65535:
            return
        pre_qemu(state)
        last_mnemonic = [x.strip() for x in gdb.getInstruction().split(':')][1].split('\t')[0]
        gdb.stepi()
        post_qemu(state, last_mnemonic)
        last_instruction = state.cpu.instruction
        pre_mcore(state)
    m.run()
if __name__ == '__main__':
    args = argv[1:]
    if len(args) == 0:
        print(f'usage: python {argv[0]} PROGRAM1 ...')
        exit()
    verify(args)