from __future__ import annotations
import gdb
import pwnlib
import pwndbg.gdblib.proc
from pwndbg.gdblib import typeinfo
from pwndbg.lib.arch import Arch
ARCHS = ('x86-64', 'i386', 'aarch64', 'mips', 'powerpc', 'sparc', 'arm', 'armcm', 'rv32', 'rv64')
pwnlib_archs_mapping = {'x86-64': 'amd64', 'i386': 'i386', 'aarch64': 'aarch64', 'mips': 'mips', 'powerpc': 'powerpc', 'sparc': 'sparc', 'arm': 'arm', 'armcm': 'thumb', 'rv32': 'riscv32', 'rv64': 'riscv64'}
arch = Arch('i386', typeinfo.ptrsize, 'little')

def _get_arch(ptrsize: int):
    if False:
        for i in range(10):
            print('nop')
    not_exactly_arch = False
    if 'little' in gdb.execute('show endian', to_string=True).lower():
        endian = 'little'
    else:
        endian = 'big'
    if pwndbg.gdblib.proc.alive:
        arch = gdb.newest_frame().architecture().name()
    else:
        arch = gdb.execute('show architecture', to_string=True).strip()
        not_exactly_arch = True
    for match in ARCHS:
        if match in arch:
            if match == 'arm' and '-m' in arch:
                match = 'armcm'
            return (match, ptrsize, endian)
    if not_exactly_arch:
        raise RuntimeError(f'Could not deduce architecture from: {arch}')
    return (arch, ptrsize, endian)

def update() -> None:
    if False:
        while True:
            i = 10
    (arch_name, ptrsize, endian) = _get_arch(typeinfo.ptrsize)
    arch.update(arch_name, ptrsize, endian)
    pwnlib.context.context.arch = pwnlib_archs_mapping[arch_name]
    pwnlib.context.context.bits = ptrsize * 8