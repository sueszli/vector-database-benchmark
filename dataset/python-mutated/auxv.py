from __future__ import annotations
import os
import re
import sys
import gdb
import pwndbg.gdblib.abi
import pwndbg.gdblib.arch
import pwndbg.gdblib.events
import pwndbg.gdblib.info
import pwndbg.gdblib.memory
import pwndbg.gdblib.qemu
import pwndbg.gdblib.regs
import pwndbg.gdblib.stack
import pwndbg.gdblib.typeinfo
example_info_auxv_linux = '\n33   AT_SYSINFO_EHDR      System-supplied DSO\'s ELF header 0x7ffff7ffa000\n16   AT_HWCAP             Machine-dependent CPU capability hints 0xfabfbff\n6    AT_PAGESZ            System page size               4096\n17   AT_CLKTCK            Frequency of times()           100\n3    AT_PHDR              Program headers for program    0x400040\n4    AT_PHENT             Size of program header entry   56\n5    AT_PHNUM             Number of program headers      9\n7    AT_BASE              Base address of interpreter    0x7ffff7dda000\n8    AT_FLAGS             Flags                          0x0\n9    AT_ENTRY             Entry point of program         0x42020b\n11   AT_UID               Real user ID                   1000\n12   AT_EUID              Effective user ID              1000\n13   AT_GID               Real group ID                  1000\n14   AT_EGID              Effective group ID             1000\n23   AT_SECURE            Boolean, was exec setuid-like? 0\n25   AT_RANDOM            Address of 16 random bytes     0x7fffffffdb39\n31   AT_EXECFN            File name of executable        0x7fffffffefee "/bin/bash"\n15   AT_PLATFORM          String identifying platform    0x7fffffffdb49 "x86_64"\n0    AT_NULL              End of vector                  0x0\n'
AT_CONSTANTS = {0: 'AT_NULL', 1: 'AT_IGNORE', 2: 'AT_EXECFD', 3: 'AT_PHDR', 4: 'AT_PHENT', 5: 'AT_PHNUM', 6: 'AT_PAGESZ', 7: 'AT_BASE', 8: 'AT_FLAGS', 9: 'AT_ENTRY', 10: 'AT_NOTELF', 11: 'AT_UID', 12: 'AT_EUID', 13: 'AT_GID', 14: 'AT_EGID', 15: 'AT_PLATFORM', 16: 'AT_HWCAP', 17: 'AT_CLKTCK', 18: 'AT_FPUCW', 19: 'AT_DCACHEBSIZE', 20: 'AT_ICACHEBSIZE', 21: 'AT_UCACHEBSIZE', 22: 'AT_IGNOREPPC', 23: 'AT_SECURE', 24: 'AT_BASE_PLATFORM', 25: 'AT_RANDOM', 31: 'AT_EXECFN', 32: 'AT_SYSINFO', 33: 'AT_SYSINFO_EHDR', 34: 'AT_L1I_CACHESHAPE', 35: 'AT_L1D_CACHESHAPE', 36: 'AT_L2_CACHESHAPE', 37: 'AT_L3_CACHESHAPE'}
sys.modules[__name__].__dict__.update({v: k for (k, v) in AT_CONSTANTS.items()})

class AUXV(dict):

    def set(self, const, value) -> None:
        if False:
            i = 10
            return i + 15
        name = AT_CONSTANTS.get(const, 'AT_UNKNOWN%i' % const)
        if name in ['AT_EXECFN', 'AT_PLATFORM']:
            try:
                value = gdb.Value(value)
                value = value.cast(pwndbg.gdblib.typeinfo.pchar)
                value = value.string()
            except Exception:
                value = 'couldnt read AUXV!'
        self[name] = value

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        return self.get(attr)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return str({k: v for (k, v) in self.items() if v is not None})

@pwndbg.lib.cache.cache_until('objfile', 'start')
def get():
    if False:
        return 10
    return use_info_auxv() or walk_stack() or AUXV()

def use_info_auxv():
    if False:
        while True:
            i = 10
    lines = pwndbg.gdblib.info.auxv().splitlines()
    if not lines:
        return None
    auxv = AUXV()
    for line in lines:
        match = re.match('([0-9]+) .*? (0x[0-9a-f]+|[0-9]+$)', line)
        if not match:
            print(f"Warning: Skipping auxv entry '{line}'")
            continue
        (const, value) = (int(match.group(1)), int(match.group(2), 0))
        auxv.set(const, value)
    return auxv

def find_stack_boundary(addr):
    if False:
        while True:
            i = 10
    addr = pwndbg.lib.memory.page_align(int(addr))
    try:
        while True:
            if b'\x7fELF' == pwndbg.gdblib.memory.read(addr, 4):
                break
            addr += pwndbg.lib.memory.PAGE_SIZE
    except gdb.MemoryError:
        pass
    return addr

def walk_stack():
    if False:
        while True:
            i = 10
    if not pwndbg.gdblib.abi.linux:
        return None
    if pwndbg.gdblib.qemu.is_qemu_kernel():
        return None
    auxv = walk_stack2(0)
    if not auxv:
        auxv = walk_stack2(1)
    if not auxv.get('AT_EXECFN', None):
        try:
            auxv['AT_EXECFN'] = _get_execfn()
        except gdb.MemoryError:
            pass
    return auxv

def walk_stack2(offset=0):
    if False:
        for i in range(10):
            print('nop')
    sp = pwndbg.gdblib.regs.sp
    if not sp:
        return AUXV()
    end = find_stack_boundary(sp)
    p = gdb.Value(end).cast(pwndbg.gdblib.typeinfo.ulong.pointer())
    p -= offset
    p -= 2
    try:
        while p.dereference() != 0 or (p + 1).dereference() != 0:
            p -= 2
        for i in range(1024):
            if p.dereference() == AT_BASE:
                break
            p -= 2
        else:
            return AUXV()
        while (p - 2).dereference() < 37:
            p -= 2
        auxv = AUXV()
        while True:
            const = int((p + 0).dereference()) & pwndbg.gdblib.arch.ptrmask
            value = int((p + 1).dereference()) & pwndbg.gdblib.arch.ptrmask
            if const == AT_NULL:
                break
            auxv.set(const, value)
            p += 2
        return auxv
    except gdb.MemoryError:
        return AUXV()

def _get_execfn():
    if False:
        while True:
            i = 10
    if not pwndbg.gdblib.memory.peek(pwndbg.gdblib.regs.sp):
        return
    addr = pwndbg.gdblib.stack.find_upper_stack_boundary(pwndbg.gdblib.regs.sp)
    while pwndbg.gdblib.memory.byte(addr - 1) == 0:
        addr -= 1
    while pwndbg.gdblib.memory.byte(addr - 1) != 0:
        addr -= 1
    v = pwndbg.gdblib.strings.get(addr, 1024)
    if v:
        return os.path.abspath(v)