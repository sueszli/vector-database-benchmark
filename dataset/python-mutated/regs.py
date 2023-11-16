"""
Reading register value from the inferior, and provides a
standardized interface to registers like "sp" and "pc".
"""
from __future__ import annotations
import ctypes
import re
import sys
from types import ModuleType
import gdb
import pwndbg.gdblib.arch
import pwndbg.gdblib.events
import pwndbg.gdblib.proc
import pwndbg.gdblib.remote
import pwndbg.lib.cache
from pwndbg.lib.regs import reg_sets

@pwndbg.gdblib.proc.OnlyWhenRunning
def gdb_get_register(name: str):
    if False:
        for i in range(10):
            print('nop')
    return gdb.selected_frame().read_register(name)
PTRACE_ARCH_PRCTL = 30
ARCH_GET_FS = 4099
ARCH_GET_GS = 4100

class module(ModuleType):
    last: dict[str, int] = {}

    @pwndbg.lib.cache.cache_until('stop', 'prompt')
    def __getattr__(self, attr: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        attr = attr.lstrip('$')
        try:
            value = gdb_get_register(attr)
            if value is None and attr.lower() == 'xpsr':
                value = gdb_get_register('xPSR')
            size = pwndbg.gdblib.typeinfo.unsigned.get(value.type.sizeof, pwndbg.gdblib.typeinfo.ulong)
            value = value.cast(size)
            if attr == 'pc' and pwndbg.gdblib.arch.current == 'i8086':
                value += self.cs * 16
            value = int(value)
            return value & pwndbg.gdblib.arch.ptrmask
        except (ValueError, gdb.error):
            return None

    def __setattr__(self, attr, val):
        if False:
            print('Hello World!')
        if attr in ('last', 'previous'):
            return super().__setattr__(attr, val)
        else:
            gdb.execute(f'set ${attr} = {val}')

    @pwndbg.lib.cache.cache_until('stop', 'prompt')
    def __getitem__(self, item: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, str):
            print('Unknown register type: %r' % item)
            return None
        item = item.lstrip('$')
        item = getattr(self, item.lower())
        if isinstance(item, int):
            return int(item) & pwndbg.gdblib.arch.ptrmask
        return item

    def __contains__(self, reg) -> bool:
        if False:
            for i in range(10):
                print('nop')
        regs = set(reg_sets[pwndbg.gdblib.arch.current]) | {'pc', 'sp'}
        return reg in regs

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        regs = set(reg_sets[pwndbg.gdblib.arch.current]) | {'pc', 'sp'}
        yield from regs

    @property
    def current(self):
        if False:
            while True:
                i = 10
        return reg_sets[pwndbg.gdblib.arch.current]

    @property
    def gpr(self):
        if False:
            return 10
        return reg_sets[pwndbg.gdblib.arch.current].gpr

    @property
    def common(self):
        if False:
            for i in range(10):
                print('nop')
        return reg_sets[pwndbg.gdblib.arch.current].common

    @property
    def frame(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return reg_sets[pwndbg.gdblib.arch.current].frame

    @property
    def retaddr(self):
        if False:
            print('Hello World!')
        return reg_sets[pwndbg.gdblib.arch.current].retaddr

    @property
    def flags(self):
        if False:
            return 10
        return reg_sets[pwndbg.gdblib.arch.current].flags

    @property
    def stack(self):
        if False:
            for i in range(10):
                print('nop')
        return reg_sets[pwndbg.gdblib.arch.current].stack

    @property
    def retval(self):
        if False:
            print('Hello World!')
        return reg_sets[pwndbg.gdblib.arch.current].retval

    @property
    def all(self):
        if False:
            return 10
        regs = reg_sets[pwndbg.gdblib.arch.current]
        retval: list[str] = []
        for regset in (regs.pc, regs.stack, regs.frame, regs.retaddr, regs.flags, regs.gpr, regs.misc):
            if regset is None:
                continue
            elif isinstance(regset, (list, tuple)):
                retval.extend(regset)
            elif isinstance(regset, dict):
                retval.extend(regset.keys())
            else:
                retval.append(regset)
        return retval

    def fix(self, expression):
        if False:
            return 10
        for regname in set(self.all + ['sp', 'pc']):
            expression = re.sub(f'\\$?\\b{regname}\\b', '$' + regname, expression)
        return expression

    def items(self):
        if False:
            while True:
                i = 10
        for regname in self.all:
            yield (regname, self[regname])
    reg_sets = reg_sets

    @property
    def changed(self):
        if False:
            for i in range(10):
                print('nop')
        delta = []
        for (reg, value) in self.previous.items():
            if self[reg] != value:
                delta.append(reg)
        return delta

    @property
    @pwndbg.lib.cache.cache_until('stop')
    def fsbase(self):
        if False:
            for i in range(10):
                print('nop')
        return self._fs_gs_helper('fs_base', ARCH_GET_FS)

    @property
    @pwndbg.lib.cache.cache_until('stop')
    def gsbase(self):
        if False:
            i = 10
            return i + 15
        return self._fs_gs_helper('gs_base', ARCH_GET_GS)

    @pwndbg.lib.cache.cache_until('stop')
    def _fs_gs_helper(self, regname: str, which):
        if False:
            while True:
                i = 10
        "Supports fetching based on segmented addressing, a la fs:[0x30].\n        Requires ptrace'ing the child directory if i386."
        if pwndbg.gdblib.arch.current == 'x86-64':
            return gdb_get_register(regname)
        if pwndbg.gdblib.remote.is_remote():
            return 0
        (pid, lwpid, tid) = gdb.selected_thread().ptid
        ppvoid = ctypes.POINTER(ctypes.c_void_p)
        value = ppvoid(ctypes.c_void_p())
        value.contents.value = 0
        libc = ctypes.CDLL('libc.so.6')
        result = libc.ptrace(PTRACE_ARCH_PRCTL, lwpid, value, which)
        if result == 0:
            return (value.contents.value or 0) & pwndbg.gdblib.arch.ptrmask
        return 0

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<module pwndbg.gdblib.regs>'
tether = sys.modules[__name__]
sys.modules[__name__] = module(__name__, '')

@pwndbg.gdblib.events.cont
@pwndbg.gdblib.events.stop
def update_last() -> None:
    if False:
        while True:
            i = 10
    M: module = sys.modules[__name__]
    M.previous = M.last
    M.last = {k: M[k] for k in M.common}
    if pwndbg.gdblib.config.show_retaddr_reg:
        M.last.update({k: M[k] for k in M.retaddr})