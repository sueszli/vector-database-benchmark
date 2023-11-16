"""
Provides values which would be available from /proc which
are not fulfilled by other modules and some process/gdb flow
related information.
"""
from __future__ import annotations
import functools
import sys
from types import ModuleType
from typing import Any
from typing import Callable
import gdb
from elftools.elf.relocation import Relocation
import pwndbg.gdblib.qemu
import pwndbg.lib.cache
import pwndbg.lib.memory

class module(ModuleType):

    @property
    def pid(self):
        if False:
            i = 10
            return i + 15
        if pwndbg.gdblib.qemu.is_qemu_usermode():
            return pwndbg.gdblib.qemu.pid()
        i = gdb.selected_inferior()
        if i is not None:
            return i.pid
        return 0

    @property
    def tid(self):
        if False:
            print('Hello World!')
        if pwndbg.gdblib.qemu.is_qemu_usermode():
            return pwndbg.gdblib.qemu.pid()
        i = gdb.selected_thread()
        if i is not None:
            return i.ptid[1]
        return self.pid

    @property
    def thread_id(self):
        if False:
            while True:
                i = 10
        return gdb.selected_thread().num

    @property
    def alive(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Informs whether the process has a thread. However, note that it will\n        still return True for a segfaulted thread. To detect that, consider\n        using the `stopped_with_signal` method.\n        '
        return gdb.selected_thread() is not None

    @property
    def thread_is_stopped(self):
        if False:
            print('Hello World!')
        '\n        This detects whether selected thread is stopped.\n        It is not stopped in situations when gdb is executing commands\n        that are attached to a breakpoint by `command` command.\n\n        For more info see issue #229 ( https://github.com/pwndbg/pwndbg/issues/299 )\n        :return: Whether gdb executes commands attached to bp with `command` command.\n        '
        return gdb.selected_thread().is_stopped()

    @property
    def stopped_with_signal(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns whether the program has stopped with a signal\n\n        Can be used to detect segfaults (but will also detect other signals)\n        '
        return 'It stopped with signal ' in gdb.execute('info program', to_string=True)

    @property
    def exe(self):
        if False:
            while True:
                i = 10
        '\n        Returns the debugged file name.\n\n        On remote targets, this may be prefixed with "target:" string.\n        See this by executing those in two terminals:\n        1. gdbserver 127.0.0.1:1234 /bin/ls\n        2. gdb -ex "target remote :1234" -ex "pi pwndbg.gdblib.proc.exe"\n\n        If you need to process the debugged file use:\n            `pwndbg.gdblib.file.get_proc_exe_file()`\n            (This will call `pwndbg.gdblib.file.get_file(pwndbg.gdblib.proc.exe, try_local_path=True)`)\n        '
        return gdb.current_progspace().filename

    @property
    @pwndbg.lib.cache.cache_until('start', 'stop')
    def binary_base_addr(self) -> int:
        if False:
            print('Hello World!')
        return self.binary_vmmap[0].start

    @property
    @pwndbg.lib.cache.cache_until('start', 'stop')
    def binary_vmmap(self) -> tuple[pwndbg.lib.memory.Page, ...]:
        if False:
            i = 10
            return i + 15
        return tuple((p for p in pwndbg.gdblib.vmmap.get() if p.objfile == self.exe))

    @pwndbg.lib.cache.cache_until('start', 'objfile')
    def dump_elf_data_section(self) -> tuple[int, int, bytes] | None:
        if False:
            while True:
                i = 10
        "\n        Dump .data section of current process's ELF file\n        "
        return pwndbg.gdblib.elf.dump_section_by_name(self.exe, '.data', try_local_path=True)

    @pwndbg.lib.cache.cache_until('start', 'objfile')
    def dump_relocations_by_section_name(self, section_name: str) -> tuple[Relocation, ...] | None:
        if False:
            print('Hello World!')
        "\n        Dump relocations of a section by section name of current process's ELF file\n        "
        return pwndbg.gdblib.elf.dump_relocations_by_section_name(self.exe, section_name, try_local_path=True)

    @pwndbg.lib.cache.cache_until('start', 'objfile')
    def get_data_section_address(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Find .data section address of current process.\n        '
        out = pwndbg.gdblib.info.files()
        for line in out.splitlines():
            if line.endswith(' is .data'):
                return int(line.split()[0], 16)
        return 0

    @pwndbg.lib.cache.cache_until('start', 'objfile')
    def get_got_section_address(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Find .got section address of current process.\n        '
        out = pwndbg.gdblib.info.files()
        for line in out.splitlines():
            if line.endswith(' is .got'):
                return int(line.split()[0], 16)
        return 0

    def OnlyWhenRunning(self, func):
        if False:
            print('Hello World!')

        @functools.wraps(func)
        def wrapper(*a, **kw):
            if False:
                while True:
                    i = 10
            if self.alive:
                return func(*a, **kw)
        return wrapper
OnlyWhenRunning: Callable[[Any], Any]
tether = sys.modules[__name__]
sys.modules[__name__] = module(__name__, '')