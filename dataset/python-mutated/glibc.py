"""
Get information about the GLibc
"""
from __future__ import annotations
import functools
import os
import re
import gdb
from elftools.elf.relocation import Relocation
import pwndbg.gdblib.config
import pwndbg.gdblib.elf
import pwndbg.gdblib.file
import pwndbg.gdblib.info
import pwndbg.gdblib.memory
import pwndbg.gdblib.proc
import pwndbg.gdblib.symbol
import pwndbg.heap
import pwndbg.lib.cache
import pwndbg.search
from pwndbg.color import message
safe_lnk = pwndbg.gdblib.config.add_param('safe-linking', None, 'whether glibc use safe-linking (on/off/auto)', param_class=gdb.PARAM_AUTO_BOOLEAN)
glibc_version = pwndbg.gdblib.config.add_param('glibc', '', 'GLIBC version for heap heuristics resolution (e.g. 2.31)', scope='heap')

@pwndbg.gdblib.config.trigger(glibc_version)
def set_glibc_version() -> None:
    if False:
        print('Hello World!')
    ret = re.search('(\\d+)\\.(\\d+)', glibc_version.value)
    if ret:
        glibc_version.value = tuple(map(int, ret.groups()))
        return
    print(message.warn(f'Invalid GLIBC version: `{glibc_version.value}`, you should provide something like: 2.31 or 2.34'))
    glibc_version.revert_default()

@pwndbg.gdblib.proc.OnlyWhenRunning
def get_version() -> tuple[int, ...] | None:
    if False:
        print('Hello World!')
    return glibc_version or _get_version()

@pwndbg.gdblib.proc.OnlyWhenRunning
@pwndbg.lib.cache.cache_until('start', 'objfile')
def _get_version() -> tuple[int, ...] | None:
    if False:
        for i in range(10):
            print('nop')
    if pwndbg.heap.current.libc_has_debug_syms():
        addr = pwndbg.gdblib.symbol.address('__libc_version')
        if addr is not None:
            ver = pwndbg.gdblib.memory.string(addr)
            return tuple((int(_) for _ in ver.split(b'.')))
    libc_filename = get_libc_filename_from_info_sharedlibrary()
    if not libc_filename:
        return None
    result = pwndbg.gdblib.elf.dump_section_by_name(libc_filename, '.rodata', try_local_path=True)
    if not result:
        return None
    (_, _, data) = result
    banner_start = data.find(b'GNU C Library')
    if banner_start == -1:
        return None
    banner = data[banner_start:data.find(b'\x00', banner_start)]
    ret = re.search(b'release version (\\d+)\\.(\\d+)', banner)
    return tuple((int(_) for _ in ret.groups())) if ret else None

@pwndbg.gdblib.proc.OnlyWhenRunning
@pwndbg.lib.cache.cache_until('start', 'objfile')
def get_libc_filename_from_info_sharedlibrary() -> str | None:
    if False:
        i = 10
        return i + 15
    '\n    Get the filename of the libc by parsing the output of `info sharedlibrary`.\n    '
    possible_libc_path = []
    for path in pwndbg.gdblib.info.sharedlibrary_paths():
        basename = os.path.basename(path[7:] if path.startswith('target:') else path)
        if basename == 'libc.so.6':
            return path
        elif re.search('^libc6?[-_\\.]', basename):
            possible_libc_path.append(path)
    if possible_libc_path:
        return possible_libc_path[0]
    return None

@pwndbg.gdblib.proc.OnlyWhenRunning
@pwndbg.lib.cache.cache_until('start', 'objfile')
def dump_elf_data_section() -> tuple[int, int, bytes] | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Dump .data section of libc ELF file\n    '
    libc_filename = get_libc_filename_from_info_sharedlibrary()
    if not libc_filename:
        return None
    return pwndbg.gdblib.elf.dump_section_by_name(libc_filename, '.data', try_local_path=True)

@pwndbg.gdblib.proc.OnlyWhenRunning
@pwndbg.lib.cache.cache_until('start', 'objfile')
def dump_relocations_by_section_name(section_name: str) -> tuple[Relocation, ...] | None:
    if False:
        while True:
            i = 10
    '\n    Dump relocations of a section by section name of libc ELF file\n    '
    libc_filename = get_libc_filename_from_info_sharedlibrary()
    if not libc_filename:
        return None
    return pwndbg.gdblib.elf.dump_relocations_by_section_name(libc_filename, section_name, try_local_path=True)

@pwndbg.gdblib.proc.OnlyWhenRunning
@pwndbg.lib.cache.cache_until('start', 'objfile')
def get_data_section_address() -> int:
    if False:
        i = 10
        return i + 15
    '\n    Find .data section address of libc\n    '
    libc_filename = get_libc_filename_from_info_sharedlibrary()
    if not libc_filename:
        return 0
    out = pwndbg.gdblib.info.files()
    for line in out.splitlines():
        if line.endswith(' is .data in ' + libc_filename):
            return int(line.split()[0], 16)
    return 0

@pwndbg.gdblib.proc.OnlyWhenRunning
@pwndbg.lib.cache.cache_until('start', 'objfile')
def get_got_section_address() -> int:
    if False:
        return 10
    '\n    Find .got section address of libc\n    '
    libc_filename = get_libc_filename_from_info_sharedlibrary()
    if not libc_filename:
        return 0
    out = pwndbg.gdblib.info.files()
    for line in out.splitlines():
        if line.endswith(' is .got in ' + libc_filename):
            return int(line.split()[0], 16)
    return 0

def OnlyWhenGlibcLoaded(function):
    if False:
        while True:
            i = 10

    @functools.wraps(function)
    def _OnlyWhenGlibcLoaded(*a, **kw):
        if False:
            for i in range(10):
                print('nop')
        if get_version() is not None:
            return function(*a, **kw)
        else:
            print(f'{function.__name__}: GLibc not loaded yet.')
    return _OnlyWhenGlibcLoaded

@OnlyWhenGlibcLoaded
def check_safe_linking():
    if False:
        return 10
    '\n    Safe-linking is a glibc 2.32 mitigation; see:\n    - https://lanph3re.blogspot.com/2020/08/blog-post.html\n    - https://research.checkpoint.com/2020/safe-linking-eliminating-a-20-year-old-malloc-exploit-primitive/\n    '
    return (get_version() >= (2, 32) or safe_lnk) and safe_lnk is not False