"""PEP 656 support.

This module implements logic to detect if the currently running Python is
linked against musl, and what musl version is used.
"""
import contextlib
import functools
import operator
import os
import re
import struct
import subprocess
import sys
from typing import IO, Iterator, NamedTuple, Optional, Tuple

def _read_unpacked(f: IO[bytes], fmt: str) -> Tuple[int, ...]:
    if False:
        while True:
            i = 10
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

def _parse_ld_musl_from_elf(f: IO[bytes]) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Detect musl libc location by parsing the Python executable.\n\n    Based on: https://gist.github.com/lyssdod/f51579ae8d93c8657a5564aefc2ffbca\n    ELF header: https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.eheader.html\n    '
    f.seek(0)
    try:
        ident = _read_unpacked(f, '16B')
    except struct.error:
        return None
    if ident[:4] != tuple(b'\x7fELF'):
        return None
    f.seek(struct.calcsize('HHI'), 1)
    try:
        (e_fmt, p_fmt, p_idx) = {1: ('IIIIHHH', 'IIIIIIII', (0, 1, 4)), 2: ('QQQIHHH', 'IIQQQQQQ', (0, 2, 5))}[ident[4]]
    except KeyError:
        return None
    else:
        p_get = operator.itemgetter(*p_idx)
    try:
        (_, e_phoff, _, _, _, e_phentsize, e_phnum) = _read_unpacked(f, e_fmt)
    except struct.error:
        return None
    for i in range(e_phnum + 1):
        f.seek(e_phoff + e_phentsize * i)
        try:
            (p_type, p_offset, p_filesz) = p_get(_read_unpacked(f, p_fmt))
        except struct.error:
            return None
        if p_type != 3:
            continue
        f.seek(p_offset)
        interpreter = os.fsdecode(f.read(p_filesz)).strip('\x00')
        if 'musl' not in interpreter:
            return None
        return interpreter
    return None

class _MuslVersion(NamedTuple):
    major: int
    minor: int

def _parse_musl_version(output: str) -> Optional[_MuslVersion]:
    if False:
        while True:
            i = 10
    lines = [n for n in (n.strip() for n in output.splitlines()) if n]
    if len(lines) < 2 or lines[0][:4] != 'musl':
        return None
    m = re.match('Version (\\d+)\\.(\\d+)', lines[1])
    if not m:
        return None
    return _MuslVersion(major=int(m.group(1)), minor=int(m.group(2)))

@functools.lru_cache()
def _get_musl_version(executable: str) -> Optional[_MuslVersion]:
    if False:
        print('Hello World!')
    "Detect currently-running musl runtime version.\n\n    This is done by checking the specified executable's dynamic linking\n    information, and invoking the loader to parse its output for a version\n    string. If the loader is musl, the output would be something like::\n\n        musl libc (x86_64)\n        Version 1.2.2\n        Dynamic Program Loader\n    "
    with contextlib.ExitStack() as stack:
        try:
            f = stack.enter_context(open(executable, 'rb'))
        except OSError:
            return None
        ld = _parse_ld_musl_from_elf(f)
    if not ld:
        return None
    proc = subprocess.run([ld], stderr=subprocess.PIPE, universal_newlines=True)
    return _parse_musl_version(proc.stderr)

def platform_tags(arch: str) -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    'Generate musllinux tags compatible to the current platform.\n\n    :param arch: Should be the part of platform tag after the ``linux_``\n        prefix, e.g. ``x86_64``. The ``linux_`` prefix is assumed as a\n        prerequisite for the current platform to be musllinux-compatible.\n\n    :returns: An iterator of compatible musllinux tags.\n    '
    sys_musl = _get_musl_version(sys.executable)
    if sys_musl is None:
        return
    for minor in range(sys_musl.minor, -1, -1):
        yield f'musllinux_{sys_musl.major}_{minor}_{arch}'
if __name__ == '__main__':
    import sysconfig
    plat = sysconfig.get_platform()
    assert plat.startswith('linux-'), 'not linux'
    print('plat:', plat)
    print('musl:', _get_musl_version(sys.executable))
    print('tags:', end=' ')
    for t in platform_tags(re.sub('[.-]', '_', plat.split('-', 1)[-1])):
        print(t, end='\n      ')