"""Read information from Core Dumps.

Core dumps are extremely useful when writing exploits, even outside of
the normal act of debugging things.

Using Corefiles to Automate Exploitation
----------------------------------------

For example, if you have a trivial buffer overflow and don't want to
open up a debugger or calculate offsets, you can use a generated core
dump to extract the relevant information.

.. code-block:: c

    #include <string.h>
    #include <stdlib.h>
    #include <unistd.h>
    void win() {
        system("sh");
    }
    int main(int argc, char** argv) {
        char buffer[64];
        strcpy(buffer, argv[1]);
    }

.. code-block:: shell

    $ gcc crash.c -m32 -o crash -fno-stack-protector

.. code-block:: python

    from pwn import *

    # Generate a cyclic pattern so that we can auto-find the offset
    payload = cyclic(128)

    # Run the process once so that it crashes
    process(['./crash', payload]).wait()

    # Get the core dump
    core = Coredump('./core')

    # Our cyclic pattern should have been used as the crashing address
    assert pack(core.eip) in payload

    # Cool! Now let's just replace that value with the address of 'win'
    crash = ELF('./crash')
    payload = fit({
        cyclic_find(core.eip): crash.symbols.win
    })

    # Get a shell!
    io = process(['./crash', payload])
    io.sendline(b'id')
    print(io.recvline())
    # uid=1000(user) gid=1000(user) groups=1000(user)

Module Members
----------------------------------------

"""
from __future__ import absolute_import
from __future__ import division
import collections
import ctypes
import glob
import gzip
import re
import os
import socket
import subprocess
import tempfile
from io import BytesIO, StringIO
import elftools
from elftools.common.utils import roundup
from elftools.common.utils import struct_parse
from elftools.construct import CString
from pwnlib import atexit
from pwnlib.context import context
from pwnlib.elf.datatypes import *
from pwnlib.elf.elf import ELF
from pwnlib.log import getLogger
from pwnlib.tubes.process import process
from pwnlib.tubes.ssh import ssh_channel
from pwnlib.tubes.tube import tube
from pwnlib.util.fiddling import b64d
from pwnlib.util.fiddling import enhex
from pwnlib.util.fiddling import unhex
from pwnlib.util.misc import read
from pwnlib.util.misc import write
from pwnlib.util.packing import _decode
from pwnlib.util.packing import pack
from pwnlib.util.packing import unpack_many
log = getLogger(__name__)
prstatus_types = {'i386': elf_prstatus_i386, 'amd64': elf_prstatus_amd64, 'arm': elf_prstatus_arm, 'aarch64': elf_prstatus_aarch64}
prpsinfo_types = {32: elf_prpsinfo_32, 64: elf_prpsinfo_64}
siginfo_types = {32: elf_siginfo_32, 64: elf_siginfo_64}

def iter_notes(self):
    if False:
        i = 10
        return i + 15
    ' Iterates the list of notes in the segment.\n    '
    offset = self['p_offset']
    end = self['p_offset'] + self['p_filesz']
    while offset < end:
        note = struct_parse(self.elffile.structs.Elf_Nhdr, self.stream, stream_pos=offset)
        note['n_offset'] = offset
        offset += self.elffile.structs.Elf_Nhdr.sizeof()
        self.stream.seek(offset)
        disk_namesz = roundup(note['n_namesz'], 2)
        with context.local(encoding='latin-1'):
            note['n_name'] = _decode(CString('').parse(self.stream.read(disk_namesz)))
            offset += disk_namesz
            desc_data = _decode(self.stream.read(note['n_descsz']))
            note['n_desc'] = desc_data
        offset += roundup(note['n_descsz'], 2)
        note['n_size'] = offset - note['n_offset']
        yield note

class Mapping(object):
    """Encapsulates information about a memory mapping in a :class:`Corefile`.
    """

    def __init__(self, core, name, start, stop, flags, page_offset):
        if False:
            i = 10
            return i + 15
        self._core = core
        self.name = name or ''
        self.start = start
        self.stop = stop
        self.size = stop - start
        self.page_offset = page_offset or 0
        self.flags = flags

    @property
    def path(self):
        if False:
            while True:
                i = 10
        ':class:`str`: Alias for :attr:`.Mapping.name`'
        return self.name

    @property
    def address(self):
        if False:
            i = 10
            return i + 15
        ':class:`int`: Alias for :data:`Mapping.start`.'
        return self.start

    @property
    def permstr(self):
        if False:
            while True:
                i = 10
        ':class:`str`: Human-readable memory permission string, e.g. ``r-xp``.'
        flags = self.flags
        return ''.join(['r' if flags & 4 else '-', 'w' if flags & 2 else '-', 'x' if flags & 1 else '-', 'p'])

    def __str__(self):
        if False:
            return 10
        return '%x-%x %s %x %s' % (self.start, self.stop, self.permstr, self.size, self.name)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '%s(%r, start=%#x, stop=%#x, size=%#x, flags=%#x, page_offset=%#x)' % (self.__class__.__name__, self.name, self.start, self.stop, self.size, self.flags, self.page_offset)

    def __int__(self):
        if False:
            while True:
                i = 10
        return self.start

    @property
    def data(self):
        if False:
            for i in range(10):
                print('nop')
        ':class:`str`: Memory of the mapping.'
        return self._core.read(self.start, self.size)

    def __getitem__(self, item):
        if False:
            i = 10
            return i + 15
        if isinstance(item, slice):
            start = int(item.start or self.start)
            stop = int(item.stop or self.stop)
            if start < 0:
                start += self.stop
            if stop < 0:
                stop += self.stop
            if not self.start <= start <= stop <= self.stop:
                log.error('Byte range [%#x:%#x] not within range [%#x:%#x]', start, stop, self.start, self.stop)
            data = self._core.read(start, stop - start)
            if item.step == 1:
                return data
            return data[::item.step]
        return self._core.read(item, 1)

    def __contains__(self, item):
        if False:
            i = 10
            return i + 15
        if isinstance(item, Mapping):
            return self.start <= item.start and item.stop <= self.stop
        return self.start <= item < self.stop

    def find(self, sub, start=None, end=None):
        if False:
            i = 10
            return i + 15
        'Similar to str.find() but works on our address space'
        if start is None:
            start = self.start
        if end is None:
            end = self.stop
        result = self.data.find(sub, start - self.address, end - self.address)
        if result == -1:
            return result
        return result + self.address

    def rfind(self, sub, start=None, end=None):
        if False:
            print('Hello World!')
        'Similar to str.rfind() but works on our address space'
        if start is None:
            start = self.start
        if end is None:
            end = self.stop
        result = self.data.rfind(sub, start - self.address, end - self.address)
        if result == -1:
            return result
        return result + self.address

class Corefile(ELF):
    """Enhances the information available about a corefile (which is an extension
    of the ELF format) by permitting extraction of information about the mapped
    data segments, and register state.

    Registers can be accessed directly, e.g. via ``core_obj.eax`` and enumerated
    via :data:`Corefile.registers`.

    Memory can be accessed directly via :meth:`.read` or :meth:`.write`, and also
    via :meth:`.pack` or :meth:`.unpack` or even :meth:`.string`.

    Arguments:
        core: Path to the core file.  Alternately, may be a :class:`.process` instance,
              and the core file will be located automatically.

    ::

        >>> c = Corefile('./core')
        >>> hex(c.eax)
        '0xfff5f2e0'
        >>> c.registers
        {'eax': 4294308576,
         'ebp': 1633771891,
         'ebx': 4151132160,
         'ecx': 4294311760,
         'edi': 0,
         'edx': 4294308700,
         'eflags': 66050,
         'eip': 1633771892,
         'esi': 0,
         'esp': 4294308656,
         'orig_eax': 4294967295,
         'xcs': 35,
         'xds': 43,
         'xes': 43,
         'xfs': 0,
         'xgs': 99,
         'xss': 43}

    Mappings can be iterated in order via :attr:`Corefile.mappings`.

    ::

        >>> Corefile('./core').mappings
        [Mapping('/home/user/pwntools/crash', start=0x8048000, stop=0x8049000, size=0x1000, flags=0x5, page_offset=0x0),
         Mapping('/home/user/pwntools/crash', start=0x8049000, stop=0x804a000, size=0x1000, flags=0x4, page_offset=0x1),
         Mapping('/home/user/pwntools/crash', start=0x804a000, stop=0x804b000, size=0x1000, flags=0x6, page_offset=0x2),
         Mapping(None, start=0xf7528000, stop=0xf7529000, size=0x1000, flags=0x6, page_offset=0x0),
         Mapping('/lib/i386-linux-gnu/libc-2.19.so', start=0xf7529000, stop=0xf76d1000, size=0x1a8000, flags=0x5, page_offset=0x0),
         Mapping('/lib/i386-linux-gnu/libc-2.19.so', start=0xf76d1000, stop=0xf76d2000, size=0x1000, flags=0x0, page_offset=0x1a8),
         Mapping('/lib/i386-linux-gnu/libc-2.19.so', start=0xf76d2000, stop=0xf76d4000, size=0x2000, flags=0x4, page_offset=0x1a9),
         Mapping('/lib/i386-linux-gnu/libc-2.19.so', start=0xf76d4000, stop=0xf76d5000, size=0x1000, flags=0x6, page_offset=0x1aa),
         Mapping(None, start=0xf76d5000, stop=0xf76d8000, size=0x3000, flags=0x6, page_offset=0x0),
         Mapping(None, start=0xf76ef000, stop=0xf76f1000, size=0x2000, flags=0x6, page_offset=0x0),
         Mapping('[vdso]', start=0xf76f1000, stop=0xf76f2000, size=0x1000, flags=0x5, page_offset=0x0),
         Mapping('/lib/i386-linux-gnu/ld-2.19.so', start=0xf76f2000, stop=0xf7712000, size=0x20000, flags=0x5, page_offset=0x0),
         Mapping('/lib/i386-linux-gnu/ld-2.19.so', start=0xf7712000, stop=0xf7713000, size=0x1000, flags=0x4, page_offset=0x20),
         Mapping('/lib/i386-linux-gnu/ld-2.19.so', start=0xf7713000, stop=0xf7714000, size=0x1000, flags=0x6, page_offset=0x21),
         Mapping('[stack]', start=0xfff3e000, stop=0xfff61000, size=0x23000, flags=0x6, page_offset=0x0)]

    Examples:

        Let's build an example binary which should eat ``R0=0xdeadbeef``
        and ``PC=0xcafebabe``.

        If we run the binary and then wait for it to exit, we can get its
        core file.

        >>> context.clear(arch='arm')
        >>> shellcode = shellcraft.mov('r0', 0xdeadbeef)
        >>> shellcode += shellcraft.mov('r1', 0xcafebabe)
        >>> shellcode += 'bx r1'
        >>> address = 0x41410000
        >>> elf = ELF.from_assembly(shellcode, vma=address)
        >>> io = elf.process(env={'HELLO': 'WORLD'})
        >>> io.poll(block=True)
        -11

        You can specify a full path a la ``Corefile('/path/to/core')``,
        but you can also just access the :attr:`.process.corefile` attribute.

        There's a lot of behind-the-scenes logic to locate the corefile for
        a given process, but it's all handled transparently by Pwntools.

        >>> core = io.corefile

        The core file has a :attr:`exe` property, which is a :class:`.Mapping`
        object.  Each mapping can be accessed with virtual addresses via subscript, or
        contents can be examined via the :attr:`.Mapping.data` attribute.

        >>> core.exe # doctest: +ELLIPSIS
        Mapping('/.../step3', start=..., stop=..., size=0x1000, flags=0x..., page_offset=...)
        >>> hex(core.exe.address)
        '0x41410000'

        The core file also has registers which can be accessed direclty.
        Pseudo-registers :attr:`pc` and :attr:`sp` are available on all architectures,
        to make writing architecture-agnostic code more simple.
        If this were an amd64 corefile, we could access e.g. ``core.rax``.

        >>> core.pc == 0xcafebabe
        True
        >>> core.r0 == 0xdeadbeef
        True
        >>> core.sp == core.r13
        True

        We may not always know which signal caused the core dump, or what address
        caused a segmentation fault.  Instead of accessing registers directly, we
        can also extract this information from the core dump via :attr:`fault_addr`
        and :attr:`signal`.

        On QEMU-generated core dumps, this information is unavailable, so we
        substitute the value of PC.  In our example, that's correct anyway.

        >>> core.fault_addr == 0xcafebabe
        True
        >>> core.signal
        11

        Core files can also be generated from running processes.
        This requires GDB to be installed, and can only be done with native processes.
        Getting a "complete" corefile requires GDB 7.11 or better.

        >>> elf = ELF(which('bash-static'))
        >>> context.clear(binary=elf)
        >>> env = dict(os.environ)
        >>> env['HELLO'] = 'WORLD'
        >>> io = process(elf.path, env=env)
        >>> io.sendline(b'echo hello')
        >>> io.recvline()
        b'hello\\n'

        The process is still running, but accessing its :attr:`.process.corefile` property
        automatically invokes GDB to attach and dump a corefile.

        >>> core = io.corefile
        >>> io.close()

        The corefile can be inspected and read from, and even exposes various mappings

        >>> core.exe # doctest: +ELLIPSIS
        Mapping('.../bin/bash-static', start=..., stop=..., size=..., flags=..., page_offset=...)
        >>> core.exe.data[0:4]
        b'\\x7fELF'

        It also supports all of the features of :class:`ELF`, so you can :meth:`.read`
        or :meth:`.write` or even the helpers like :meth:`.pack` or :meth:`.unpack`.

        Don't forget to call :meth:`.ELF.save` to save the changes to disk.

        >>> core.read(elf.address, 4)
        b'\\x7fELF'
        >>> core.pack(core.sp, 0xdeadbeef)
        >>> core.save()

        Let's re-load it as a new :attr:`Corefile` object and have a look!

        >>> core2 = Corefile(core.path)
        >>> hex(core2.unpack(core2.sp))
        '0xdeadbeef'

        Various other mappings are available by name, for the first segment of:

        * :attr:`.exe` the executable
        * :attr:`.libc` the loaded libc, if any
        * :attr:`.stack` the stack mapping
        * :attr:`.vvar`
        * :attr:`.vdso`
        * :attr:`.vsyscall`

        On Linux, 32-bit Intel binaries should have a VDSO section via :attr:`vdso`.  
        Since our ELF is statically linked, there is no libc which gets mapped.

        >>> core.vdso.data[:4]
        b'\\x7fELF'
        >>> core.libc

        But if we dump a corefile from a dynamically-linked binary, the :attr:`.libc`
        will be loaded.

        >>> process('bash').corefile.libc # doctest: +ELLIPSIS
        Mapping('.../libc...so...', start=0x..., stop=0x..., size=0x..., flags=..., page_offset=...)

        The corefile also contains a :attr:`.stack` property, which gives
        us direct access to the stack contents.  On Linux, the very top of the stack
        should contain two pointer-widths of NULL bytes, preceded by the NULL-
        terminated path to the executable (as passed via the first arg to ``execve``).

        >>> core.stack # doctest: +ELLIPSIS
        Mapping('[stack]', start=0x..., stop=0x..., size=0x..., flags=0x6, page_offset=0x0)

        When creating a process, the kernel puts the absolute path of the binary and some
        padding bytes at the end of the stack.  We can look at those by looking at 
        ``core.stack.data``.

        >>> size = len('/bin/bash-static') + 8
        >>> core.stack.data[-size:]
        b'bin/bash-static\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00'

        We can also directly access the environment variables and arguments, via
        :attr:`.argc`, :attr:`.argv`, and :attr:`.env`.

        >>> 'HELLO' in core.env
        True
        >>> core.string(core.env['HELLO'])
        b'WORLD'
        >>> core.getenv('HELLO')
        b'WORLD'
        >>> core.argc
        1
        >>> core.argv[0] in core.stack
        True
        >>> core.string(core.argv[0]) # doctest: +ELLIPSIS
        b'.../bin/bash-static'

        Corefiles can also be pulled from remote machines via SSH!

        >>> s = ssh(user='travis', host='example.pwnme', password='demopass')
        >>> _ = s.set_working_directory()
        >>> elf = ELF.from_assembly(shellcraft.trap())
        >>> path = s.upload(elf.path)
        >>> _ =s.chmod('+x', path)
        >>> io = s.process(path)
        >>> io.wait(1)
        -1
        >>> io.corefile.signal == signal.SIGTRAP # doctest: +SKIP
        True

        Make sure fault_addr synthesis works for amd64 on ret.

        >>> context.clear(arch='amd64')
        >>> elf = ELF.from_assembly('push 1234; ret')
        >>> io = elf.process()
        >>> io.wait(1)
        >>> io.corefile.fault_addr
        1234

        Corefile.getenv() works correctly, even if the environment variable's
        value contains embedded '='. Corefile is able to find the stack, even
        if the stack pointer doesn't point at the stack.

        >>> elf = ELF.from_assembly(shellcraft.crash())
        >>> io = elf.process(env={'FOO': 'BAR=BAZ'})
        >>> io.wait(1)
        >>> core = io.corefile
        >>> core.getenv('FOO')
        b'BAR=BAZ'
        >>> core.sp == 0
        True
        >>> core.sp in core.stack
        False

        Corefile gracefully handles the stack being filled with garbage, including
        argc / argv / envp being overwritten.

        >>> context.clear(arch='i386')
        >>> assembly = '''
        ... LOOP:
        ...   mov dword ptr [esp], 0x41414141
        ...   pop eax
        ...   jmp LOOP
        ... '''
        >>> elf = ELF.from_assembly(assembly)
        >>> io = elf.process()
        >>> io.wait(2)
        >>> core = io.corefile
        >>> core.argc, core.argv, core.env
        (0, [], {})
        >>> core.stack.data.endswith(b'AAAA')
        True
        >>> core.fault_addr == core.sp
        True
    """
    _fill_gaps = False

    def __init__(self, *a, **kw):
        if False:
            print('Hello World!')
        self.prstatus = None
        self.prpsinfo = None
        self.siginfo = None
        self.mappings = []
        self.stack = None
        '\n        Environment variables read from the stack.\n        Keys are the environment variable name, values are the memory \n        address of the variable.\n        \n        Use :meth:`.getenv` or :meth:`.string` to retrieve the textual value.\n        \n        Note: If ``FOO=BAR`` is in the environment, ``self.env[\'FOO\']`` is the address of the string ``"BAR\x00"``.\n        '
        self.env = {}
        self.envp_address = 0
        self.argv = []
        self.argv_address = 0
        self.argc = 0
        self.argc_address = 0
        self.at_execfn = 0
        self.at_entry = 0
        try:
            super(Corefile, self).__init__(*a, **kw)
        except IOError:
            log.warning('No corefile.  Have you set /proc/sys/kernel/core_pattern?')
            raise
        self.load_addr = 0
        self._address = 0
        if self.elftype != 'CORE':
            log.error('%s is not a valid corefile' % self.file.name)
        if self.arch not in prstatus_types:
            log.warn_once('%s does not use a supported corefile architecture, registers are unavailable' % self.file.name)
        prstatus_type = prstatus_types.get(self.arch)
        prpsinfo_type = prpsinfo_types.get(self.bits)
        siginfo_type = siginfo_types.get(self.bits)
        with log.waitfor('Parsing corefile...') as w:
            self._load_mappings()
            for segment in self.segments:
                if not isinstance(segment, elftools.elf.segments.NoteSegment):
                    continue
                for note in iter_notes(segment):
                    if not isinstance(note.n_desc, bytes):
                        note['n_desc'] = note.n_desc.encode('latin1')
                    if prstatus_type and note.n_descsz == ctypes.sizeof(prstatus_type) and (note.n_type in ('NT_GNU_ABI_TAG', 'NT_PRSTATUS')):
                        self.NT_PRSTATUS = note
                        self.prstatus = prstatus_type.from_buffer_copy(note.n_desc)
                    if prpsinfo_type and note.n_descsz == ctypes.sizeof(prpsinfo_type) and (note.n_type in ('NT_GNU_ABI_TAG', 'NT_PRPSINFO')):
                        self.NT_PRPSINFO = note
                        self.prpsinfo = prpsinfo_type.from_buffer_copy(note.n_desc)
                    if note.n_type in (1397311305, 'NT_SIGINFO'):
                        self.NT_SIGINFO = note
                        self.siginfo = siginfo_type.from_buffer_copy(note.n_desc)
                    if note.n_type in (constants.NT_FILE, 'NT_FILE'):
                        with context.local(bytes=self.bytes):
                            self._parse_nt_file(note)
                    if note.n_type in (constants.NT_AUXV, 'NT_AUXV'):
                        self.NT_AUXV = note
                        with context.local(bytes=self.bytes):
                            self._parse_auxv(note)
            if not self.stack and self.mappings:
                self.stack = self.mappings[-1].stop
            if self.stack and self.mappings:
                for mapping in self.mappings:
                    if self.stack in mapping or self.stack == mapping.stop:
                        mapping.name = '[stack]'
                        self.stack = mapping
                        break
                else:
                    log.warn('Could not find the stack!')
                    self.stack = None
            with context.local(bytes=self.bytes):
                try:
                    self._parse_stack()
                except ValueError:
                    pass
            self.exe
            self._describe_core()

    def _parse_nt_file(self, note):
        if False:
            print('Hello World!')
        t = tube()
        t.unrecv(note.n_desc)
        count = t.unpack()
        page_size = t.unpack()
        starts = []
        addresses = {}
        for i in range(count):
            start = t.unpack()
            end = t.unpack()
            offset = t.unpack()
            starts.append((start, offset))
        for i in range(count):
            filename = t.recvuntil(b'\x00', drop=True)
            if not isinstance(filename, str):
                filename = filename.decode('utf-8')
            (start, offset) = starts[i]
            for mapping in self.mappings:
                if mapping.start == start:
                    mapping.name = filename
                    mapping.page_offset = offset
        self.mappings = sorted(self.mappings, key=lambda m: m.start)
        vvar = vdso = vsyscall = False
        for mapping in reversed(self.mappings):
            if mapping.name:
                continue
            if not vsyscall and mapping.start == 18446744073699065856:
                mapping.name = '[vsyscall]'
                vsyscall = True
                continue
            if mapping.start == self.at_sysinfo_ehdr or (not vdso and mapping.size in [4096, 8192] and (mapping.flags == 5) and (self.read(mapping.start, 4) == b'\x7fELF')):
                mapping.name = '[vdso]'
                vdso = True
                continue
            if not vvar and mapping.size == 8192 and (mapping.flags == 4):
                mapping.name = '[vvar]'
                vvar = True
                continue

    @property
    def vvar(self):
        if False:
            print('Hello World!')
        ':class:`Mapping`: Mapping for the vvar section'
        for m in self.mappings:
            if m.name == '[vvar]':
                return m

    @property
    def vdso(self):
        if False:
            print('Hello World!')
        ':class:`Mapping`: Mapping for the vdso section'
        for m in self.mappings:
            if m.name == '[vdso]':
                return m

    @property
    def vsyscall(self):
        if False:
            return 10
        ':class:`Mapping`: Mapping for the vsyscall section'
        for m in self.mappings:
            if m.name == '[vsyscall]':
                return m

    @property
    def libc(self):
        if False:
            while True:
                i = 10
        ':class:`Mapping`: First mapping for ``libc.so``'
        expr = '^libc\\b.*so(?:\\.6)?$'
        for m in self.mappings:
            if not m.name:
                continue
            basename = os.path.basename(m.name)
            if re.match(expr, basename):
                return m

    @property
    def exe(self):
        if False:
            while True:
                i = 10
        ':class:`Mapping`: First mapping for the executable file.'
        if not self.at_entry:
            return None
        first_segment_for_name = {}
        for m in self.mappings:
            first_segment_for_name.setdefault(m.name, m)
        for m in self.mappings:
            if m.start <= self.at_entry < m.stop:
                if not m.name and self.at_execfn:
                    m.name = self.string(self.at_execfn)
                    if not isinstance(m.name, str):
                        m.name = m.name.decode('utf-8')
                return first_segment_for_name.get(m.name, m)

    @property
    def pid(self):
        if False:
            i = 10
            return i + 15
        ':class:`int`: PID of the process which created the core dump.'
        if self.prstatus:
            return int(self.prstatus.pr_pid)

    @property
    def ppid(self):
        if False:
            i = 10
            return i + 15
        ':class:`int`: Parent PID of the process which created the core dump.'
        if self.prstatus:
            return int(self.prstatus.pr_ppid)

    @property
    def signal(self):
        if False:
            print('Hello World!')
        ':class:`int`: Signal which caused the core to be dumped.\n\n        Example:\n\n            >>> elf = ELF.from_assembly(shellcraft.trap())\n            >>> io = elf.process()\n            >>> io.wait(1)\n            >>> io.corefile.signal == signal.SIGTRAP\n            True\n\n            >>> elf = ELF.from_assembly(shellcraft.crash())\n            >>> io = elf.process()\n            >>> io.wait(1)\n            >>> io.corefile.signal == signal.SIGSEGV\n            True\n        '
        if self.siginfo:
            return int(self.siginfo.si_signo)
        if self.prstatus:
            return int(self.prstatus.pr_cursig)

    @property
    def fault_addr(self):
        if False:
            return 10
        ":class:`int`: Address which generated the fault, for the signals\n            SIGILL, SIGFPE, SIGSEGV, SIGBUS.  This is only available in native\n            core dumps created by the kernel.  If the information is unavailable,\n            this returns the address of the instruction pointer.\n\n\n        Example:\n\n            >>> elf = ELF.from_assembly('mov eax, 0xdeadbeef; jmp eax', arch='i386')\n            >>> io = elf.process()\n            >>> io.wait(1)\n            >>> io.corefile.fault_addr == io.corefile.eax == 0xdeadbeef\n            True\n        "
        if not self.siginfo:
            return getattr(self, 'pc', 0)
        fault_addr = int(self.siginfo.sigfault_addr)
        if fault_addr == 0 and self.siginfo.si_code == 128:
            try:
                code = self.read(self.pc, 1)
                RET = b'\xc3'
                if code == RET:
                    fault_addr = self.unpack(self.sp)
            except Exception:
                pass
        return fault_addr

    @property
    def _pc_register(self):
        if False:
            for i in range(10):
                print('nop')
        name = {'i386': 'eip', 'amd64': 'rip'}.get(self.arch, 'pc')
        return name

    @property
    def pc(self):
        if False:
            return 10
        ':class:`int`: The program counter for the Corefile\n\n        This is a cross-platform way to get e.g. ``core.eip``, ``core.rip``, etc.\n        '
        return self.registers.get(self._pc_register, None)

    @property
    def _sp_register(self):
        if False:
            for i in range(10):
                print('nop')
        name = {'i386': 'esp', 'amd64': 'rsp'}.get(self.arch, 'sp')
        return name

    @property
    def sp(self):
        if False:
            i = 10
            return i + 15
        ':class:`int`: The stack pointer for the Corefile\n\n        This is a cross-platform way to get e.g. ``core.esp``, ``core.rsp``, etc.\n        '
        return self.registers.get(self._sp_register, None)

    def _describe(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _describe_core(self):
        if False:
            print('Hello World!')
        gnu_triplet = '-'.join(map(str, (self.arch, self.bits, self.endian)))
        fields = [repr(self.path), '%-10s %s' % ('Arch:', gnu_triplet), '%-10s %#x' % ('%s:' % self._pc_register.upper(), self.pc or 0), '%-10s %#x' % ('%s:' % self._sp_register.upper(), self.sp or 0)]
        if self.exe and self.exe.name:
            fields += ['%-10s %s' % ('Exe:', '%r (%#x)' % (self.exe.name, self.exe.address))]
        if self.fault_addr:
            fields += ['%-10s %#x' % ('Fault:', self.fault_addr)]
        log.info_once('\n'.join(fields))

    def _load_mappings(self):
        if False:
            while True:
                i = 10
        for s in self.segments:
            if s.header.p_type != 'PT_LOAD':
                continue
            mapping = Mapping(self, None, s.header.p_vaddr, s.header.p_vaddr + s.header.p_memsz, s.header.p_flags, None)
            self.mappings.append(mapping)

    def _parse_auxv(self, note):
        if False:
            for i in range(10):
                print('nop')
        t = tube()
        t.unrecv(note.n_desc)
        for i in range(0, note.n_descsz, context.bytes * 2):
            key = t.unpack()
            value = t.unpack()
            if key == constants.AT_EXECFN:
                self.at_execfn = value
                value = value & ~4095
                value += 4096
                self.stack = value
            if key == constants.AT_ENTRY:
                self.at_entry = value
            if key == constants.AT_PHDR:
                self.at_phdr = value
            if key == constants.AT_BASE:
                self.at_base = value
            if key == constants.AT_SYSINFO_EHDR:
                self.at_sysinfo_ehdr = value

    def _parse_stack(self):
        if False:
            return 10
        stack = self.stack
        if not stack:
            return
        if not stack.data.endswith(b'\x00' * context.bytes):
            log.warn_once('End of the stack is corrupted, skipping stack parsing (got: %s)', enhex(self.data[-context.bytes:]))
            return
        if not self.at_execfn:
            address = stack.stop
            address -= 2 * self.bytes
            address -= 1
            address = stack.rfind(b'\x00', None, address)
            address += 1
            self.at_execfn = address
        address = self.at_execfn - 1
        try:
            if stack[address] != b'\x00':
                log.warning('Error parsing corefile stack: Could not find end of environment')
                return
        except ValueError:
            log.warning('Error parsing corefile stack: Address out of bounds')
            return
        address = stack.rfind(b'\x00', None, address)
        last_env_addr = address + 1
        p_last_env_addr = stack.find(pack(last_env_addr), None, last_env_addr)
        if p_last_env_addr < 0:
            log.warn_once('Error parsing corefile stack: Found bad environment at %#x', last_env_addr)
            return
        envp_nullterm = p_last_env_addr + context.bytes
        if self.unpack(envp_nullterm) != 0:
            log.warning('Error parsing corefile stack: Could not find end of environment variables')
            return
        p_end_of_argv = stack.rfind(pack(0), None, p_last_env_addr)
        self.envp_address = p_end_of_argv + self.bytes
        env_pointer_data = stack[self.envp_address:p_last_env_addr + self.bytes]
        for pointer in unpack_many(env_pointer_data):
            if pointer not in stack:
                continue
            try:
                name_value = self.string(pointer)
            except Exception:
                continue
            (name, _) = name_value.split(b'=', 1)
            end = pointer + len(name_value) + 1
            if end not in stack:
                continue
            if not isinstance(name, str):
                name = name.decode('utf-8', 'surrogateescape')
            self.env[name] = pointer + len(name) + len('=')
        address = p_end_of_argv - self.bytes
        while self.unpack(address) in stack:
            address -= self.bytes
        self.argc_address = address
        self.argc = self.unpack(self.argc_address)
        self.argv_address = self.argc_address + self.bytes
        self.argv = unpack_many(stack[self.argv_address:p_end_of_argv])

    @property
    def maps(self):
        if False:
            print('Hello World!')
        ":class:`str`: A printable string which is similar to /proc/xx/maps.\n\n        ::\n\n            >>> print(Corefile('./core').maps)\n            8048000-8049000 r-xp 1000 /home/user/pwntools/crash\n            8049000-804a000 r--p 1000 /home/user/pwntools/crash\n            804a000-804b000 rw-p 1000 /home/user/pwntools/crash\n            f7528000-f7529000 rw-p 1000 None\n            f7529000-f76d1000 r-xp 1a8000 /lib/i386-linux-gnu/libc-2.19.so\n            f76d1000-f76d2000 ---p 1000 /lib/i386-linux-gnu/libc-2.19.so\n            f76d2000-f76d4000 r--p 2000 /lib/i386-linux-gnu/libc-2.19.so\n            f76d4000-f76d5000 rw-p 1000 /lib/i386-linux-gnu/libc-2.19.so\n            f76d5000-f76d8000 rw-p 3000 None\n            f76ef000-f76f1000 rw-p 2000 None\n            f76f1000-f76f2000 r-xp 1000 [vdso]\n            f76f2000-f7712000 r-xp 20000 /lib/i386-linux-gnu/ld-2.19.so\n            f7712000-f7713000 r--p 1000 /lib/i386-linux-gnu/ld-2.19.so\n            f7713000-f7714000 rw-p 1000 /lib/i386-linux-gnu/ld-2.19.so\n            fff3e000-fff61000 rw-p 23000 [stack]\n        "
        return '\n'.join(map(str, self.mappings))

    def getenv(self, name):
        if False:
            return 10
        "getenv(name) -> int\n\n        Read an environment variable off the stack, and return its contents.\n\n        Arguments:\n            name(str): Name of the environment variable to read.\n\n        Returns:\n            :class:`str`: The contents of the environment variable.\n\n        Example:\n\n            >>> elf = ELF.from_assembly(shellcraft.trap())\n            >>> io = elf.process(env={'GREETING': 'Hello!'})\n            >>> io.wait(1)\n            >>> io.corefile.getenv('GREETING')\n            b'Hello!'\n        "
        if not isinstance(name, str):
            name = name.decode('utf-8', 'surrogateescape')
        if name not in self.env:
            log.error('Environment variable %r not set' % name)
        return self.string(self.env[name])

    @property
    def registers(self):
        if False:
            i = 10
            return i + 15
        ":class:`dict`: All available registers in the coredump.\n\n        Example:\n\n            >>> elf = ELF.from_assembly('mov eax, 0xdeadbeef;' + shellcraft.trap(), arch='i386')\n            >>> io = elf.process()\n            >>> io.wait(1)\n            >>> io.corefile.registers['eax'] == 0xdeadbeef\n            True\n        "
        if not self.prstatus:
            return {}
        rv = {}
        for k in dir(self.prstatus.pr_reg):
            if k.startswith('_'):
                continue
            try:
                rv[k] = int(getattr(self.prstatus.pr_reg, k))
            except Exception:
                pass
        return rv

    def debug(self):
        if False:
            print('Hello World!')
        'Open the corefile under a debugger.'
        import pwnlib.gdb
        pwnlib.gdb.attach(self, exe=self.exe.path)

    def __getattr__(self, attribute):
        if False:
            i = 10
            return i + 15
        if attribute.startswith('_') or not self.prstatus:
            raise AttributeError(attribute)
        if hasattr(self.prstatus, attribute):
            return getattr(self.prstatus, attribute)
        return getattr(self.prstatus.pr_reg, attribute)

    def _populate_got(*a):
        if False:
            i = 10
            return i + 15
        pass

    def _populate_plt(*a):
        if False:
            return 10
        pass

class Core(Corefile):
    """Alias for :class:`.Corefile`"""

class Coredump(Corefile):
    """Alias for :class:`.Corefile`"""

class CorefileFinder(object):

    def __init__(self, proc):
        if False:
            i = 10
            return i + 15
        if proc.poll() is None:
            log.error('Process %i has not exited' % proc.pid)
        self.process = proc
        self.pid = proc.pid
        self.uid = proc.suid
        self.gid = proc.sgid
        self.exe = proc.executable
        self.basename = os.path.basename(self.exe)
        self.cwd = proc.cwd
        if isinstance(proc, process):
            self.read = read
            self.unlink = os.unlink
        elif isinstance(proc, ssh_channel):
            self.read = proc.parent.read
            self.unlink = proc.parent.unlink
        self.kernel_core_pattern = self.read('/proc/sys/kernel/core_pattern').strip()
        self.kernel_core_uses_pid = bool(int(self.read('/proc/sys/kernel/core_uses_pid')))
        log.debug('core_pattern: %r' % self.kernel_core_pattern)
        log.debug('core_uses_pid: %r' % self.kernel_core_uses_pid)
        self.interpreter = self.binfmt_lookup()
        log.debug('interpreter: %r' % self.interpreter)
        core_path = 'core.%i' % proc.pid
        self.core_path = None
        if os.path.isfile(core_path):
            log.debug('Found core immediately: %r' % core_path)
            self.core_path = core_path
        if not self.core_path:
            log.debug('Looking for QEMU corefile')
            self.core_path = self.qemu_corefile()
        if not self.core_path:
            log.debug('Looking for native corefile')
            self.core_path = self.native_corefile()
        if not self.core_path:
            return
        core_pid = self.load_core_check_pid()
        if context.rename_corefiles:
            new_path = 'core.%i' % core_pid
            if core_pid > 0 and new_path != self.core_path:
                write(new_path, self.read(self.core_path))
                try:
                    self.unlink(self.core_path)
                except (IOError, OSError):
                    log.warn('Could not delete %r' % self.core_path)
                self.core_path = new_path
        if core_pid != self.pid:
            log.warn('Corefile PID does not match! (got %i)' % core_pid)
        elif context.delete_corefiles:
            atexit.register(lambda : os.unlink(self.core_path))

    def load_core_check_pid(self):
        if False:
            i = 10
            return i + 15
        "Test whether a Corefile matches our process\n\n        Speculatively load a Corefile without informing the user, so that we\n        can check if it matches the process we're looking for.\n\n        Arguments:\n            path(str): Path to the corefile on disk\n\n        Returns:\n            `bool`: ``True`` if the Corefile matches, ``False`` otherwise.\n        "
        try:
            with context.quiet:
                with tempfile.NamedTemporaryFile() as tmp:
                    tmp.write(self.read(self.core_path))
                    tmp.flush()
                    return Corefile(tmp.name).pid
        except Exception:
            pass
        return -1

    def apport_corefile(self):
        if False:
            print('Hello World!')
        "Find the apport crash for the process, and extract the core file.\n\n        Arguments:\n            process(process): Process object we're looking for.\n\n        Returns:\n            `str`: Raw core file contents\n        "
        crash_data = self.apport_read_crash_data()
        log.debug('Apport Crash Data:\n%s' % crash_data)
        if crash_data:
            return self.apport_crash_extract_corefile(crash_data)

    def apport_crash_extract_corefile(self, crashfile_data):
        if False:
            return 10
        'Extract a corefile from an apport crash file contents.\n\n        Arguments:\n            crashfile_data(str): Crash file contents\n\n        Returns:\n            `str`: Raw binary data for the core file, or ``None``.\n        '
        file = StringIO(crashfile_data)
        for line in file:
            if line.startswith(' Pid:'):
                pid = int(line.split()[-1])
                if pid == self.pid:
                    break
        else:
            return
        for line in file:
            if line.startswith('CoreDump: base64'):
                break
        else:
            return
        chunks = []
        for line in file:
            if not line.startswith(' '):
                break
            chunks.append(b64d(line))
        compressed_data = b''.join(chunks)
        compressed_file = BytesIO(compressed_data)
        gzip_file = gzip.GzipFile(fileobj=compressed_file)
        core_data = gzip_file.read()
        return core_data

    def apport_read_crash_data(self):
        if False:
            while True:
                i = 10
        'Find the apport crash for the process\n\n        Returns:\n            `str`: Raw contents of the crash file or ``None``.\n        '
        uid = self.uid
        crash_name = self.exe.replace('/', '_')
        crash_path = '/var/crash/%s.%i.crash' % (crash_name, uid)
        try:
            log.debug('Looking for Apport crash at %r' % crash_path)
            data = self.read(crash_path)
        except Exception:
            return None
        try:
            self.unlink(crash_path)
        except Exception:
            pass
        return data

    def systemd_coredump_corefile(self):
        if False:
            for i in range(10):
                print('nop')
        "Find the systemd-coredump crash for the process and dump it to a file.\n\n        Arguments:\n            process(process): Process object we're looking for.\n\n        Returns:\n            `str`: Filename of core file, if coredump was found.\n        "
        filename = 'core.%s.%i.coredumpctl' % (self.basename, self.pid)
        try:
            subprocess.check_call(['coredumpctl', 'dump', '--output=%s' % filename, str(self.pid)], stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT, shell=False)
            return filename
        except subprocess.CalledProcessError as e:
            log.debug('coredumpctl failed with status: %d' % e.returncode)

    def native_corefile(self):
        if False:
            i = 10
            return i + 15
        'Find the corefile for a native crash.\n\n        Arguments:\n            process(process): Process whose crash we should find.\n\n        Returns:\n            `str`: Filename of core file.\n        '
        if self.kernel_core_pattern.startswith(b'|'):
            log.debug('Checking for corefile (piped)')
            return self.native_corefile_pipe()
        log.debug('Checking for corefile (pattern)')
        return self.native_corefile_pattern()

    def native_corefile_pipe(self):
        if False:
            return 10
        'Find the corefile for a piped core_pattern\n\n        Supports apport and systemd-coredump.\n\n        Arguments:\n            process(process): Process whose crash we should find.\n\n        Returns:\n            `str`: Filename of core file.\n        '
        if b'/apport' in self.kernel_core_pattern:
            log.debug('Found apport in core_pattern')
            apport_core = self.apport_corefile()
            if apport_core:
                filename = 'core.%s.%i.apport' % (self.basename, self.pid)
                with open(filename, 'wb+') as f:
                    f.write(apport_core)
                return filename
            filename = self.apport_coredump()
            if filename:
                return filename
            self.kernel_core_pattern = 'core'
            return self.native_corefile_pattern()
        elif b'systemd-coredump' in self.kernel_core_pattern:
            log.debug('Found systemd-coredump in core_pattern')
            return self.systemd_coredump_corefile()
        else:
            log.warn_once('Unsupported core_pattern: %r', self.kernel_core_pattern)
            return None

    def native_corefile_pattern(self):
        if False:
            i = 10
            return i + 15
        "\n        %%  a single % character\n        %c  core file size soft resource limit of crashing process (since Linux 2.6.24)\n        %d  dump mode—same as value returned by prctl(2) PR_GET_DUMPABLE (since Linux 3.7)\n        %e  executable filename (without path prefix)\n        %E  pathname of executable, with slashes ('/') replaced by exclamation marks ('!') (since Linux 3.0).\n        %g  (numeric) real GID of dumped process\n        %h  hostname (same as nodename returned by uname(2))\n        %i  TID of thread that triggered core dump, as seen in the PID namespace in which the thread resides (since Linux 3.18)\n        %I  TID of thread that triggered core dump, as seen in the initial PID namespace (since Linux 3.18)\n        %p  PID of dumped process, as seen in the PID namespace in which the process resides\n        %P  PID of dumped process, as seen in the initial PID namespace (since Linux 3.12)\n        %s  number of signal causing dump\n        %t  time of dump, expressed as seconds since the Epoch, 1970-01-01 00:00:00 +0000 (UTC)\n        %u  (numeric) real UID of dumped process\n        "
        replace = {'%%': '%', '%e': os.path.basename(self.interpreter) or self.basename, '%E': self.exe.replace('/', '!'), '%g': str(self.gid), '%h': socket.gethostname(), '%i': str(self.pid), '%I': str(self.pid), '%p': str(self.pid), '%P': str(self.pid), '%s': str(-self.process.poll()), '%u': str(self.uid)}
        replace = dict(((re.escape(k), v) for (k, v) in replace.items()))
        pattern = re.compile('|'.join(replace.keys()))
        if not hasattr(self.kernel_core_pattern, 'encode'):
            self.kernel_core_pattern = self.kernel_core_pattern.decode('utf-8')
        core_pattern = self.kernel_core_pattern
        corefile_path = pattern.sub(lambda m: replace[re.escape(m.group(0))], core_pattern)
        if self.kernel_core_uses_pid:
            corefile_path += '.%i' % self.pid
        if os.pathsep not in corefile_path:
            corefile_path = os.path.join(self.cwd, corefile_path)
        log.debug('Trying corefile_path: %r' % corefile_path)
        try:
            self.read(corefile_path)
            return corefile_path
        except Exception as e:
            log.debug('No dice: %s' % e)

    def qemu_corefile(self):
        if False:
            i = 10
            return i + 15
        'qemu_corefile() -> str\n\n        Retrieves the path to a QEMU core dump.\n        '
        corefile_name = 'qemu_{basename}_*_{pid}.core'
        corefile_name = corefile_name.format(basename=self.basename, pid=self.pid)
        corefile_path = os.path.join(self.cwd, corefile_name)
        log.debug('Trying corefile_path: %r' % corefile_path)
        for corefile in sorted(glob.glob(corefile_path), reverse=True):
            return corefile

    def apport_coredump(self):
        if False:
            i = 10
            return i + 15
        'Find new-style apport coredump of executables not belonging\n        to a system package\n        '
        boot_id = read('/proc/sys/kernel/random/boot_id').strip().decode()
        path = self.exe.replace('/', '_')
        corefile_name = 'core.{path}.{uid}.{boot_id}.{pid}.*'.format(path=path, uid=self.uid, boot_id=boot_id, pid=self.pid)
        corefile_path = os.path.join('/var/lib/apport/coredump', corefile_name)
        log.debug('Trying corefile_path: %r' % corefile_path)
        for corefile in sorted(glob.glob(corefile_path), reverse=True):
            return corefile

    def binfmt_lookup(self):
        if False:
            i = 10
            return i + 15
        'Parses /proc/sys/fs/binfmt_misc to find the interpreter for a file'
        binfmt_misc = '/proc/sys/fs/binfmt_misc'
        if not isinstance(self.process, process):
            log.debug('Not a process')
            return ''
        if self.process._qemu:
            return self.process._qemu
        if not os.path.isdir(binfmt_misc):
            log.debug('No binfmt_misc dir')
            return ''
        exe_data = bytearray(self.read(self.exe))
        for entry in os.listdir(binfmt_misc):
            keys = {}
            path = os.path.join(binfmt_misc, entry)
            try:
                data = self.read(path).decode()
            except Exception:
                continue
            for line in data.splitlines():
                try:
                    (k, v) = line.split(None)
                except ValueError:
                    continue
                keys[k] = v
            if 'magic' not in keys:
                continue
            magic = bytearray(unhex(keys['magic']))
            mask = bytearray(b'\xff' * len(magic))
            if 'mask' in keys:
                mask = bytearray(unhex(keys['mask']))
            for (i, mag) in enumerate(magic):
                if exe_data[i] & mask[i] != mag:
                    break
            else:
                return keys['interpreter']
        return ''