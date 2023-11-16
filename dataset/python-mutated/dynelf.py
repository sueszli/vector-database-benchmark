"""
Resolve symbols in loaded, dynamically-linked ELF binaries.
Given a function which can leak data at an arbitrary address,
any symbol in any loaded library can be resolved.

Example
^^^^^^^^

::

    # Assume a process or remote connection
    p = process('./pwnme')

    # Declare a function that takes a single address, and
    # leaks at least one byte at that address.
    def leak(address):
        data = p.read(address, 4)
        log.debug("%#x => %s", address, enhex(data or ''))
        return data

    # For the sake of this example, let's say that we
    # have any of these pointers.  One is a pointer into
    # the target binary, the other two are pointers into libc
    main   = 0xfeedf4ce
    libc   = 0xdeadb000
    system = 0xdeadbeef

    # With our leaker, and a pointer into our target binary,
    # we can resolve the address of anything.
    #
    # We do not actually need to have a copy of the target
    # binary for this to work.
    d = DynELF(leak, main)
    assert d.lookup(None,     'libc') == libc
    assert d.lookup('system', 'libc') == system

    # However, if we *do* have a copy of the target binary,
    # we can speed up some of the steps.
    d = DynELF(leak, main, elf=ELF('./pwnme'))
    assert d.lookup(None,     'libc') == libc
    assert d.lookup('system', 'libc') == system

    # Alternately, we can resolve symbols inside another library,
    # given a pointer into it.
    d = DynELF(leak, libc + 0x1234)
    assert d.lookup('system')      == system

DynELF
"""
from __future__ import absolute_import
from __future__ import division
import ctypes
from elftools.elf.enums import ENUM_D_TAG
from pwnlib import elf
from pwnlib import libcdb
from pwnlib.context import context
from pwnlib.elf import ELF
from pwnlib.elf import constants
from pwnlib.log import getLogger
from pwnlib.memleak import MemLeak
from pwnlib.util.fiddling import enhex
from pwnlib.util.packing import _need_bytes
log = getLogger(__name__)
sizeof = ctypes.sizeof

def sysv_hash(symbol):
    if False:
        return 10
    'sysv_hash(str) -> int\n\n    Function used to generate SYSV-style hashes for strings.\n    '
    h = 0
    g = 0
    for c in bytearray(_need_bytes(symbol, 4, 128)):
        h = (h << 4) + c
        g = h & 4026531840
        h ^= g >> 24
        h &= ~g
    return h & 4294967295

def gnu_hash(s):
    if False:
        print('Hello World!')
    'gnu_hash(str) -> int\n\n    Function used to generated GNU-style hashes for strings.\n    '
    s = bytearray(_need_bytes(s, 4, 128))
    h = 5381
    for c in s:
        h = h * 33 + c
    return h & 4294967295

class DynELF(object):
    """
    DynELF knows how to resolve symbols in remote processes via an infoleak or
    memleak vulnerability encapsulated by :class:`pwnlib.memleak.MemLeak`.

    Implementation Details:

        Resolving Functions:

            In all ELFs which export symbols for importing by other libraries,
            (e.g. ``libc.so``) there are a series of tables which give exported
            symbol names, exported symbol addresses, and the ``hash`` of those
            exported symbols.  By applying a hash function to the name of the
            desired symbol (e.g., ``'printf'``), it can be located in the hash
            table.  Its location in the hash table provides an index into the
            string name table (strtab_), and the symbol address (symtab_).

            Assuming we have the base address of ``libc.so``, the way to resolve
            the address of ``printf`` is to locate the ``symtab``, ``strtab``,
            and hash table. The string ``"printf"`` is hashed according to the
            style of the hash table (SYSV_ or GNU_), and the hash table is
            walked until a matching entry is located. We can verify an exact
            match by checking the string table, and then get the offset into
            ``libc.so`` from the ``symtab``.

        Resolving Library Addresses:

            If we have a pointer into a dynamically-linked executable, we can
            leverage an internal linker structure called the `link map`_. This
            is a linked list structure which contains information about each
            loaded library, including its full path and base address.

            A pointer to the ``link map`` can be found in two ways.  Both are
            referenced from entries in the DYNAMIC_ array.

            - In non-RELRO binaries, a pointer is placed in the `.got.plt`_ area
              in the binary. This is marked by finding the DT_PLTGOT_ area in the
              binary.
            - In all binaries, a pointer can be found in the area described by
              the DT_DEBUG_ area.  This exists even in stripped binaries.

            For maximum flexibility, both mechanisms are used exhaustively.

    .. _symtab:    https://refspecs.linuxbase.org/elf/gabi4+/ch4.symtab.html
    .. _strtab:    https://refspecs.linuxbase.org/elf/gabi4+/ch4.strtab.html
    .. _.got.plt:  https://refspecs.linuxbase.org/LSB_3.1.1/LSB-Core-generic/LSB-Core-generic/specialsections.html
    .. _DYNAMIC:   http://www.sco.com/developers/gabi/latest/ch5.dynamic.html#dynamic_section
    .. _SYSV:      https://refspecs.linuxbase.org/elf/gabi4+/ch5.dynamic.html#hash
    .. _GNU:       https://blogs.oracle.com/solaris/post/gnu-hash-elf-sections
    .. _DT_DEBUG:  https://reverseengineering.stackexchange.com/questions/6525/elf-link-map-when-linked-as-relro
    .. _link map:  https://sourceware.org/git/?p=glibc.git;a=blob;f=elf/link.h;h=eaca8028e45a859ac280301a6e955a14eed1b887;hb=HEAD#l84
    .. _DT_PLTGOT: https://refspecs.linuxfoundation.org/ELF/zSeries/lzsabi0_zSeries/x2251.html
    """

    def __init__(self, leak, pointer=None, elf=None, libcdb=True):
        if False:
            print('Hello World!')
        '\n        Instantiates an object which can resolve symbols in a running binary\n        given a :class:`pwnlib.memleak.MemLeak` leaker and a pointer inside\n        the binary.\n\n        Arguments:\n            leak(MemLeak): Instance of pwnlib.memleak.MemLeak for leaking memory\n            pointer(int):  A pointer into a loaded ELF file\n            elf(str,ELF):  Path to the ELF file on disk, or a loaded :class:`pwnlib.elf.ELF`.\n            libcdb(bool):  Attempt to use libcdb to speed up libc lookups\n        '
        self.libcdb = libcdb
        self._elfclass = None
        self._elftype = None
        self._link_map = None
        self._waitfor = None
        self._bases = {}
        self._dynamic = None
        if not (pointer or (elf and elf.address)):
            log.error('Must specify either a pointer into a module and/or an ELF file with a valid base address')
        pointer = pointer or elf.address
        if not isinstance(leak, MemLeak):
            leak = MemLeak(leak)
        if not elf:
            log.warn_once('No ELF provided.  Leaking is much faster if you have a copy of the ELF being leaked.')
        self.elf = elf
        self.leak = leak
        self.libbase = self._find_base(pointer or elf.address)
        if elf:
            self._find_linkmap_assisted(elf)

    @classmethod
    def for_one_lib_only(cls, leak, ptr):
        if False:
            for i in range(10):
                print('nop')
        return cls(leak, ptr)

    @classmethod
    def from_lib_ptr(cls, leak, ptr):
        if False:
            while True:
                i = 10
        return cls(leak, ptr)

    @staticmethod
    def find_base(leak, ptr):
        if False:
            return 10
        'Given a :class:`pwnlib.memleak.MemLeak` object and a pointer into a\n        library, find its base address.\n        '
        return DynELF(leak, ptr).libbase

    @property
    def elfclass(self):
        if False:
            while True:
                i = 10
        '32 or 64'
        if not self._elfclass:
            elfclass = self.leak.field(self.libbase, elf.Elf_eident.EI_CLASS)
            self._elfclass = {constants.ELFCLASS32: 32, constants.ELFCLASS64: 64}[elfclass]
        return self._elfclass

    @property
    def elftype(self):
        if False:
            print('Hello World!')
        "e_type from the elf header. In practice the value will almost always\n        be 'EXEC' or 'DYN'. If the value is architecture-specific (between\n        ET_LOPROC and ET_HIPROC) or invalid, KeyError is raised.\n        "
        if not self._elftype:
            Ehdr = {32: elf.Elf32_Ehdr, 64: elf.Elf64_Ehdr}[self.elfclass]
            elftype = self.leak.field(self.libbase, Ehdr.e_type)
            self._elftype = {constants.ET_NONE: 'NONE', constants.ET_REL: 'REL', constants.ET_EXEC: 'EXEC', constants.ET_DYN: 'DYN', constants.ET_CORE: 'CORE'}[elftype]
        return self._elftype

    @property
    def link_map(self):
        if False:
            print('Hello World!')
        'Pointer to the runtime link_map object'
        if not self._link_map:
            self._link_map = self._find_linkmap()
        return self._link_map

    @property
    def dynamic(self):
        if False:
            return 10
        '\n        Returns:\n            Pointer to the ``.DYNAMIC`` area.\n        '
        if not self._dynamic:
            self._dynamic = self._find_dynamic_phdr()
        return self._dynamic

    def _find_linkmap_assisted(self, path):
        if False:
            i = 10
            return i + 15
        'Uses an ELF file to assist in finding the link_map.\n        '
        if isinstance(path, ELF):
            path = path.path
        with context.local(log_level='error'):
            elf = ELF(path)
        elf.address = self.libbase
        w = self.waitfor('Loading from %r' % elf.path)
        real_leak = self.leak

        @MemLeak
        def fake_leak(address):
            if False:
                print('Hello World!')
            try:
                return elf.read(address, 4)
            except ValueError:
                return real_leak.b(address)
        self.leak = fake_leak
        w.status('Searching for DT_PLTGOT')
        pltgot = self._find_dt(constants.DT_PLTGOT)
        w.status('Searching for DT_DEBUG')
        debug = self._find_dt(constants.DT_DEBUG)
        self.leak = real_leak
        self._find_linkmap(pltgot, debug)
        self.success('Done')

    def _find_base(self, ptr):
        if False:
            for i in range(10):
                print('nop')
        page_size = 4096
        page_mask = ~(page_size - 1)
        ptr &= page_mask
        w = None
        while True:
            if self.leak.compare(ptr, b'\x7fELF'):
                break
            fast = self._find_base_optimized(ptr)
            if fast:
                ptr = fast
                continue
            ptr -= page_size
            if ptr < 0:
                raise ValueError('Address is negative, something is wrong!')
            w = w or self.waitfor('Finding base address')
            self.status('%#x' % ptr)
        if w:
            self.success('%#x' % ptr)
        return ptr

    def _find_base_optimized(self, ptr):
        if False:
            print('Hello World!')
        if not self.elf:
            return None
        ptr += 32
        data = self.leak.n(ptr, 32)
        if not data:
            return None
        matches = list(self.elf.search(data))
        if len(matches) != 1:
            return None
        candidate = matches[0]
        candidate -= self.elf.address
        if candidate & 4095 != 32:
            return None
        ptr -= candidate
        return ptr

    def _find_dynamic_phdr(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the address of the first Program Header with the type\n        PT_DYNAMIC.\n        '
        leak = self.leak
        base = self.libbase
        Ehdr = {32: elf.Elf32_Ehdr, 64: elf.Elf64_Ehdr}[self.elfclass]
        Phdr = {32: elf.Elf32_Phdr, 64: elf.Elf64_Phdr}[self.elfclass]
        self.status('PT_DYNAMIC')
        phead = base + leak.field(base, Ehdr.e_phoff)
        self.status('PT_DYNAMIC header = %#x' % phead)
        phnum = leak.field(base, Ehdr.e_phnum)
        self.status('PT_DYNAMIC count = %#x' % phnum)
        for i in range(phnum):
            if leak.field_compare(phead, Phdr.p_type, constants.PT_DYNAMIC):
                break
            phead += sizeof(Phdr)
        else:
            self.failure('Could not find Program Header of type PT_DYNAMIC')
            return None
        dynamic = leak.field(phead, Phdr.p_vaddr)
        self.status('PT_DYNAMIC @ %#x' % dynamic)
        dynamic = self._make_absolute_ptr(dynamic)
        return dynamic

    def _find_dt(self, tag):
        if False:
            print('Hello World!')
        '\n        Find an entry in the DYNAMIC array.\n\n        Arguments:\n            tag(int): Single tag to find\n\n        Returns:\n            Pointer to the data described by the specified entry.\n        '
        leak = self.leak
        base = self.libbase
        dynamic = self.dynamic
        name = next((k for (k, v) in ENUM_D_TAG.items() if v == tag))
        Dyn = {32: elf.Elf32_Dyn, 64: elf.Elf64_Dyn}[self.elfclass]
        while not leak.field_compare(dynamic, Dyn.d_tag, constants.DT_NULL):
            if leak.field_compare(dynamic, Dyn.d_tag, tag):
                break
            dynamic += sizeof(Dyn)
        else:
            self.failure('Could not find tag %s' % name)
            return None
        self.status('Found %s at %#x' % (name, dynamic))
        ptr = leak.field(dynamic, Dyn.d_ptr)
        ptr = self._make_absolute_ptr(ptr)
        return ptr

    def _find_linkmap(self, pltgot=None, debug=None):
        if False:
            print('Hello World!')
        '\n        The linkmap is a chained structure created by the loader at runtime\n        which contains information on the names and load addresses of all\n        libraries.\n\n        For non-RELRO binaries, a pointer to this is stored in the .got.plt\n        area.\n\n        For RELRO binaries, a pointer is additionally stored in the DT_DEBUG\n        area.\n        '
        w = self.waitfor('Finding linkmap')
        Got = {32: elf.Elf_i386_GOT, 64: elf.Elf_x86_64_GOT}[self.elfclass]
        r_debug = {32: elf.Elf32_r_debug, 64: elf.Elf64_r_debug}[self.elfclass]
        linkmap = None
        if not pltgot:
            w.status('Finding linkmap: DT_PLTGOT')
            pltgot = self._find_dt(constants.DT_PLTGOT)
        if pltgot:
            w.status('GOT.linkmap')
            linkmap = self.leak.field(pltgot, Got.linkmap)
            w.status('GOT.linkmap %#x' % linkmap)
        if not linkmap:
            debug = debug or self._find_dt(constants.DT_DEBUG)
            if debug:
                w.status('r_debug.linkmap')
                linkmap = self.leak.field(debug, r_debug.r_map)
                w.status('r_debug.linkmap %#x' % linkmap)
        if not linkmap:
            w.failure('Could not find DT_PLTGOT or DT_DEBUG')
            return None
        linkmap = self._make_absolute_ptr(linkmap)
        w.success('%#x' % linkmap)
        return linkmap

    def waitfor(self, msg):
        if False:
            i = 10
            return i + 15
        if not self._waitfor:
            self._waitfor = log.waitfor(msg)
        else:
            self.status(msg)
        return self._waitfor

    def failure(self, msg):
        if False:
            print('Hello World!')
        if not self._waitfor:
            log.failure(msg)
        else:
            self._waitfor.failure(msg)
            self._waitfor = None

    def success(self, msg):
        if False:
            while True:
                i = 10
        if not self._waitfor:
            log.success(msg)
        else:
            self._waitfor.success(msg)
            self._waitfor = None

    def status(self, msg):
        if False:
            print('Hello World!')
        if not self._waitfor:
            log.info(msg)
        else:
            self._waitfor.status(msg)

    @property
    def libc(self):
        if False:
            print('Hello World!')
        'libc(self) -> ELF\n\n        Leak the Build ID of the remote libc.so, download the file,\n        and load an ``ELF`` object with the correct base address.\n\n        Returns:\n            An ELF object, or None.\n        '
        libc = b'libc.so'
        with self.waitfor('Downloading libc'):
            dynlib = self._dynamic_load_dynelf(libc)
            self.status('Trying lookup based on Build ID')
            build_id = dynlib._lookup_build_id(libc)
            if not build_id:
                return None
            self.status('Trying lookup based on Build ID: %s' % build_id)
            path = libcdb.search_by_build_id(build_id)
            if not path:
                return None
            libc = ELF(path)
            libc.address = dynlib.libbase
            return libc

    def lookup(self, symb=None, lib=None):
        if False:
            for i in range(10):
                print('nop')
        "lookup(symb = None, lib = None) -> int\n\n        Find the address of ``symbol``, which is found in ``lib``.\n\n        Arguments:\n            symb(str): Named routine to look up\n              If omitted, the base address of the library will be returned.\n            lib(str): Substring to match for the library name.\n              If omitted, the current library is searched.\n              If set to ``'libc'``, ``'libc.so'`` is assumed.\n\n        Returns:\n            Address of the named symbol, or :const:`None`.\n        "
        result = None
        if lib == 'libc':
            lib = 'libc.so'
        if symb:
            symb = _need_bytes(symb, min_wrong=128)
        if symb and lib:
            pretty = '%r in %r' % (symb, lib)
        else:
            pretty = repr(symb or lib)
        if not pretty:
            self.failure('Must specify a library or symbol')
        self.waitfor('Resolving %s' % pretty)
        if lib is not None:
            dynlib = self._dynamic_load_dynelf(lib)
        else:
            dynlib = self
        if dynlib is None:
            log.failure('Could not find %r', lib)
            return None
        if symb and self.libcdb:
            self.status('Trying lookup based on Build ID')
            build_id = dynlib._lookup_build_id(lib=lib)
            if build_id:
                log.info('Trying lookup based on Build ID: %s', build_id)
                path = libcdb.search_by_build_id(build_id)
                if path:
                    with context.local(log_level='error'):
                        e = ELF(path)
                        e.address = dynlib.libbase
                        result = e.symbols[symb]
        if symb and (not result):
            self.status('Trying remote lookup')
            result = dynlib._lookup(symb)
        if not symb:
            result = dynlib.libbase
        if result:
            self.success('%#x' % result)
        else:
            self.failure('Could not find %s' % pretty)
        return result

    def bases(self):
        if False:
            i = 10
            return i + 15
        'Resolve base addresses of all loaded libraries.\n\n        Return a dictionary mapping library path to its base address.\n        '
        if not self._bases:
            leak = self.leak
            LinkMap = {32: elf.Elf32_Link_Map, 64: elf.Elf64_Link_Map}[self.elfclass]
            cur = self.link_map
            while leak.field(cur, LinkMap.l_prev):
                cur = leak.field(cur, LinkMap.l_prev)
            while cur:
                p_name = leak.field(cur, LinkMap.l_name)
                name = leak.s(p_name)
                addr = leak.field(cur, LinkMap.l_addr)
                cur = leak.field(cur, LinkMap.l_next)
                log.debug('Found %r @ %#x', name, addr)
                self._bases[name] = addr
        return self._bases

    def _dynamic_load_dynelf(self, libname):
        if False:
            while True:
                i = 10
        "_dynamic_load_dynelf(libname) -> DynELF\n\n        Looks up information about a loaded library via the link map.\n\n        Arguments:\n            libname(str):  Name of the library to resolve, or a substring (e.g. 'libc.so')\n\n        Returns:\n            A DynELF instance for the loaded library, or None.\n        "
        cur = self.link_map
        leak = self.leak
        LinkMap = {32: elf.Elf32_Link_Map, 64: elf.Elf64_Link_Map}[self.elfclass]
        while leak.field(cur, LinkMap.l_prev):
            cur = leak.field(cur, LinkMap.l_prev)
        libname = _need_bytes(libname, 2, 128)
        while cur:
            self.status('link_map entry %#x' % cur)
            p_name = leak.field(cur, LinkMap.l_name)
            name = leak.s(p_name)
            if libname in name:
                break
            if name:
                self.status('Skipping %s' % name)
            cur = leak.field(cur, LinkMap.l_next)
        else:
            self.failure('Could not find library with name containing %r' % libname)
            return None
        libbase = leak.field(cur, LinkMap.l_addr)
        self.status('Resolved library %r at %#x' % (libname, libbase))
        lib = DynELF(leak, libbase)
        lib._dynamic = leak.field(cur, LinkMap.l_ld)
        lib._waitfor = self._waitfor
        return lib

    def _lookup(self, symb):
        if False:
            for i in range(10):
                print('nop')
        'Performs the actual symbol lookup within one ELF file.'
        leak = self.leak
        Dyn = {32: elf.Elf32_Dyn, 64: elf.Elf64_Dyn}[self.elfclass]
        name = lambda tag: next((k for (k, v) in ENUM_D_TAG.items() if v == tag))
        self.status('.gnu.hash/.hash, .strtab and .symtab offsets')
        hshtab = self._find_dt(constants.DT_GNU_HASH)
        strtab = self._find_dt(constants.DT_STRTAB)
        symtab = self._find_dt(constants.DT_SYMTAB)
        if hshtab:
            hshtype = 'gnu'
        else:
            hshtab = self._find_dt(constants.DT_HASH)
            hshtype = 'sysv'
        if not all([strtab, symtab, hshtab]):
            self.failure('Could not find all tables')
        strtab = self._make_absolute_ptr(strtab)
        symtab = self._make_absolute_ptr(symtab)
        hshtab = self._make_absolute_ptr(hshtab)
        routine = {'sysv': self._resolve_symbol_sysv, 'gnu': self._resolve_symbol_gnu}[hshtype]
        return routine(self.libbase, symb, hshtab, strtab, symtab)

    def _resolve_symbol_sysv(self, libbase, symb, hshtab, strtab, symtab):
        if False:
            return 10
        '\n        Internal Documentation:\n            See the ELF manual for more information.  Search for the phrase\n            "A hash table of Elf32_Word objects supports symbol table access", or see:\n            https://docs.oracle.com/cd/E19504-01/802-6319/6ia12qkfo/index.html#chapter6-48031\n\n            .. code-block:: c\n\n                struct Elf_Hash {\n                    uint32_t nbucket;\n                    uint32_t nchain;\n                    uint32_t bucket[nbucket];\n                    uint32_t chain[nchain];\n                }\n\n            You can force an ELF to use this type of symbol table by compiling\n            with \'gcc -Wl,--hash-style=sysv\'\n        '
        self.status('.hash parms')
        leak = self.leak
        Sym = {32: elf.Elf32_Sym, 64: elf.Elf64_Sym}[self.elfclass]
        nbucket = leak.field(hshtab, elf.Elf_HashTable.nbucket)
        bucketaddr = hshtab + sizeof(elf.Elf_HashTable)
        chain = bucketaddr + nbucket * 4
        self.status('hashmap')
        hsh = sysv_hash(symb) % nbucket
        idx = leak.d(bucketaddr, hsh)
        while idx != constants.STN_UNDEF:
            sym = symtab + idx * sizeof(Sym)
            symtype = leak.field(sym, Sym.st_info) & 15
            if symtype == constants.STT_FUNC:
                name = leak.s(strtab + leak.field(sym, Sym.st_name))
                if name == symb:
                    addr = libbase + leak.field(sym, Sym.st_value)
                    return addr
                self.status('%r (hash collision)' % name)
            idx = leak.d(chain, idx)
        else:
            self.failure('Could not find a SYSV hash that matched %#x' % hsh)
            return None

    def _resolve_symbol_gnu(self, libbase, symb, hshtab, strtab, symtab):
        if False:
            i = 10
            return i + 15
        "\n        Internal Documentation:\n            The GNU hash structure is a bit more complex than the normal hash\n            structure.\n\n            Again, Oracle has good documentation.\n            https://blogs.oracle.com/solaris/post/gnu-hash-elf-sections\n\n            You can force an ELF to use this type of symbol table by compiling\n            with 'gcc -Wl,--hash-style=gnu'\n        "
        self.status('.gnu.hash parms')
        leak = self.leak
        Sym = {32: elf.Elf32_Sym, 64: elf.Elf64_Sym}[self.elfclass]
        nbuckets = leak.field(hshtab, elf.GNU_HASH.nbuckets)
        symndx = leak.field(hshtab, elf.GNU_HASH.symndx)
        maskwords = leak.field(hshtab, elf.GNU_HASH.maskwords)
        elfword = self.elfclass // 8
        buckets = hshtab + sizeof(elf.GNU_HASH) + elfword * maskwords
        chains = buckets + 4 * nbuckets
        self.status('hash chain index')
        hsh = gnu_hash(symb)
        bucket = hsh % nbuckets
        ndx = leak.d(buckets, bucket)
        if ndx == 0:
            self.failure('Empty chain')
            return None
        chain = chains + 4 * (ndx - symndx)
        self.status('hash chain')
        i = 0
        hsh &= ~1
        hsh2 = 0
        while not hsh2 & 1:
            hsh2 = leak.d(chain, i)
            if hsh == hsh2 & ~1:
                sym = symtab + sizeof(Sym) * (ndx + i)
                name = leak.s(strtab + leak.field(sym, Sym.st_name))
                if name == symb:
                    offset = leak.field(sym, Sym.st_value)
                    addr = offset + libbase
                    return addr
                self.status('%r (hash collision)' % name)
            i += 1
        else:
            self.failure('Could not find a GNU hash that matched %#x' % hsh)
            return None

    def _lookup_build_id(self, lib=None):
        if False:
            return 10
        libbase = self.libbase
        if not self.link_map:
            self.status('No linkmap found')
            return None
        if lib is not None:
            libbase = self.lookup(symb=None, lib=lib)
        if not libbase:
            self.status("Couldn't find libc base")
            return None
        for offset in libcdb.get_build_id_offsets():
            address = libbase + offset
            if self.leak.compare(address + 12, b'GNU\x00'):
                return enhex(b''.join(self.leak.raw(address + 16, 20)))
            else:
                self.status('Build ID not found at offset %#x' % offset)
                pass

    def _make_absolute_ptr(self, ptr_or_offset):
        if False:
            print('Hello World!')
        "For shared libraries (or PIE executables), many ELF fields may\n        contain offsets rather than actual pointers. If the ELF type is 'DYN',\n        the argument may be an offset. It will not necessarily be an offset,\n        because the run-time linker may have fixed it up to be a real pointer\n        already. In this case an educated guess is made, and the ELF base\n        address is added to the value if it is determined to be an offset.\n        "
        if_ptr = ptr_or_offset
        if_offset = ptr_or_offset + self.libbase
        if self.elftype != 'DYN':
            return if_ptr
        if 0 < ptr_or_offset < self.libbase:
            return if_offset
        else:
            return if_ptr

    def stack(self):
        if False:
            return 10
        'Finds a pointer to the stack via __environ, which is an exported\n        symbol in libc, which points to the environment block.\n        '
        symbols = ['environ', '_environ', '__environ']
        for symbol in symbols:
            environ = self.lookup(symbol, 'libc')
            if environ:
                break
        else:
            log.error('Could not find the stack')
        stack = self.leak.p(environ)
        self.success('*environ: %#x' % stack)
        return stack

    def heap(self):
        if False:
            i = 10
            return i + 15
        'Finds the beginning of the heap via __curbrk, which is an exported\n        symbol in the linker, which points to the current brk.\n        '
        curbrk = self.lookup('__curbrk', 'libc')
        brk = self.leak.p(curbrk)
        self.success('*curbrk: %#x' % brk)
        return brk

    def _find_mapped_pages(self, readonly=False, page_size=4096):
        if False:
            print('Hello World!')
        '\n        A generator of all mapped pages, as found using the Program Headers.\n\n        Yields tuples of the form: (virtual address, memory size)\n        '
        leak = self.leak
        base = self.libbase
        Ehdr = {32: elf.Elf32_Ehdr, 64: elf.Elf64_Ehdr}[self.elfclass]
        Phdr = {32: elf.Elf32_Phdr, 64: elf.Elf64_Phdr}[self.elfclass]
        phead = base + leak.field(base, Ehdr.e_phoff)
        phnum = leak.field(base, Ehdr.e_phnum)
        for i in range(phnum):
            if leak.field_compare(phead, Phdr.p_type, constants.PT_LOAD):
                if leak.field_compare(phead, Phdr.p_align, page_size) and (readonly or leak.field(phead, Phdr.p_flags) & 2 != 0):
                    vaddr = leak.field(phead, Phdr.p_vaddr)
                    memsz = leak.field(phead, Phdr.p_memsz)
                    if vaddr < base:
                        vaddr += base
                    yield (vaddr, memsz)
            phead += sizeof(Phdr)

    def dump(self, libs=False, readonly=False):
        if False:
            while True:
                i = 10
        "dump(libs = False, readonly = False)\n\n        Dumps the ELF's memory pages to allow further analysis.\n\n        Arguments:\n            libs(bool, optional): True if should dump the libraries too (False by default)\n            readonly(bool, optional): True if should dump read-only pages (False by default)\n\n        Returns:\n            a dictionary of the form: { address : bytes }\n        "
        leak = self.leak
        page_size = 4096
        pages = {}
        for (vaddr, memsz) in self._find_mapped_pages(readonly, page_size):
            offset = vaddr % page_size
            if offset != 0:
                memsz += offset
                vaddr -= offset
            memsz += (page_size - memsz % page_size) % page_size
            pages[vaddr] = leak.n(vaddr, memsz)
        if libs:
            for lib_name in self.bases():
                if len(lib_name) == 0:
                    continue
                dyn_lib = self._dynamic_load_dynelf(lib_name)
                if dyn_lib is not None:
                    pages.update(dyn_lib.dump(readonly=readonly))
        return pages