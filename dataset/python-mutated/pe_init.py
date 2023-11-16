from __future__ import print_function
from builtins import range
import array
from functools import reduce
import logging
import struct
from future.builtins import int as int_types
from future.utils import PY3
from miasm.loader import pe
from miasm.loader.strpatchwork import StrPatchwork
log = logging.getLogger('peparse')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(levelname)-8s]: %(message)s'))
log.addHandler(console_handler)
log.setLevel(logging.WARN)

class ContentManager(object):

    def __get__(self, owner, _):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(owner, '_content'):
            return owner._content

    def __set__(self, owner, new_content):
        if False:
            i = 10
            return i + 15
        owner.resize(len(owner._content), len(new_content))
        owner._content = new_content

    def __delete__(self, owner):
        if False:
            i = 10
            return i + 15
        self.__set__(owner, None)

class ContectRva(object):

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        self.parent = parent

    def get(self, rva_start, rva_stop=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get data in RVA view starting at @rva_start, stopping at @rva_stop\n        @rva_start: rva start address\n        @rva_stop: rva stop address\n        '
        if rva_start is None:
            raise IOError('Out of range')
        if rva_start < 0:
            raise IOError('Out of range')
        if rva_stop is not None:
            if rva_stop > len(self.parent.img_rva):
                rva_stop = len(self.parent.img_rva)
            if rva_start > len(self.parent.img_rva):
                raise ValueError('Out of range')
            return self.parent.img_rva[rva_start:rva_stop]
        if rva_start > len(self.parent.img_rva):
            raise ValueError('Out of range')
        return self.parent.img_rva[rva_start]

    def set(self, rva, data):
        if False:
            while True:
                i = 10
        '\n        Set @data in RVA view starting at @start\n        @rva: rva start address\n        @data: data to set\n        '
        if not isinstance(rva, int_types):
            raise ValueError('addr must be int/long')
        if rva < 0:
            raise ValueError('Out of range')
        if rva + len(data) > len(self.parent.img_rva):
            raise ValueError('Out of range')
        self.parent.img_rva[rva] = data

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        if isinstance(item, slice):
            assert item.step is None
            return self.get(item.start, item.stop)
        return self.get(item)

    def __setitem__(self, item, data):
        if False:
            i = 10
            return i + 15
        if isinstance(item, slice):
            rva = item.start
        else:
            rva = item
        self.set(rva, data)

class ContentVirtual(object):

    def __init__(self, parent):
        if False:
            return 10
        self.parent = parent

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        raise DeprecationWarning('Replace code by virt.get(start, [stop])')

    def __setitem__(self, item, data):
        if False:
            return 10
        raise DeprecationWarning('Replace code by virt.set(start, data)')

    def __call__(self, ad_start, ad_stop=None, ad_step=None):
        if False:
            while True:
                i = 10
        raise DeprecationWarning('Replace code by virt.get(start, stop)')

    def get(self, virt_start, virt_stop=None):
        if False:
            return 10
        '\n        Get data in VIRTUAL view starting at @virt_start, stopping at @virt_stop\n        @virt_start: virt start address\n        @virt_stop: virt stop address\n        '
        rva_start = self.parent.virt2rva(virt_start)
        if virt_stop != None:
            rva_stop = self.parent.virt2rva(virt_stop)
        else:
            rva_stop = None
        return self.parent.rva.get(rva_start, rva_stop)

    def set(self, addr, data):
        if False:
            print('Hello World!')
        '\n        Set @data in VIRTUAL view starting at @start\n        @addr: virtual start address\n        @data: data to set\n        '
        if not isinstance(addr, int_types):
            raise ValueError('addr must be int/long')
        self.parent.rva.set(self.parent.virt2rva(addr), data)

    def max_addr(self):
        if False:
            for i in range(10):
                print('nop')
        section = self.parent.SHList[-1]
        length = section.addr + section.size + self.parent.NThdr.ImageBase
        return int(length)

    def find(self, pattern, start=0, end=None):
        if False:
            for i in range(10):
                print('nop')
        if start != 0:
            start = self.parent.virt2rva(start)
        if end != None:
            end = self.parent.virt2rva(end)
        ret = self.parent.img_rva.find(pattern, start, end)
        if ret == -1:
            return -1
        return self.parent.rva2virt(ret)

    def rfind(self, pattern, start=0, end=None):
        if False:
            return 10
        if start != 0:
            start = self.parent.virt2rva(start)
        if end != None:
            end = self.parent.virt2rva(end)
        ret = self.parent.img_rva.rfind(pattern, start, end)
        if ret == -1:
            return -1
        return self.parent.rva2virt(ret)

    def is_addr_in(self, addr):
        if False:
            while True:
                i = 10
        return self.parent.is_in_virt_address(addr)

def compute_crc(raw, olds):
    if False:
        for i in range(10):
            print('nop')
    out = 0
    data = raw[:]
    if len(raw) % 2:
        end = struct.unpack('B', data[-1:])[0]
        data = data[:-1]
    if (len(raw) & ~1) % 4:
        out += struct.unpack('H', data[:2])[0]
        data = data[2:]
    data = array.array('I', data)
    out = reduce(lambda x, y: x + y, data, out)
    out -= olds
    while out > 4294967295:
        out = (out >> 32) + (out & 4294967295)
    while out > 65535:
        out = (out & 65535) + (out >> 16 & 65535)
    if len(raw) % 2:
        out += end
    out += len(data)
    return out

class PE(object):
    content = ContentManager()

    def __init__(self, pestr=None, loadfrommem=False, parse_resources=True, parse_delay=True, parse_reloc=True, wsize=32, **kwargs):
        if False:
            i = 10
            return i + 15
        self._rva = ContectRva(self)
        self._virt = ContentVirtual(self)
        self.img_rva = StrPatchwork()
        if pestr is None:
            self._content = StrPatchwork()
            self._sex = 0
            self._wsize = wsize
            self.Doshdr = pe.Doshdr(self)
            self.NTsig = pe.NTsig(self)
            self.Coffhdr = pe.Coffhdr(self)
            if self._wsize == 32:
                Opthdr = pe.Opthdr32
            else:
                Opthdr = pe.Opthdr64
            self.Opthdr = Opthdr(self)
            self.NThdr = pe.NThdr(self)
            self.NThdr.optentries = [pe.Optehdr(self) for _ in range(16)]
            self.NThdr.CheckSum = 0
            self.SHList = pe.SHList(self)
            self.SHList.shlist = []
            self.NThdr.sizeofheaders = 4096
            self.DirImport = pe.DirImport(self)
            self.DirExport = pe.DirExport(self)
            self.DirDelay = pe.DirDelay(self)
            self.DirReloc = pe.DirReloc(self)
            self.DirRes = pe.DirRes(self)
            self.DirTls = pe.DirTls(self)
            self.Doshdr.magic = 23117
            self.Doshdr.lfanew = 224
            self.NTsig.signature = 17744
            if wsize == 32:
                self.Opthdr.magic = 267
            elif wsize == 64:
                self.Opthdr.magic = 523
            else:
                raise ValueError('unknown pe size %r' % wsize)
            self.Opthdr.majorlinkerversion = 7
            self.Opthdr.minorlinkerversion = 0
            self.NThdr.filealignment = 4096
            self.NThdr.sectionalignment = 4096
            self.NThdr.majoroperatingsystemversion = 5
            self.NThdr.minoroperatingsystemversion = 1
            self.NThdr.MajorImageVersion = 5
            self.NThdr.MinorImageVersion = 1
            self.NThdr.majorsubsystemversion = 4
            self.NThdr.minorsubsystemversion = 0
            self.NThdr.subsystem = 3
            if wsize == 32:
                self.NThdr.dllcharacteristics = 32768
            else:
                self.NThdr.dllcharacteristics = 32768
            self.NThdr.sizeofstackreserve = 2097152
            self.NThdr.sizeofstackcommit = 4096
            self.NThdr.sizeofheapreserve = 1048576
            self.NThdr.sizeofheapcommit = 4096
            self.NThdr.ImageBase = 4194304
            self.NThdr.sizeofheaders = 4096
            self.NThdr.numberofrvaandsizes = 16
            self.NTsig.signature = 17744
            if wsize == 32:
                self.Coffhdr.machine = 332
            elif wsize == 64:
                self.Coffhdr.machine = 34404
            else:
                raise ValueError('unknown pe size %r' % wsize)
            if wsize == 32:
                self.Coffhdr.characteristics = 271
                self.Coffhdr.sizeofoptionalheader = 224
            else:
                self.Coffhdr.characteristics = 34
                self.Coffhdr.sizeofoptionalheader = 240
        else:
            self._content = StrPatchwork(pestr)
            self.loadfrommem = loadfrommem
            self.parse_content(parse_resources=parse_resources, parse_delay=parse_delay, parse_reloc=parse_reloc)

    def isPE(self):
        if False:
            i = 10
            return i + 15
        if self.NTsig is None:
            return False
        return self.NTsig.signature == 17744

    def parse_content(self, parse_resources=True, parse_delay=True, parse_reloc=True):
        if False:
            while True:
                i = 10
        off = 0
        self._sex = 0
        self._wsize = 32
        self.Doshdr = pe.Doshdr.unpack(self.content, off, self)
        off = self.Doshdr.lfanew
        if off > len(self.content):
            log.warn('ntsig after eof!')
            self.NTsig = None
            return
        self.NTsig = pe.NTsig.unpack(self.content, off, self)
        self.DirImport = None
        self.DirExport = None
        self.DirDelay = None
        self.DirReloc = None
        self.DirRes = None
        if self.NTsig.signature != 17744:
            log.warn('not a valid pe!')
            return
        off += len(self.NTsig)
        (self.Coffhdr, length) = pe.Coffhdr.unpack_l(self.content, off, self)
        off += length
        self._wsize = ord(self.content[off + 1]) * 32
        if self._wsize == 32:
            Opthdr = pe.Opthdr32
        else:
            Opthdr = pe.Opthdr64
        if len(self.content) < 512:
            self.content += (512 - len(self.content)) * b'\x00'
        (self.Opthdr, length) = Opthdr.unpack_l(self.content, off, self)
        self.NThdr = pe.NThdr.unpack(self.content, off + length, self)
        self.img_rva[0] = self.content[:self.NThdr.sizeofheaders]
        off += self.Coffhdr.sizeofoptionalheader
        self.SHList = pe.SHList.unpack(self.content, off, self)
        filealignment = self.NThdr.filealignment
        sectionalignment = self.NThdr.sectionalignment
        for section in self.SHList.shlist:
            virt_size = (section.size // sectionalignment + 1) * sectionalignment
            if self.loadfrommem:
                section.offset = section.addr
            if self.NThdr.sectionalignment > 4096:
                raw_off = 512 * (section.offset // 512)
            else:
                raw_off = section.offset
            if raw_off != section.offset:
                log.warn('unaligned raw section (%x %x)!', raw_off, section.offset)
            section.data = StrPatchwork()
            if section.rawsize == 0:
                rounded_size = 0
            else:
                if section.rawsize % filealignment:
                    rs = (section.rawsize // filealignment + 1) * filealignment
                else:
                    rs = section.rawsize
                rounded_size = rs
            if rounded_size > virt_size:
                rounded_size = min(rounded_size, section.size)
            data = self.content[raw_off:raw_off + rounded_size]
            section.data = data
            length = len(data)
            data += b'\x00' * ((length + 4095 & 4294963200) - length)
            self.img_rva[section.addr] = data
        self.img_rva = self.img_rva
        try:
            self.DirImport = pe.DirImport.unpack(self.img_rva, self.NThdr.optentries[pe.DIRECTORY_ENTRY_IMPORT].rva, self)
        except pe.InvalidOffset:
            log.warning('cannot parse DirImport, skipping')
            self.DirImport = pe.DirImport(self)
        try:
            self.DirExport = pe.DirExport.unpack(self.img_rva, self.NThdr.optentries[pe.DIRECTORY_ENTRY_EXPORT].rva, self)
        except pe.InvalidOffset:
            log.warning('cannot parse DirExport, skipping')
            self.DirExport = pe.DirExport(self)
        if len(self.NThdr.optentries) > pe.DIRECTORY_ENTRY_DELAY_IMPORT:
            self.DirDelay = pe.DirDelay(self)
            if parse_delay:
                try:
                    self.DirDelay = pe.DirDelay.unpack(self.img_rva, self.NThdr.optentries[pe.DIRECTORY_ENTRY_DELAY_IMPORT].rva, self)
                except pe.InvalidOffset:
                    log.warning('cannot parse DirDelay, skipping')
        if len(self.NThdr.optentries) > pe.DIRECTORY_ENTRY_BASERELOC:
            self.DirReloc = pe.DirReloc(self)
            if parse_reloc:
                try:
                    self.DirReloc = pe.DirReloc.unpack(self.img_rva, self.NThdr.optentries[pe.DIRECTORY_ENTRY_BASERELOC].rva, self)
                except pe.InvalidOffset:
                    log.warning('cannot parse DirReloc, skipping')
        if len(self.NThdr.optentries) > pe.DIRECTORY_ENTRY_RESOURCE:
            self.DirRes = pe.DirRes(self)
            if parse_resources:
                self.DirRes = pe.DirRes(self)
                try:
                    self.DirRes = pe.DirRes.unpack(self.img_rva, self.NThdr.optentries[pe.DIRECTORY_ENTRY_RESOURCE].rva, self)
                except pe.InvalidOffset:
                    log.warning('cannot parse DirRes, skipping')
        if len(self.NThdr.optentries) > pe.DIRECTORY_ENTRY_TLS:
            self.DirTls = pe.DirTls(self)
            try:
                self.DirTls = pe.DirTls.unpack(self.img_rva, self.NThdr.optentries[pe.DIRECTORY_ENTRY_TLS].rva, self)
            except pe.InvalidOffset:
                log.warning('cannot parse DirTls, skipping')

    def resize(self, old, new):
        if False:
            print('Hello World!')
        pass

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return self.content[item]

    def __setitem__(self, item, data):
        if False:
            return 10
        self.content.__setitem__(item, data)
        return

    def getsectionbyrva(self, rva):
        if False:
            i = 10
            return i + 15
        if self.SHList is None:
            return None
        for section in self.SHList.shlist:
            '\n            TODO CHECK:\n            some binaries have import rva outside section, but addresses\n            seems to be rounded\n            '
            mask = self.NThdr.sectionalignment - 1
            if section.addr <= rva < section.addr + section.size + mask & ~mask:
                return section
        return None

    def getsectionbyvad(self, vad):
        if False:
            i = 10
            return i + 15
        return self.getsectionbyrva(self.virt2rva(vad))

    def getsectionbyoff(self, off):
        if False:
            print('Hello World!')
        if self.SHList is None:
            return None
        for section in self.SHList.shlist:
            if section.offset <= off < section.offset + section.rawsize:
                return section
        return None

    def getsectionbyname(self, name):
        if False:
            return 10
        if self.SHList is None:
            return None
        for section in self.SHList:
            if section.name.strip(b'\x00').decode() == name:
                return section
        return None

    def is_rva_ok(self, rva):
        if False:
            for i in range(10):
                print('nop')
        return self.getsectionbyrva(rva) is not None

    def rva2off(self, rva):
        if False:
            while True:
                i = 10
        if rva < self.NThdr.sizeofheaders:
            return rva
        section = self.getsectionbyrva(rva)
        if section is None:
            raise pe.InvalidOffset('cannot get offset for 0x%X' % rva)
        soff = section.offset // self.NThdr.filealignment * self.NThdr.filealignment
        return rva - section.addr + soff

    def off2rva(self, off):
        if False:
            print('Hello World!')
        section = self.getsectionbyoff(off)
        if section is None:
            return
        return off - section.offset + section.addr

    def virt2rva(self, addr):
        if False:
            return 10
        '\n        Return rva of virtual address @addr; None if addr is below ImageBase\n        '
        if addr is None:
            return None
        rva = addr - self.NThdr.ImageBase
        if rva < 0:
            return None
        return rva

    def rva2virt(self, rva):
        if False:
            for i in range(10):
                print('nop')
        if rva is None:
            return
        return rva + self.NThdr.ImageBase

    def virt2off(self, addr):
        if False:
            print('Hello World!')
        '\n        Return offset of virtual address @addr\n        '
        rva = self.virt2rva(addr)
        if rva is None:
            return None
        return self.rva2off(rva)

    def off2virt(self, off):
        if False:
            return 10
        return self.rva2virt(self.off2rva(off))

    def is_in_virt_address(self, addr):
        if False:
            while True:
                i = 10
        if addr < self.NThdr.ImageBase:
            return False
        addr = self.virt2rva(addr)
        for section in self.SHList.shlist:
            if section.addr <= addr < section.addr + section.size:
                return True
        return False

    def get_drva(self):
        if False:
            return 10
        print('Deprecated: Use PE.rva instead of PE.drva')
        return self._rva

    def get_rva(self):
        if False:
            while True:
                i = 10
        return self._rva
    drva = property(get_drva)
    rva = property(get_rva)

    def get_virt(self):
        if False:
            while True:
                i = 10
        return self._virt
    virt = property(get_virt)

    def build_content(self):
        if False:
            return 10
        content = StrPatchwork()
        content[0] = bytes(self.Doshdr)
        for section in self.SHList.shlist:
            content[section.offset:section.offset + section.rawsize] = bytes(section.data)
        section_last = self.SHList.shlist[-1]
        size = section_last.addr + section_last.size + (self.NThdr.sectionalignment - 1)
        size &= ~(self.NThdr.sectionalignment - 1)
        self.NThdr.sizeofimage = size
        off = self.Doshdr.lfanew
        content[off] = bytes(self.NTsig)
        off += len(self.NTsig)
        content[off] = bytes(self.Coffhdr)
        off += len(self.Coffhdr)
        off_shlist = off + self.Coffhdr.sizeofoptionalheader
        content[off] = bytes(self.Opthdr)
        off += len(self.Opthdr)
        content[off] = bytes(self.NThdr)
        off += len(self.NThdr)
        off = off_shlist
        content[off] = bytes(self.SHList)
        for section in self.SHList:
            if off + len(bytes(self.SHList)) > section.offset:
                log.warn('section offset overlap pe hdr 0x%x 0x%x' % (off + len(bytes(self.SHList)), section.offset))
        self.DirImport.build_content(content)
        self.DirExport.build_content(content)
        self.DirDelay.build_content(content)
        self.DirReloc.build_content(content)
        self.DirRes.build_content(content)
        self.DirTls.build_content(content)
        if (self.Doshdr.lfanew + len(self.NTsig) + len(self.Coffhdr)) % 4:
            log.warn('non aligned coffhdr, bad crc calculation')
        crcs = compute_crc(bytes(content), self.NThdr.CheckSum)
        content[self.Doshdr.lfanew + len(self.NTsig) + len(self.Coffhdr) + 64] = struct.pack('I', crcs)
        return bytes(content)

    def __bytes__(self):
        if False:
            while True:
                i = 10
        return self.build_content()

    def __str__(self):
        if False:
            print('Hello World!')
        if PY3:
            return repr(self)
        return self.__bytes__()

    def export_funcs(self):
        if False:
            return 10
        if self.DirExport is None:
            print('no export dir found')
            return (None, None)
        all_func = {}
        for (i, export) in enumerate(self.DirExport.f_names):
            all_func[export.name.name] = self.rva2virt(self.DirExport.f_address[self.DirExport.f_nameordinals[i].ordinal].rva)
            all_func[self.DirExport.f_nameordinals[i].ordinal + self.DirExport.expdesc.base] = self.rva2virt(self.DirExport.f_address[self.DirExport.f_nameordinals[i].ordinal].rva)
        return all_func

    def reloc_to(self, imgbase):
        if False:
            return 10
        offset = imgbase - self.NThdr.ImageBase
        if self.DirReloc is None:
            log.warn('no relocation found!')
        for rel in self.DirReloc.reldesc:
            rva = rel.rva
            for reloc in rel.rels:
                (reloc_type, off) = reloc.rel
                if reloc_type == 0 and off == 0:
                    continue
                if reloc_type != 3:
                    raise NotImplementedError('Reloc type not supported')
                off += rva
                value = struct.unpack('I', self.rva.get(off, off + 4))[0]
                value += offset
                self.rva.set(off, struct.pack('I', value & 4294967295))
        self.NThdr.ImageBase = imgbase