"""
High-level abstraction of Minidump file
"""
from builtins import range
import struct
from miasm.loader.strpatchwork import StrPatchwork
from miasm.loader import minidump as mp

class MemorySegment(object):
    """Stand for a segment in memory with additional information"""

    def __init__(self, offset, memory_desc, module=None, memory_info=None):
        if False:
            for i in range(10):
                print('nop')
        self.offset = offset
        self.memory_desc = memory_desc
        self.module = module
        self.memory_info = memory_info
        self.minidump = self.memory_desc.parent_head

    @property
    def address(self):
        if False:
            for i in range(10):
                print('nop')
        return self.memory_desc.StartOfMemoryRange

    @property
    def size(self):
        if False:
            i = 10
            return i + 15
        if isinstance(self.memory_desc, mp.MemoryDescriptor64):
            return self.memory_desc.DataSize
        elif isinstance(self.memory_desc, mp.MemoryDescriptor):
            return self.memory_desc.Memory.DataSize
        raise TypeError

    @property
    def name(self):
        if False:
            return 10
        if not self.module:
            return ''
        name = mp.MinidumpString.unpack(self.minidump._content, self.module.ModuleNameRva.rva, self.minidump)
        return b''.join((struct.pack('B', x) for x in name.Buffer)).decode('utf-16')

    @property
    def content(self):
        if False:
            return 10
        return self.minidump._content[self.offset:self.offset + self.size]

    @property
    def protect(self):
        if False:
            print('Hello World!')
        if self.memory_info:
            return self.memory_info.Protect
        return None

    @property
    def pretty_protect(self):
        if False:
            return 10
        if self.protect is None:
            return 'UNKNOWN'
        return mp.memProtect[self.protect]

class Minidump(object):
    """Stand for a Minidump file

    Here is a few limitation:
     - only < 4GB Minidump are supported (LocationDescriptor handling)
     - only Stream relative to memory mapping are implemented

    Official description is available on MSDN:
    https://msdn.microsoft.com/en-us/library/ms680378(VS.85).aspx
    """
    _sex = 0
    _wsize = 32

    def __init__(self, minidump_str):
        if False:
            i = 10
            return i + 15
        self._content = StrPatchwork(minidump_str)
        self.modulelist = None
        self.memory64list = None
        self.memorylist = None
        self.memoryinfolist = None
        self.systeminfo = None
        self.streams = []
        self.threads = None
        self.parse_content()
        self.memory = {}
        self.build_memory()

    def parse_content(self):
        if False:
            return 10
        'Build structures corresponding to current content'
        offset = 0
        self.minidumpHDR = mp.MinidumpHDR.unpack(self._content, offset, self)
        assert self.minidumpHDR.Magic == 1347241037
        base_offset = self.minidumpHDR.StreamDirectoryRva.rva
        empty_stream = mp.StreamDirectory(StreamType=0, Location=mp.LocationDescriptor(DataSize=0, Rva=mp.Rva(rva=0)))
        streamdir_size = len(empty_stream)
        for i in range(self.minidumpHDR.NumberOfStreams):
            stream_offset = base_offset + i * streamdir_size
            stream = mp.StreamDirectory.unpack(self._content, stream_offset, self)
            self.streams.append(stream)
            datasize = stream.Location.DataSize
            offset = stream.Location.Rva.rva
            if stream.StreamType == mp.streamType.ModuleListStream:
                self.modulelist = mp.ModuleList.unpack(self._content, offset, self)
            elif stream.StreamType == mp.streamType.MemoryListStream:
                self.memorylist = mp.MemoryList.unpack(self._content, offset, self)
            elif stream.StreamType == mp.streamType.Memory64ListStream:
                self.memory64list = mp.Memory64List.unpack(self._content, offset, self)
            elif stream.StreamType == mp.streamType.MemoryInfoListStream:
                self.memoryinfolist = mp.MemoryInfoList.unpack(self._content, offset, self)
            elif stream.StreamType == mp.streamType.SystemInfoStream:
                self.systeminfo = mp.SystemInfo.unpack(self._content, offset, self)
        for stream in self.streams:
            datasize = stream.Location.DataSize
            offset = stream.Location.Rva.rva
            if self.systeminfo is not None and stream.StreamType == mp.streamType.ThreadListStream:
                self.threads = mp.ThreadList.unpack(self._content, offset, self)

    def build_memory(self):
        if False:
            while True:
                i = 10
        'Build an easier to use memory view based on ModuleList and\n        Memory64List streams'
        addr2module = dict(((module.BaseOfImage, module) for module in (self.modulelist.Modules if self.modulelist else [])))
        addr2meminfo = dict(((memory.BaseAddress, memory) for memory in (self.memoryinfolist.MemoryInfos if self.memoryinfolist else [])))
        mode64 = self.minidumpHDR.Flags & mp.minidumpType.MiniDumpWithFullMemory
        if mode64:
            offset = self.memory64list.BaseRva
            memranges = self.memory64list.MemoryRanges
        else:
            memranges = self.memorylist.MemoryRanges
        for memory in memranges:
            if not mode64:
                offset = memory.Memory.Rva.rva
            base_address = memory.StartOfMemoryRange
            module = addr2module.get(base_address, None)
            meminfo = addr2meminfo.get(base_address, None)
            self.memory[base_address] = MemorySegment(offset, memory, module, meminfo)
            if mode64:
                offset += memory.DataSize
        if mode64:
            assert all((addr in self.memory for addr in addr2module))

    def get(self, virt_start, virt_stop):
        if False:
            while True:
                i = 10
        'Return the content at the (virtual addresses)\n        [virt_start:virt_stop]'
        for addr in self.memory:
            if virt_start <= addr <= virt_stop:
                break
        else:
            return b''
        memory = self.memory[addr]
        shift = addr - virt_start
        last = virt_stop - addr
        if last > memory.size:
            raise RuntimeError('Multi-page not implemented')
        return self._content[memory.offset + shift:memory.offset + last]