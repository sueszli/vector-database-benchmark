""" An AS for processing crash dumps """
import struct
import volatility.obj as obj
import volatility.addrspace as addrspace
page_shift = 12

class WindowsCrashDumpSpace32(addrspace.AbstractRunBasedMemory):
    """ This AS supports windows Crash Dump format """
    order = 30
    dumpsig = 'PAGEDUMP'
    headertype = '_DMP_HEADER'
    headerpages = 1
    _long_struct = struct.Struct('=I')

    def __init__(self, base, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.as_assert(base, 'No base Address Space')
        addrspace.AbstractRunBasedMemory.__init__(self, base, config, **kwargs)
        self.as_assert(base.read(0, 8) == self.dumpsig, 'Header signature invalid')
        self.as_assert(self.profile.has_type(self.headertype), self.headertype + ' not available in profile')
        self.header = obj.Object(self.headertype, 0, base)
        self.as_assert(self.header.DumpType == 1, 'Unsupported dump format')
        offset = self.headerpages
        for x in self.header.PhysicalMemoryBlockBuffer.Run:
            self.runs.append((x.BasePage.v() * 4096, offset * 4096, x.PageCount.v() * 4096))
            offset += x.PageCount.v()
        self.dtb = self.header.DirectoryTableBase.v()

    def get_header(self):
        if False:
            i = 10
            return i + 15
        return self.header

    def get_base(self):
        if False:
            for i in range(10):
                print('nop')
        return self.base

    def read_long(self, addr):
        if False:
            for i in range(10):
                print('nop')
        _baseaddr = self.translate(addr)
        string = self.read(addr, 4)
        if not string:
            return obj.NoneObject('Could not read data at ' + str(addr))
        (longval,) = self._long_struct.unpack(string)
        return longval

    def get_available_addresses(self):
        if False:
            return 10
        ' This returns the ranges  of valid addresses '
        for run in self.runs:
            yield (run[0], run[2])

    def close(self):
        if False:
            i = 10
            return i + 15
        self.base.close()

class WindowsCrashDumpSpace64(WindowsCrashDumpSpace32):
    """ This AS supports windows Crash Dump format """
    order = 30
    dumpsig = 'PAGEDU64'
    headertype = '_DMP_HEADER64'
    headerpages = 2