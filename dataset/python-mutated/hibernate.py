""" A Hiber file Address Space """
import volatility.addrspace as addrspace
import volatility.obj as obj
import volatility.win32.xpress as xpress
import struct
PAGE_SIZE = 4096
page_shift = 12

class Store(object):

    def __init__(self, limit=50):
        if False:
            for i in range(10):
                print('nop')
        self.limit = limit
        self.cache = {}
        self.seq = []
        self.size = 0

    def put(self, key, item):
        if False:
            i = 10
            return i + 15
        self.cache[key] = item
        self.size += len(item)
        self.seq.append(key)
        if len(self.seq) >= self.limit:
            key = self.seq.pop(0)
            self.size -= len(self.cache[key])
            del self.cache[key]

    def get(self, key):
        if False:
            print('Hello World!')
        return self.cache[key]

class WindowsHiberFileSpace32(addrspace.BaseAddressSpace):
    """ This is a hibernate address space for windows hibernation files.

    In order for us to work we need to:
    1) have a valid baseAddressSpace
    2) the first 4 bytes must be 'hibr' or 'wake'
        otherwise we bruteforce to find self.header.FirstTablePage in
        _get_first_table_page() this occurs with a zeroed PO_MEMORY_IMAGE header
    """
    order = 10

    def __init__(self, base, config, **kwargs):
        if False:
            return 10
        self.as_assert(base, 'No base Address Space')
        addrspace.BaseAddressSpace.__init__(self, base, config, **kwargs)
        self.runs = []
        self.PageDict = {}
        self.HighestPage = 0
        self.PageIndex = 0
        self.AddressList = []
        self.LookupCache = {}
        self.PageCache = Store(50)
        self.MemRangeCnt = 0
        self.entry_count = 255
        self._long_struct = struct.Struct('=I')
        self.as_assert(self.profile.has_type('PO_MEMORY_IMAGE'), 'PO_MEMORY_IMAGE is not available in profile')
        self.header = obj.Object('PO_MEMORY_IMAGE', 0, base)
        if self.header.Signature.lower() not in ['hibr', 'wake']:
            self.header = obj.NoneObject('Invalid hibernation header')
        volmag = obj.VolMagic(base)
        self.entry_count = volmag.HibrEntryCount.v()
        PROC_PAGE = volmag.HibrProcPage.v()
        pageno = self._get_first_table_page()
        self.as_assert(pageno is not None, 'No xpress signature found')
        self.as_assert(pageno <= 10, 'Bad profile for PO_MEMORY_RANGE')
        self.ProcState = obj.Object('_KPROCESSOR_STATE', PROC_PAGE * 4096, base)
        self.dtb = self.ProcState.SpecialRegisters.Cr3.v()
        self.build_page_cache()

    def _get_first_table_page(self):
        if False:
            while True:
                i = 10
        if self.header != None:
            return self.header.FirstTablePage
        for i in range(10):
            if self.base.read(i * PAGE_SIZE, 8) == '\x81\x81xpress':
                return i - 1
        return None

    def build_page_cache(self):
        if False:
            return 10
        XpressIndex = 0
        XpressHeader = obj.Object('_IMAGE_XPRESS_HEADER', (self._get_first_table_page() + 1) * 4096, self.base)
        XpressBlockSize = self.get_xpress_block_size(XpressHeader)
        MemoryArrayOffset = self._get_first_table_page() * 4096
        while MemoryArrayOffset:
            MemoryArray = obj.Object('_PO_MEMORY_RANGE_ARRAY', MemoryArrayOffset, self.base)
            EntryCount = MemoryArray.MemArrayLink.EntryCount.v()
            for i in MemoryArray.RangeTable:
                start = i.StartPage.v()
                end = i.EndPage.v()
                LocalPageCnt = end - start
                self.as_assert(LocalPageCnt > 0, 'Negative Page Count Range')
                if end > self.HighestPage:
                    self.HighestPage = end
                self.AddressList.append((start * 4096, LocalPageCnt * 4096))
                for j in range(0, LocalPageCnt):
                    if XpressIndex and XpressIndex % 16 == 0:
                        (XpressHeader, XpressBlockSize) = self.next_xpress(XpressHeader, XpressBlockSize)
                    PageNumber = start + j
                    XpressPage = XpressIndex % 16
                    if XpressHeader.obj_offset not in self.PageDict:
                        self.PageDict[XpressHeader.obj_offset] = [(PageNumber, XpressBlockSize, XpressPage)]
                    else:
                        self.PageDict[XpressHeader.obj_offset].append((PageNumber, XpressBlockSize, XpressPage))
                    self.LookupCache[PageNumber] = (XpressHeader.obj_offset, XpressBlockSize, XpressPage)
                    self.PageIndex += 1
                    XpressIndex += 1
            NextTable = MemoryArray.MemArrayLink.NextTable.v()
            if NextTable and EntryCount == self.entry_count:
                MemoryArrayOffset = NextTable * 4096
                self.MemRangeCnt += 1
                (XpressHeader, XpressBlockSize) = self.next_xpress(XpressHeader, XpressBlockSize)
                while XpressHeader.obj_offset < MemoryArrayOffset:
                    (XpressHeader, XpressBlockSize) = self.next_xpress(XpressHeader, 0)
                XpressIndex = 0
            else:
                MemoryArrayOffset = 0

    def next_xpress(self, XpressHeader, XpressBlockSize):
        if False:
            for i in range(10):
                print('nop')
        XpressHeaderOffset = XpressBlockSize + XpressHeader.obj_offset + XpressHeader.size()
        BLOCKSIZE = 1024
        original_offset = XpressHeaderOffset
        while 1:
            data = self.base.read(XpressHeaderOffset, BLOCKSIZE)
            Magic_offset = data.find('\x81\x81xpress')
            if Magic_offset >= 0:
                XpressHeaderOffset += Magic_offset
                break
            else:
                XpressHeaderOffset += len(data)
            if XpressHeaderOffset - original_offset > 10240:
                return (None, None)
        XpressHeader = obj.Object('_IMAGE_XPRESS_HEADER', XpressHeaderOffset, self.base)
        XpressBlockSize = self.get_xpress_block_size(XpressHeader)
        return (XpressHeader, XpressBlockSize)

    def get_xpress_block_size(self, xpress_header):
        if False:
            while True:
                i = 10
        u0B = xpress_header.u0B.v() << 24
        u0A = xpress_header.u0A.v() << 16
        u09 = xpress_header.u09.v() << 8
        Size = u0B + u0A + u09
        Size = Size >> 10
        Size = Size + 1
        if Size % 8 == 0:
            return Size
        return (Size & ~7) + 8

    def get_header(self):
        if False:
            while True:
                i = 10
        return self.header

    def get_base(self):
        if False:
            i = 10
            return i + 15
        return self.base

    def is_paging(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ProcState.SpecialRegisters.Cr0.v() >> 31 & 1

    def is_pse(self):
        if False:
            while True:
                i = 10
        return self.ProcState.SpecialRegisters.Cr4.v() >> 4 & 1

    def is_pae(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ProcState.SpecialRegisters.Cr4.v() >> 5 & 1

    def get_addr(self, addr):
        if False:
            for i in range(10):
                print('nop')
        page = addr >> page_shift
        if page in self.LookupCache:
            (hoffset, size, pageoffset) = self.LookupCache[page]
            return (hoffset, size, pageoffset)
        return (None, None, None)

    def get_block_offset(self, _xb, addr):
        if False:
            for i in range(10):
                print('nop')
        page = addr >> page_shift
        if page in self.LookupCache:
            (_hoffset, _size, pageoffset) = self.LookupCache[page]
            return pageoffset
        return None

    def is_valid_address(self, addr):
        if False:
            while True:
                i = 10
        (XpressHeaderOffset, _XpressBlockSize, _XpressPage) = self.get_addr(addr)
        return XpressHeaderOffset != None

    def read_xpress(self, baddr, BlockSize):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.PageCache.get(baddr)
        except KeyError:
            data_read = self.base.read(baddr, BlockSize)
            if BlockSize == 65536:
                data_uz = data_read
            else:
                data_uz = xpress.xpress_decode(data_read)
                self.PageCache.put(baddr, data_uz)
            return data_uz

    def _partial_read(self, addr, len):
        if False:
            for i in range(10):
                print('nop')
        ' A function which reads as much as possible from the current page.\n\n        May return a short read.\n        '
        page_offset = addr & 4095
        available = min(PAGE_SIZE - page_offset, len)
        (ImageXpressHeader, BlockSize, XpressPage) = self.get_addr(addr)
        if not ImageXpressHeader:
            return None
        baddr = ImageXpressHeader + 32
        data = self.read_xpress(baddr, BlockSize)
        offset = XpressPage * 4096 + page_offset
        return data[offset:offset + available]

    def read(self, addr, length, zread=False):
        if False:
            print('Hello World!')
        result = ''
        while length > 0:
            data = self._partial_read(addr, length)
            if not data:
                break
            addr += len(data)
            length -= len(data)
            result += data
        if result == '':
            if zread:
                return '\x00' * length
            result = obj.NoneObject('Unable to read data at ' + str(addr) + ' for length ' + str(length))
        return result

    def zread(self, addr, length):
        if False:
            i = 10
            return i + 15
        stuff_read = self.read(addr, length, zread=True)
        return stuff_read

    def read_long(self, addr):
        if False:
            while True:
                i = 10
        _baseaddr = self.get_addr(addr)
        string = self.read(addr, 4)
        if not string:
            return obj.NoneObject('Could not read long at ' + str(addr))
        (longval,) = self._long_struct.unpack(string)
        return longval

    def get_available_pages(self):
        if False:
            print('Hello World!')
        page_list = []
        for (_i, xb) in enumerate(self.PageDict.keys()):
            for (page, _size, _offset) in self.PageDict[xb]:
                page_list.append([page * 4096, 4096])
        return page_list

    def get_address_range(self):
        if False:
            return 10
        ' This relates to the logical address range that is indexable '
        size = self.HighestPage * 4096 + 4096
        return [0, size]

    def check_address_range(self, addr):
        if False:
            while True:
                i = 10
        memrange = self.get_address_range()
        if addr < memrange[0] or addr > memrange[1]:
            raise IOError

    def get_available_addresses(self):
        if False:
            while True:
                i = 10
        ' This returns the ranges  of valid addresses '
        for i in self.AddressList:
            yield i

    def close(self):
        if False:
            i = 10
            return i + 15
        self.base.close()