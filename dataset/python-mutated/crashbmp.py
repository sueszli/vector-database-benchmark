""" An AS for processing Windows Bitmap crash dumps """
import struct
import volatility.obj as obj
import volatility.addrspace as addrspace
import volatility.plugins.addrspaces.crash as crash

class BitmapDmpVTypes(obj.ProfileModification):
    conditions = {'os': lambda x: x == 'windows', 'memory_model': lambda x: x == '64bit'}

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update({'_FULL_DUMP64': [56, {'Signature': [0, ['array', 4, ['unsigned char']]], 'ValidDump': [4, ['array', 4, ['unsigned char']]], 'DumpOptions': [8, ['unsigned long long']], 'HeaderSize': [32, ['unsigned long long']], 'BitmapSize': [40, ['unsigned long long']], 'Pages': [48, ['unsigned long long']], 'Buffer': [56, ['array', lambda x: (x.BitmapSize + 7) / 8, ['unsigned char']]], 'Buffer2': [56, ['array', lambda x: (x.BitmapSize + 31) / 32, ['unsigned long']]]}]})

class WindowsCrashDumpSpace64BitMap(crash.WindowsCrashDumpSpace32):
    """ This AS supports Windows BitMap Crash Dump format """
    order = 29
    dumpsig = 'PAGEDU64'
    headertype = '_DMP_HEADER64'
    headerpages = 19
    bitmaphdroffset = 8192

    def __init__(self, base, config, **kwargs):
        if False:
            i = 10
            return i + 15
        self.as_assert(base, 'No base Address Space')
        addrspace.AbstractRunBasedMemory.__init__(self, base, config, **kwargs)
        self.as_assert(base.read(0, 8) == self.dumpsig, 'Header signature invalid')
        self.as_assert(self.profile.has_type(self.headertype), self.headertype + ' not available in profile')
        self.header = obj.Object(self.headertype, 0, base)
        self.as_assert(self.header.DumpType == 5, 'Unsupported dump format')
        self.bitmaphdr = obj.Object('_FULL_DUMP64', self.bitmaphdroffset, base)
        fdmp_buff = base.read(self.bitmaphdroffset, self.bitmaphdr.HeaderSize - self.bitmaphdroffset)
        bufferas = addrspace.BufferAddressSpace(self._config, data=fdmp_buff)
        self.bitmaphdr2 = obj.Object('_FULL_DUMP64', vm=bufferas, offset=0)
        firstbit = None
        firstoffset = 0
        lastbit = None
        lastbitseen = 0
        offset = self.bitmaphdr2.HeaderSize
        for i in range(0, (self.bitmaphdr2.BitmapSize + 31) / 32):
            if self.bitmaphdr.Buffer2[i] == 0:
                if firstbit != None:
                    lastbit = (i - 1) * 32 + 31
                    self.runs.append((firstbit * 4096, firstoffset, (lastbit - firstbit + 1) * 4096))
                    firstbit = None
            elif self.bitmaphdr.Buffer2[i] == 4294967295:
                if firstbit == None:
                    firstoffset = offset
                    firstbit = i * 32
                offset = offset + 32 * 4096
            else:
                wordoffset = i * 32
                for j in range(0, 32):
                    BitAddr = wordoffset + j
                    ByteOffset = BitAddr >> 3
                    ByteAddress = self.bitmaphdr2.Buffer[ByteOffset]
                    ShiftCount = BitAddr & 7
                    if ByteAddress >> ShiftCount & 1:
                        if firstbit == None:
                            firstoffset = offset
                            firstbit = BitAddr
                        offset = offset + 4096
                    elif firstbit != None:
                        lastbit = BitAddr - 1
                        self.runs.append((firstbit * 4096, firstoffset, (lastbit - firstbit + 1) * 4096))
                        firstbit = None
            lastbitseen = i * 32 + 31
        if firstbit != None:
            self.runs.append((firstbit * 4096, firstoffset, (lastbitseen - firstbit + 1) * 4096))
        self.dtb = self.header.DirectoryTableBase.v()