import zlib
import volatility.obj as obj
import volatility.plugins.addrspaces.standard as standard

class HPAKVTypes(obj.ProfileModification):

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        profile.vtypes.update({'HPAK_HEADER': [32, {'Magic': [0, ['String', dict(length=4)]]}], 'HPAK_SECTION': [224, {'Header': [0, ['String', dict(length=32)]], 'Compressed': [140, ['unsigned int']], 'Length': [152, ['unsigned long long']], 'Offset': [168, ['unsigned long long']], 'NextSection': [176, ['unsigned long long']], 'CompressedSize': [184, ['unsigned long long']], 'Name': [212, ['String', dict(length=12)]]}]})
        profile.object_classes.update({'HPAK_HEADER': HPAK_HEADER})

class HPAK_HEADER(obj.CType):
    """A class for B.S. Hairy headers"""

    def Sections(self):
        if False:
            return 10
        section = obj.Object('HPAK_SECTION', offset=self.obj_vm.profile.get_obj_size('HPAK_HEADER'), vm=self.obj_vm)
        while section.is_valid():
            yield section
            section = section.NextSection.dereference_as('HPAK_SECTION')

class HPAKAddressSpace(standard.FileAddressSpace):
    """ This AS supports the HPAK format """
    order = 30

    def __init__(self, base, config, **kwargs):
        if False:
            return 10
        self.as_assert(base, 'No base Address Space')
        standard.FileAddressSpace.__init__(self, base, config, layered=True, **kwargs)
        self.header = obj.Object('HPAK_HEADER', offset=0, vm=base)
        self.as_assert(self.header.Magic == 'HPAK', 'Invalid magic found')
        self.physmem = None
        for section in self.header.Sections():
            if str(section.Header) == 'HPAKSECTHPAK_SECTION_PHYSDUMP':
                self.physmem = section
                break
        self.as_assert(self.physmem is not None, 'Cannot find the PHYSDUMP section')

    def read(self, addr, length):
        if False:
            print('Hello World!')
        return self.base.read(addr + self.physmem.Offset, length)

    def zread(self, addr, length):
        if False:
            while True:
                i = 10
        return self.base.zread(addr + self.physmem.Offset, length)

    def is_valid_address(self, addr):
        if False:
            print('Hello World!')
        return self.base.is_valid_address(addr + self.physmem.Offset)

    def get_header(self):
        if False:
            print('Hello World!')
        return self.header

    def convert_to_raw(self, outfd):
        if False:
            for i in range(10):
                print('nop')
        "The standard imageinfo plugin won't work on \n        hpak images so we provide this method. It wraps\n        the zlib compression if necessary"
        zlibdec = zlib.decompressobj(16 + zlib.MAX_WBITS)
        if self.physmem.Compressed == 1:
            length = self.physmem.CompressedSize
        else:
            length = self.physmem.Length
        chunk_size = 4096
        chunks = length / chunk_size

        def get_chunk(addr, size):
            if False:
                i = 10
                return i + 15
            data = self.base.read(addr, size)
            if self.physmem.Compressed == 1:
                data = zlibdec.decompress(data)
            return data
        for i in range(chunks):
            addr = self.physmem.Offset + i * chunk_size
            data = get_chunk(addr, chunk_size)
            outfd.write(data)
        leftover = length % chunk_size
        if leftover > 0:
            data = get_chunk(addr + chunk_size, leftover)
            outfd.write(data)
        return True