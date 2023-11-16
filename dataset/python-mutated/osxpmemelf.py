import volatility.obj as obj
import volatility.addrspace as addrspace

class OSXPmemELF(addrspace.AbstractRunBasedMemory):
    """ This AS supports VirtualBox ELF64 coredump format """
    order = 90

    def __init__(self, base, config, **kwargs):
        if False:
            i = 10
            return i + 15
        self.as_assert(base, 'No base Address Space')
        addrspace.AbstractRunBasedMemory.__init__(self, base, config, **kwargs)
        self.as_assert(base.read(0, 6) in ['\x7fELF\x02\x01', '\x7fELF\x01\x01'], 'ELF Header signature invalid')
        elf = obj.Object('elf_hdr', offset=0, vm=base)
        self.header = None
        for phdr in elf.program_headers():
            if str(phdr.p_type) != 'PT_LOAD' or phdr.p_filesz == 0 or phdr.p_filesz != phdr.p_memsz:
                continue
            self.runs.append((int(phdr.p_paddr), int(phdr.p_offset), int(phdr.p_memsz)))
        self.as_assert(len(self.runs) > 0, 'No PT_LOAD segments found')