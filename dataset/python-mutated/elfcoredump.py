import volatility.obj as obj
import volatility.addrspace as addrspace
NT_VBOXCORE = 2816
NT_VBOXCPU = 2817
DBGFCORE_MAGIC = 3222978782
DBGFCORE_FMT_VERSION = 65536
NT_QEMUCORE = 1

class DBGFCOREDESCRIPTOR(obj.CType):
    """A class for VBox core dump descriptors"""

    @property
    def Major(self):
        if False:
            i = 10
            return i + 15
        return self.u32VBoxVersion >> 24 & 255

    @property
    def Minor(self):
        if False:
            return 10
        return self.u32VBoxVersion >> 16 & 255

    @property
    def Build(self):
        if False:
            while True:
                i = 10
        return self.u32VBoxVersion & 65535

class VirtualBoxModification(obj.ProfileModification):

    def modification(self, profile):
        if False:
            print('Hello World!')
        profile.vtypes.update({'DBGFCOREDESCRIPTOR': [24, {'u32Magic': [0, ['unsigned int']], 'u32FmtVersion': [4, ['unsigned int']], 'cbSelf': [8, ['unsigned int']], 'u32VBoxVersion': [12, ['unsigned int']], 'u32VBoxRevision': [16, ['unsigned int']], 'cCpus': [20, ['unsigned int']]}]})
        profile.object_classes.update({'DBGFCOREDESCRIPTOR': DBGFCOREDESCRIPTOR})

class VirtualBoxCoreDumpElf64(addrspace.AbstractRunBasedMemory):
    """ This AS supports VirtualBox ELF64 coredump format """
    order = 30

    def __init__(self, base, config, **kwargs):
        if False:
            i = 10
            return i + 15
        self.as_assert(base, 'No base Address Space')
        addrspace.AbstractRunBasedMemory.__init__(self, base, config, **kwargs)
        self.as_assert(base.read(0, 6) in ['\x7fELF\x02\x01', '\x7fELF\x01\x01'], 'ELF Header signature invalid')
        elf = obj.Object('elf_hdr', offset=0, vm=base)
        self.as_assert(str(elf.e_type) == 'ET_CORE', 'ELF type is not a Core file')
        self.runs = []
        self.header = None
        for phdr in elf.program_headers():
            if str(phdr.p_type) == 'PT_NOTE':
                note = obj.Object('elf_note', offset=phdr.p_offset, vm=base, parent=phdr)
                self.check_note(note)
                continue
            if str(phdr.p_type) != 'PT_LOAD' or phdr.p_filesz == 0 or phdr.p_filesz != phdr.p_memsz:
                continue
            self.runs.append((int(phdr.p_paddr), int(phdr.p_offset), int(phdr.p_memsz)))
        self.validate()

    def check_note(self, note):
        if False:
            return 10
        'Check the Note type'
        if note.namesz == 'VBCORE' and note.n_type == NT_VBOXCORE:
            self.header = note.cast_descsz('DBGFCOREDESCRIPTOR')

    def validate(self):
        if False:
            return 10
        self.as_assert(self.header, 'ELF error: did not find any PT_NOTE segment with VBCORE')
        self.as_assert(self.header.u32Magic == DBGFCORE_MAGIC, 'Could not find VBox core magic signature')
        self.as_assert(self.header.u32FmtVersion & 4294967280 == DBGFCORE_FMT_VERSION, 'Unknown VBox core format version')
        self.as_assert(self.runs, 'ELF error: did not find any LOAD segment with main RAM')

class QemuCoreDumpElf(VirtualBoxCoreDumpElf64):
    """ This AS supports Qemu ELF32 and ELF64 coredump format """

    def check_note(self, note):
        if False:
            for i in range(10):
                print('nop')
        'Check the Note type'
        if str(note.namesz) == 'CORE' and note.n_type == NT_QEMUCORE:
            self.header = 1

    def validate(self):
        if False:
            i = 10
            return i + 15
        self.as_assert(self.header, 'ELF error: did not find any PT_NOTE segment with CORE')
        self.as_assert(self.runs, 'ELF error: did not find any LOAD segment with main RAM')