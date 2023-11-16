import lief
from utils import get_sample
TARGET = lief.parse(get_sample('ELF/ELF32_x86_binary_all.bin'))

def test_header():
    if False:
        i = 10
        return i + 15
    assert TARGET.interpreter == '/lib/ld-linux.so.2'
    assert TARGET.entrypoint == 1908

def test_sections():
    if False:
        i = 10
        return i + 15
    assert len(TARGET.sections) == 32
    assert TARGET.has_section('.tdata')
    text_section = TARGET.get_section('.text')
    assert text_section.type == lief.ELF.SECTION_TYPES.PROGBITS
    assert text_section.offset == 1744
    assert text_section.virtual_address == 1744
    assert text_section.size == 625
    assert text_section.alignment == 16
    assert lief.ELF.SECTION_FLAGS.ALLOC in text_section
    assert lief.ELF.SECTION_FLAGS.EXECINSTR in text_section

def test_segments():
    if False:
        i = 10
        return i + 15
    segments = TARGET.segments
    assert len(segments) == 10
    LOAD_0 = segments[2]
    LOAD_1 = segments[3]
    assert LOAD_0.type == lief.ELF.SEGMENT_TYPES.LOAD
    assert LOAD_0.file_offset == 0
    assert LOAD_0.virtual_address == 0
    assert LOAD_0.physical_size == 2868
    assert LOAD_0.virtual_size == 2868
    assert int(LOAD_0.flags) == lief.ELF.SEGMENT_FLAGS.R | lief.ELF.SEGMENT_FLAGS.X
    assert LOAD_1.type == lief.ELF.SEGMENT_TYPES.LOAD
    assert LOAD_1.file_offset == 3800
    assert LOAD_1.virtual_address == 7896
    assert LOAD_1.physical_address == 7896
    assert LOAD_1.physical_size == 328
    assert LOAD_1.virtual_size == 332
    assert int(LOAD_1.flags) == lief.ELF.SEGMENT_FLAGS.R | lief.ELF.SEGMENT_FLAGS.W

def test_dynamic():
    if False:
        while True:
            i = 10
    entries = TARGET.dynamic_entries
    assert len(entries) == 28
    assert entries[0].name == 'libc.so.6'
    assert entries[3].array == [2208, 1782]
    assert TARGET[lief.ELF.DYNAMIC_TAGS.FLAGS_1].value == 134217728

def test_relocations():
    if False:
        i = 10
        return i + 15
    dynamic_relocations = TARGET.dynamic_relocations
    pltgot_relocations = TARGET.pltgot_relocations
    assert len(dynamic_relocations) == 10
    assert len(pltgot_relocations) == 3
    assert dynamic_relocations[0].address == 7900
    assert dynamic_relocations[8].symbol.name == '__gmon_start__'
    assert dynamic_relocations[9].address == 8188
    assert pltgot_relocations[1].address == 8208
    assert pltgot_relocations[1].symbol.name == 'puts'
    assert pltgot_relocations[1].info == 4

def test_symbols():
    if False:
        while True:
            i = 10
    dynamic_symbols = TARGET.dynamic_symbols
    static_symbols = TARGET.static_symbols
    assert len(dynamic_symbols) == 27
    assert len(static_symbols) == 78
    first = TARGET.get_dynamic_symbol('first')
    assert first.value == 2217
    assert first.symbol_version.value == 32770
    assert first.symbol_version.symbol_version_auxiliary.name == 'LIBSIMPLE_1.0'
    dtor = TARGET.get_static_symbol('__cxa_finalize@@GLIBC_2.1.3')
    assert dtor.value == 0
    symbol_version_definition = TARGET.symbols_version_definition
    symbols_version_requirement = TARGET.symbols_version_requirement
    symbols_version = TARGET.symbols_version
    assert len(symbol_version_definition) == 2
    assert len(symbols_version_requirement) == 1
    assert len(symbols_version) == 27
    assert symbol_version_definition[0].hash == 6539790
    assert symbol_version_definition[0].version == 1
    assert symbol_version_definition[0].flags == 1
    assert symbol_version_definition[0].auxiliary_symbols[0].name == 'all-32.bin'
    assert symbol_version_definition[1].auxiliary_symbols[0].name == 'LIBSIMPLE_1.0'
    assert symbols_version_requirement[0].name == 'libc.so.6'
    assert symbols_version_requirement[0].version == 1
    assert symbols_version[0].value == 0

def test_notes():
    if False:
        for i in range(10):
            print('nop')
    notes = TARGET.notes
    assert len(notes) == 2
    assert notes[0].abi == lief.ELF.NoteAbi.ABI.LINUX
    assert list(notes[0].description) == [0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    assert notes[0].name == 'GNU'
    assert notes[0].type == lief.ELF.Note.TYPE.GNU_ABI_TAG
    assert notes[0].version == [3, 2, 0]

def test_symbols_sections():
    if False:
        while True:
            i = 10
    '\n    Related to this issue: https://github.com/lief-project/LIEF/issues/841\n    '
    elf = lief.parse(get_sample('ELF/ELF64_x86-64_binary_all.bin'))
    main = elf.get_static_symbol('main')
    assert main.section is not None
    assert main.section.name == '.text'
    assert elf.get_static_symbol('__gmon_start__').section is None
    assert elf.get_static_symbol('_fini').section.name == '.fini'