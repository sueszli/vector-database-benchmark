import lief
from utils import get_sample
config = lief.ELF.ParserConfig()
config.count_mtd = lief.ELF.DYNSYM_COUNT_METHODS.HASH
TARGET = lief.ELF.parse(get_sample('ELF/ELF64_x86-64_binary_rvs.bin'), config)

def test_symbols():
    if False:
        print('Hello World!')
    symbols = TARGET.dynamic_symbols
    assert len(symbols) == 10
    assert symbols[2].name == '_IO_putc'

def test_relocations():
    if False:
        i = 10
        return i + 15
    relocations = TARGET.relocations
    assert len(relocations) == 10
    assert relocations[0].symbol.name == '__gmon_start__'