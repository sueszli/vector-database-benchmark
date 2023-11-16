import lief
from utils import get_sample
TARGET = lief.parse(get_sample('ELF/ELF32_x86_library_libshellx.so'))

def test_symbols():
    if False:
        while True:
            i = 10
    symbols = [sym for (idx, sym) in enumerate(TARGET.dynamic_symbols) if idx == 0 or len(sym.name) > 0]
    assert len(symbols) == 48
    assert symbols[2].name == '__cxa_atexit'

def test_relocations():
    if False:
        while True:
            i = 10
    relocations = TARGET.relocations
    assert len(relocations) == 47
    assert relocations[10].symbol.name == 'strlen'