from __future__ import annotations
import os
import pwndbg.commands.cymbol
import pwndbg.gdblib.dt
import tests
REFERENCE_BINARY = tests.binaries.get('reference-binary.out')

def create_symbol_file(symbol, source):
    if False:
        i = 10
        return i + 15
    custom_structure_example_path = os.path.join(pwndbg.commands.cymbol.pwndbg_cachedir, symbol) + '.c'
    with open(custom_structure_example_path, 'w') as f:
        f.write(source)
    return custom_structure_example_path

def check_symbol_existance(symbol_type):
    if False:
        return 10
    try:
        pwndbg.gdblib.dt.dt(symbol_type)
    except Exception as exception:
        assert isinstance(exception, AttributeError)

def test_cymbol(start_binary):
    if False:
        while True:
            i = 10
    start_binary(REFERENCE_BINARY)
    custom_structure_example = '\n        typedef struct example_struct {\n            int a;\n            char b[16];\n            char* c;\n            void* d;\n        } example_t;\n    '
    custom_structure_example_path = create_symbol_file('example', custom_structure_example)
    assert pwndbg.commands.cymbol.OnlyWhenStructFileExists(lambda x, y: True)('dummy') is None
    assert pwndbg.commands.cymbol.OnlyWhenStructFileExists(lambda x, y: True)('example') is True
    assert pwndbg.commands.cymbol.generate_debug_symbols(custom_structure_example_path) is not None
    pwndbg.commands.cymbol.load_custom_structure('example')
    assert pwndbg.commands.cymbol.loaded_symbols.get('example') is not None
    assert 'example_t\n    +0x0000 a                    : int\n    +0x0004 b                    : char [16]\n    +0x0018 c                    : char *\n    +0x0020 d                    : void *' == pwndbg.gdblib.dt.dt('example_t').strip()
    pwndbg.commands.cymbol.unload_loaded_symbol('example')
    assert pwndbg.commands.cymbol.loaded_symbols.get('example') is None
    check_symbol_existance('example_t')
    pwndbg.commands.cymbol.load_custom_structure('example')
    pwndbg.commands.cymbol.remove_custom_structure('example')
    check_symbol_existance('example_t')