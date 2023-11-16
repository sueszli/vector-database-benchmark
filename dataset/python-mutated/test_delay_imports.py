import lief
from utils import get_sample

def test_simple():
    if False:
        return 10
    '\n    Referential test on a simple case\n    This test aims at checking we cover correctly a regular binary\n    '
    binary: lief.PE.Binary = lief.parse(get_sample('PE/test.delay.exe'))
    assert binary.has_delay_imports
    assert len(binary.delay_imports) == 2
    assert binary.get_delay_import('USER32.dll') is not None
    assert binary.has_delay_import('USER32.dll')
    assert len(binary.imported_functions) == 87
    assert len(binary.libraries) == 3
    shlwapi = binary.delay_imports[0]
    assert shlwapi.name == 'SHLWAPI.dll'
    assert shlwapi.attribute == 1
    assert shlwapi.handle == 171464
    assert shlwapi.iat == 154928
    assert shlwapi.names_table == 147272
    assert shlwapi.biat == 147328
    assert shlwapi.uiat == 0
    assert shlwapi.timestamp == 0
    assert len(shlwapi.entries) == 1
    strstra = shlwapi.entries[0]
    assert strstra.name == 'StrStrA'
    assert strstra.value == 154928
    assert strstra.iat_value == 12894362189
    assert strstra.data == 147304
    assert strstra.hint == 333
    user32 = binary.delay_imports[1]
    assert user32.name == 'USER32.dll'
    assert user32.attribute == 1
    assert user32.handle == 171472
    assert user32.iat == 154944
    assert user32.names_table == 147288
    assert user32.biat == 147344
    assert user32.uiat == 0
    assert user32.timestamp == 0
    assert len(user32.entries) == 1
    assert user32.copy() == user32
    messageboxa = user32.entries[0]
    assert messageboxa.copy() == messageboxa
    assert messageboxa.copy().copy() != user32
    assert messageboxa.ordinal == 16242
    assert messageboxa.name == 'MessageBoxA'
    assert messageboxa.value == 154944
    assert messageboxa.iat_value == 12894362189
    assert messageboxa.data == 147314
    assert messageboxa.hint == 645
    print(messageboxa)

def test_cmd():
    if False:
        i = 10
        return i + 15
    '\n    Test on cmd.exe\n    '
    binary: lief.PE.Binary = lief.parse(get_sample('PE/PE64_x86-64_binary_cmd.exe'))
    assert binary.has_delay_imports
    assert len(binary.delay_imports) == 4
    assert len(binary.imported_functions) == 247
    assert len(binary.libraries) == 8
    shell32 = binary.get_delay_import('SHELL32.dll')
    assert shell32.name == 'SHELL32.dll'
    assert shell32.attribute == 1
    assert shell32.handle == 189160
    assert shell32.iat == 188536
    assert shell32.names_table == 173472
    assert shell32.biat == 0
    assert shell32.uiat == 0
    assert shell32.timestamp == 0
    assert len(shell32.entries) == 2
    SHChangeNotify = shell32.entries[0]
    assert SHChangeNotify.name == 'SHChangeNotify'
    assert SHChangeNotify.value == 188536
    assert SHChangeNotify.iat_value == 12894362189
    assert SHChangeNotify.data == 173806
    assert SHChangeNotify.hint == 0
    ShellExecuteExW = shell32.entries[1]
    assert ShellExecuteExW.name == 'ShellExecuteExW'
    assert ShellExecuteExW.value == 188544
    assert ShellExecuteExW.iat_value == 281470681743364
    assert ShellExecuteExW.data == 173824
    assert ShellExecuteExW.hint == 0