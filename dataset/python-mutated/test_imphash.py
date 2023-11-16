import lief
import random
from utils import get_sample

def test_without_imports():
    if False:
        while True:
            i = 10
    "\n    By convention if a binary hasn't import, imphash is '0'\n    "
    binary = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    assert int(lief.PE.get_imphash(binary), 16) == 0

def test_casse():
    if False:
        print('Hello World!')
    "\n    Test that casse doesn't change the hash\n    "
    binary_lhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    binary_rhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    kernel32_lhs = binary_lhs.add_library('KERNEL32.dll')
    kernel32_lhs.add_entry('CreateMutexA')
    kernel32_rhs = binary_rhs.add_library('kernel32.dll')
    kernel32_rhs.add_entry('CrEatEMutExa')
    assert lief.PE.get_imphash(binary_lhs) == lief.PE.get_imphash(binary_rhs)

def test_order():
    if False:
        print('Hello World!')
    "\n    Test that import order doesn't change the hash\n    "
    binary_lhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    binary_rhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    fonctions = ['GetStringTypeW', 'LCMapStringW', 'GetCommandLineA', 'TerminateProcess']
    kernel32_lhs = binary_lhs.add_library('kernel32.dll')
    random.shuffle(fonctions)
    list(map(kernel32_lhs.add_entry, fonctions))
    print(kernel32_lhs)
    kernel32_rhs = binary_rhs.add_library('kernel32.dll')
    random.shuffle(fonctions)
    list(map(kernel32_rhs.add_entry, fonctions))
    print(kernel32_rhs)
    assert lief.PE.get_imphash(binary_lhs) == lief.PE.get_imphash(binary_rhs)

def test_ordinal():
    if False:
        print('Hello World!')
    '\n    Test import by ordinal\n    '
    binary_lhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    binary_rhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    fonctions = ['GetStringTypeW', 'LCMapStringW', 'GetCommandLineA', 'TerminateProcess', 'Beep', 'CheckRemoteDebuggerPresent']
    kernel32_lhs = binary_lhs.add_library('kernel32.dll')
    list(map(kernel32_lhs.add_entry, fonctions))
    kernel32_rhs = binary_rhs.add_library('kernel32.dll')
    for f in fonctions:
        if f == 'Beep':
            imp = lief.PE.ImportEntry(2147483677)
            kernel32_rhs.add_entry(imp)
        else:
            kernel32_rhs.add_entry(f)
    assert lief.PE.get_imphash(binary_lhs) == lief.PE.get_imphash(binary_rhs)

def test_order_2():
    if False:
        i = 10
        return i + 15
    "\n    Test that import order doesn't change the hash (More complex)\n    "
    binary_lhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    binary_rhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    libraries = {'KERNEL32.dll': ['GetStringTypeW', 'LCMapStringW', 'GetCommandLineA', 'TerminateProcess', 'Beep', 'CheckRemoteDebuggerPresent'], 'ntdll.dll': ['NtWriteVirtualMemory', 'NtYieldExecution', 'PfxFindPrefix', 'PfxInitialize', 'PfxInsertPrefix', 'PfxRemovePrefix', 'PropertyLengthAsVariant', 'RtlAbortRXact']}
    keys = list(libraries.keys())
    random.shuffle(keys)
    for k in keys:
        lib_lhs = binary_lhs.add_library(k)
        v = libraries[k]
        random.shuffle(v)
        for e in v:
            lib_lhs.add_entry(e)
    keys = list(libraries.keys())
    random.shuffle(keys)
    for k in keys:
        lib_rhs = binary_rhs.add_library(k)
        v = libraries[k]
        random.shuffle(v)
        for e in v:
            lib_rhs.add_entry(e)
    assert lief.PE.get_imphash(binary_lhs) == lief.PE.get_imphash(binary_rhs)

def test_different():
    if False:
        while True:
            i = 10
    '\n    Check that different imports have different hashes\n    '
    binary_lhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    binary_rhs = lief.PE.Binary(lief.PE.PE_TYPE.PE32)
    libraries = {'KERNEL32.dll': ['GetStringTypeW', 'LCMapStringW', 'GetCommandLineA', 'TerminateProcess', 'Beep', 'CheckRemoteDebuggerPresent'], 'ntdll.dll': ['NtWriteVirtualMemory', 'NtYieldExecution', 'PfxFindPrefix', 'PfxInitialize', 'PfxInsertPrefix', 'PfxRemovePrefix', 'PropertyLengthAsVariant', 'RtlAbortRXact']}
    keys = list(libraries.keys())
    random.shuffle(keys)
    for k in keys:
        lib_lhs = binary_lhs.add_library(k)
        v = libraries[k]
        random.shuffle(v)
        for e in v:
            lib_lhs.add_entry(e)
    keys = list(libraries.keys())
    random.shuffle(keys)
    for k in keys:
        lib_rhs = binary_rhs.add_library(k)
        v = libraries[k]
        random.shuffle(v)
        for e in filter(lambda e: len(e) % 2 == 0, v):
            lib_rhs.add_entry(e)
    assert lief.PE.get_imphash(binary_lhs) != lief.PE.get_imphash(binary_rhs)

def test_pefile():
    if False:
        print('Hello World!')
    '\n    Check that we can reproduce pefile output\n    '
    s1 = lief.parse(get_sample('PE/PE64_x86-64_binary_notepad.exe'))
    assert lief.PE.get_imphash(s1, lief.PE.IMPHASH_MODE.PEFILE) == '38934ee4aaaaa8dab7c73508bc6715ca'
    s2 = lief.parse(get_sample('PE/PE32_x86_binary_PGO-PGI.exe'))
    assert lief.PE.get_imphash(s2, lief.PE.IMPHASH_MODE.PEFILE) == '4d7ac2eefa8a35d9c445d71412e8e71c'