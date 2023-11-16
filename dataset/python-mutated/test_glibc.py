from __future__ import annotations
import os
import shutil
import tempfile
import gdb
import pytest
import pwndbg.gdblib.info
import pwndbg.glibc
import tests
HEAP_MALLOC_CHUNK = tests.binaries.get('heap_malloc_chunk.out')

@pytest.mark.parametrize('have_debugging_information', [True, False], ids=['does-not-have-(*)', 'have-(*)'])
def test_parsing_info_sharedlibrary_to_find_libc_filename(start_binary, have_debugging_information):
    if False:
        for i in range(10):
            print('nop')
    if not have_debugging_information:
        gdb.execute('set debug-file-directory')
    start_binary(HEAP_MALLOC_CHUNK)
    gdb.execute('break break_here')
    gdb.execute('continue')
    if not have_debugging_information:
        assert '(*)' in pwndbg.gdblib.info.sharedlibrary()
    libc_path = pwndbg.glibc.get_libc_filename_from_info_sharedlibrary()
    assert libc_path is not None
    test_libc_names = ['libc-2.36.so', 'libc6_2.36-0ubuntu4_amd64.so', 'libc.so']
    with tempfile.TemporaryDirectory() as tmp_dir:
        for test_libc_name in test_libc_names:
            test_libc_path = os.path.join(tmp_dir, test_libc_name)
            shutil.copy(libc_path, test_libc_path)
            gdb.execute(f'set environment LD_PRELOAD={test_libc_path}')
            start_binary(HEAP_MALLOC_CHUNK)
            gdb.execute('break break_here')
            gdb.execute('continue')
            if not have_debugging_information:
                assert '(*)' in pwndbg.gdblib.info.sharedlibrary()
            assert pwndbg.glibc.get_libc_filename_from_info_sharedlibrary() == test_libc_path
        test_libc_path = os.path.join(tmp_dir, 'a_weird_name_that_does_not_look_like_a_1ibc.so')
        shutil.copy(libc_path, test_libc_path)
        gdb.execute(f'set environment LD_PRELOAD={test_libc_path}')
        start_binary(HEAP_MALLOC_CHUNK)
        gdb.execute('break break_here')
        gdb.execute('continue')
        assert pwndbg.glibc.get_libc_filename_from_info_sharedlibrary() is None