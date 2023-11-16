import pytest
import lief
import pathlib
from utils import get_sample

def test_static_pie():
    if False:
        return 10
    static_pie_path = get_sample('ELF/elf64_static_pie.bin')
    static_pie = lief.parse(static_pie_path)
    assert static_pie.is_pie

def test_static():
    if False:
        print('Hello World!')
    static_path = get_sample('ELF/batch-x86-64/test.gcc.fullstatic.nothread.bin')
    static = lief.parse(static_path)
    assert not static.is_pie

def test_pie():
    if False:
        i = 10
        return i + 15
    pie_path = get_sample('ELF/batch-x86-64/test.go.pie.bin')
    pie = lief.parse(pie_path)
    assert pie.is_pie

def test_non_pie():
    if False:
        print('Hello World!')
    not_pie_path = get_sample('ELF/ELF32_x86_library_libshellx.so')
    not_pie = lief.parse(not_pie_path)
    assert not not_pie.is_pie

def test_non_pie_bin():
    if False:
        for i in range(10):
            print('nop')
    path = get_sample('ELF/ELF64_x86-64_binary_ls.bin')
    target = lief.parse(path)
    assert not target.is_pie