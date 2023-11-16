import copy
import pathlib
from PyInstaller import compat
from PyInstaller.building.datastruct import normalize_pyz_toc, normalize_toc
_BASE_TOC = [('libpython3.10.so', '/usr/lib64/libpython3.10.so', 'BINARY'), ('libsomething.so', '/usr/local/lib64/libsomething.so', 'BINARY'), ('README', '/home/user/tmp/README', 'DATA'), (str(pathlib.PurePath('data/data.csv')), '/home/user/tmp/data/data.csv', 'DATA'), ('dependency.bin', 'other_multipackage:dependency.bin', 'DEPENDENCY'), ('myextension.so', 'myextension.so', 'EXTENSION')]

def test_normalize_toc_no_duplicates():
    if False:
        for i in range(10):
            print('nop')
    toc = copy.copy(_BASE_TOC)
    expected_toc = _BASE_TOC
    normalized_toc = normalize_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_toc_duplicate_binary():
    if False:
        return 10
    toc = copy.copy(_BASE_TOC)
    toc.insert(2, ('libsomething.so', '/opt/something/lib/libsomething.so', 'BINARY'))
    expected_toc = _BASE_TOC
    normalized_toc = normalize_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_toc_duplicate_binary_case_sensitive():
    if False:
        return 10
    toc = copy.copy(_BASE_TOC)
    toc.insert(2, ('libSoMeThInG.so', '/opt/something/lib/libSoMeThInG.so', 'BINARY'))
    expected_toc = _BASE_TOC
    if compat.is_win:
        expected_toc = _BASE_TOC
    else:
        expected_toc = toc
    normalized_toc = normalize_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_toc_duplicate_data():
    if False:
        return 10
    toc = copy.copy(_BASE_TOC)
    toc.insert(3, ('README', '/home/user/tmp/README', 'DATA'))
    expected_toc = _BASE_TOC
    normalized_toc = normalize_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_toc_duplicate_data_case_sensitive():
    if False:
        while True:
            i = 10
    toc = copy.copy(_BASE_TOC)
    toc.insert(-1, ('readme', '/home/user/tmp-other/readme', 'DATA'))
    expected_toc = _BASE_TOC
    if compat.is_win:
        expected_toc = _BASE_TOC
    else:
        expected_toc = toc
    normalized_toc = normalize_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_toc_conflicting_binary_and_data1():
    if False:
        i = 10
        return i + 15
    toc = copy.copy(_BASE_TOC)
    toc.insert(2, ('libsomething.so', '/usr/local/lib64/libsomething.so', 'DATA'))
    expected_toc = _BASE_TOC
    normalized_toc = normalize_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_toc_conflicting_binary_and_data2():
    if False:
        while True:
            i = 10
    toc = copy.copy(_BASE_TOC)
    toc.insert(1, ('libsomething.so', '/usr/local/lib64/libsomething.so', 'DATA'))
    expected_toc = _BASE_TOC
    normalized_toc = normalize_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_toc_multipackage_dependency():
    if False:
        return 10
    toc = copy.copy(_BASE_TOC)
    toc.insert(0, ('dependency.bin', '/mnt/somewhere/dependency.bin', 'EXTENSION'))
    toc.insert(0, ('dependency.bin', '/mnt/somewhere/dependency.bin', 'BINARY'))
    toc.insert(0, ('dependency.bin', '/mnt/somewhere/dependency.bin', 'DATA'))
    expected_toc = _BASE_TOC
    normalized_toc = normalize_toc(toc)
    assert sorted(normalized_toc) == sorted(expected_toc)

def test_normalize_toc_with_parent_pardir_loops():
    if False:
        print('Hello World!')
    toc = [(str(pathlib.PurePath('numpy/core/../../numpy.libs/libquadmath-2d0c479f.so.0.0.0')), '/path/to/venv/lib/python3.11/site-packages/numpy/core/../../numpy.libs/libquadmath-2d0c479f.so.0.0.0', 'BINARY'), (str(pathlib.PurePath('numpy/linalg/../../numpy.libs/libquadmath-2d0c479f.so.0.0.0')), '/path/to/venv/lib/python3.11/site-packages/numpy/linalg/../../numpy.libs/libquadmath-2d0c479f.so.0.0.0', 'BINARY')]
    expected_toc = [(str(pathlib.PurePath('numpy.libs/libquadmath-2d0c479f.so.0.0.0')), '/path/to/venv/lib/python3.11/site-packages/numpy/core/../../numpy.libs/libquadmath-2d0c479f.so.0.0.0', 'BINARY')]
    normalized_toc = normalize_toc(toc)
    assert sorted(normalized_toc) == sorted(expected_toc)
_BASE_PYZ_TOC = [('copy', '/usr/lib64/python3.11/copy.py', 'PYMODULE'), ('csv', '/usr/lib64/python3.11/csv.py', 'PYMODULE'), ('dataclasses', '/usr/lib64/python3.11/dataclasses.py', 'PYMODULE'), ('datetime', '/usr/lib64/python3.11/datetime.py', 'PYMODULE'), ('decimal', '/usr/lib64/python3.11/decimal.py', 'PYMODULE'), ('mymodule1', 'mymodule1.py', 'PYMODULE'), ('mymodule2', 'mymodule2.py', 'PYMODULE')]

def test_normalize_pyz_toc_no_duplicates():
    if False:
        i = 10
        return i + 15
    toc = copy.copy(_BASE_PYZ_TOC)
    expected_toc = _BASE_PYZ_TOC
    normalized_toc = normalize_pyz_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_pyz_toc_duplicates():
    if False:
        print('Hello World!')
    toc = copy.copy(_BASE_PYZ_TOC)
    toc.insert(6, ('mymodule1', 'some-other-path/mymodule1.py', 'PYMODULE'))
    expected_toc = _BASE_PYZ_TOC
    normalized_toc = normalize_pyz_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_pyz_toc_case_sensitivity():
    if False:
        return 10
    toc = copy.copy(_BASE_PYZ_TOC)
    toc.insert(6, ('MyMoDuLe1', 'some-other-path/MyMoDuLe1.py', 'PYMODULE'))
    expected_toc = toc
    normalized_toc = normalize_pyz_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_pyz_toc_module_and_data1():
    if False:
        print('Hello World!')
    toc = copy.copy(_BASE_PYZ_TOC)
    toc.insert(5, ('mymodule1', 'data-dir/mymodule1', 'DATA'))
    expected_toc = _BASE_PYZ_TOC
    normalized_toc = normalize_pyz_toc(toc)
    assert normalized_toc == expected_toc

def test_normalize_pyz_toc_module_and_data2():
    if False:
        while True:
            i = 10
    toc = copy.copy(_BASE_PYZ_TOC)
    toc.insert(6, ('mymodule1', 'data-dir/mymodule1', 'DATA'))
    expected_toc = _BASE_PYZ_TOC
    normalized_toc = normalize_pyz_toc(toc)
    assert normalized_toc == expected_toc