import pytest
from PyInstaller.building.datastruct import TOC
ELEMS1 = (('encodings', '/usr/lib/python2.7/encodings/__init__.py', 'PYMODULE'), ('_random', '/usr/lib/python2.7/lib-dynload/_random.so', 'EXTENSION'), ('libreadline.so.6', '/lib64/libreadline.so.6', 'BINARY'))
ELEMS2 = (('li-la-lu', '/home/myself/li-la-su', 'SOMETHING'), ('schubidu', '/home/otherguy/schibidu', 'PKG'))
ELEMS3 = (('PIL.Image.py', '/usr/lib/python2.7/encodings/__init__.py', 'PYMODULE'),)
pytestmark = pytest.mark.filterwarnings('ignore:TOC class is deprecated.')

def test_init_empty():
    if False:
        while True:
            i = 10
    toc = TOC()
    assert len(toc) == 0

def test_init():
    if False:
        for i in range(10):
            print('nop')
    toc = TOC(ELEMS1)
    assert len(toc) == 3
    assert toc == list(ELEMS1)

def test_append():
    if False:
        return 10
    toc = TOC(ELEMS1)
    toc.append(('li-la-lu', '/home/myself/li-la-su', 'SOMETHING'))
    expected = list(ELEMS1)
    expected.append(('li-la-lu', '/home/myself/li-la-su', 'SOMETHING'))
    assert toc == expected

def test_append_existing():
    if False:
        for i in range(10):
            print('nop')
    toc = TOC(ELEMS1)
    toc.append(ELEMS1[-1])
    expected = list(ELEMS1)
    assert toc == expected

def test_append_keep_filename():
    if False:
        print('Hello World!')
    toc = TOC()
    entry = ('EnCodIngs', '/usr/lib/python2.7/encodings.py', 'BINARY')
    toc.append(entry)
    assert toc[0][0] == entry[0]

def test_insert():
    if False:
        return 10
    toc = TOC(ELEMS1)
    toc.insert(1, ('li-la-lu', '/home/myself/li-la-su', 'SOMETHING'))
    expected = list(ELEMS1)
    expected.insert(1, ('li-la-lu', '/home/myself/li-la-su', 'SOMETHING'))
    assert toc == expected

def test_insert_existing():
    if False:
        return 10
    toc = TOC(ELEMS1)
    toc.insert(0, ELEMS1[-1])
    toc.insert(1, ELEMS1[-1])
    expected = list(ELEMS1)
    assert toc == expected

def test_insert_keep_filename():
    if False:
        for i in range(10):
            print('nop')
    toc = TOC()
    entry = ('EnCodIngs', '/usr/lib/python2.7/encodings.py', 'BINARY')
    toc.insert(1, entry)
    assert toc[0][0] == entry[0]

def test_extend():
    if False:
        i = 10
        return i + 15
    toc = TOC(ELEMS1)
    toc.extend(ELEMS2)
    expected = list(ELEMS1)
    expected.extend(ELEMS2)
    assert toc == expected

def test_extend_existing():
    if False:
        return 10
    toc = TOC(ELEMS1)
    toc.extend(ELEMS1)
    expected = list(ELEMS1)
    assert toc == expected

def test_add_list():
    if False:
        return 10
    toc = TOC(ELEMS1)
    other = list(ELEMS2)
    result = toc + other
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1) + list(ELEMS2)
    assert result == expected

def test_add_tuple():
    if False:
        while True:
            i = 10
    toc = TOC(ELEMS1)
    other = ELEMS2
    result = toc + other
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1) + list(ELEMS2)
    assert result == expected

def test_add_toc():
    if False:
        return 10
    toc = TOC(ELEMS1)
    other = TOC(ELEMS2)
    result = toc + other
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1) + list(ELEMS2)
    assert result == expected

def test_radd_list():
    if False:
        while True:
            i = 10
    toc = TOC(ELEMS1)
    other = list(ELEMS2)
    result = other + toc
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS2) + list(ELEMS1)
    assert result == expected

def test_radd_tuple():
    if False:
        print('Hello World!')
    toc = TOC(ELEMS1)
    other = ELEMS2
    result = other + toc
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS2) + list(ELEMS1)
    assert result == expected

def test_radd_toc():
    if False:
        i = 10
        return i + 15
    toc = TOC(ELEMS1)
    other = TOC(ELEMS2)
    result = other + toc
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS2) + list(ELEMS1)
    assert result == expected

def test_sub_list():
    if False:
        while True:
            i = 10
    toc = TOC(ELEMS1) + ELEMS2
    other = list(ELEMS2)
    result = toc - other
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1)
    assert result == expected

def test_sub_tuple():
    if False:
        print('Hello World!')
    toc = TOC(ELEMS1) + ELEMS2
    other = ELEMS2
    result = toc - other
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1)
    assert result == expected

def test_sub_toc():
    if False:
        i = 10
        return i + 15
    toc = TOC(ELEMS1) + ELEMS2
    other = TOC(ELEMS2)
    result = toc - other
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1)
    assert result == expected

def test_sub_non_existing():
    if False:
        print('Hello World!')
    toc = TOC(ELEMS1)
    other = ELEMS3
    result = toc - other
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1)
    assert result == expected

def test_rsub_list():
    if False:
        while True:
            i = 10
    toc = TOC(ELEMS1)
    other = list(ELEMS1) + list(ELEMS2)
    result = other - toc
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS2)
    assert result == expected

def test_rsub_tuple():
    if False:
        for i in range(10):
            print('nop')
    toc = TOC(ELEMS1)
    other = tuple(list(ELEMS1) + list(ELEMS2))
    result = other - toc
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS2)
    assert result == expected

def test_rsub_toc():
    if False:
        print('Hello World!')
    toc = TOC(ELEMS1)
    other = TOC(ELEMS1) + ELEMS2
    result = other - toc
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS2)
    assert result == expected

def test_rsub_non_existing():
    if False:
        return 10
    toc = TOC(ELEMS3)
    other = ELEMS1
    result = other - toc
    assert result is not toc
    assert result is not other
    assert isinstance(result, TOC)
    expected = list(ELEMS1)
    assert result == expected

def test_sub_after_setitem():
    if False:
        print('Hello World!')
    toc = TOC(ELEMS1)
    toc[1] = ('lib-dynload/_random', '/usr/lib/python2.7/lib-dynload/_random.so', 'EXTENSION')
    toc -= []
    assert len(toc) == 3

def test_sub_after_sub():
    if False:
        print('Hello World!')
    toc = TOC(ELEMS1)
    toc -= [ELEMS1[0]]
    toc -= [ELEMS1[1]]
    expected = list(ELEMS1[2:])
    assert toc == expected

def test_setitem_1():
    if False:
        return 10
    toc = TOC()
    toc[:] = ELEMS1
    for e in ELEMS1:
        assert e in toc
        assert e[0] in toc.filenames

def test_setitem_2():
    if False:
        i = 10
        return i + 15
    toc = TOC(ELEMS1)
    toc[1] = ELEMS3[0]
    assert ELEMS1[0] in toc
    assert ELEMS1[0][0] in toc.filenames
    assert ELEMS3[0] in toc
    assert ELEMS3[0][0] in toc.filenames
    assert ELEMS1[2] in toc
    assert ELEMS1[2][0] in toc.filenames
    for e in toc:
        assert e[0] in toc.filenames

@pytest.mark.win32
def test_append_other_case_mixed():
    if False:
        for i in range(10):
            print('nop')
    toc = TOC(ELEMS1)
    elem = ('EnCodIngs', '/usr/lib/python2.7/encodings.py', 'BINARY')
    toc.append(elem)
    expected = list(ELEMS1)
    assert toc == expected

@pytest.mark.win32
def test_append_other_case_pymodule():
    if False:
        for i in range(10):
            print('nop')
    toc = TOC(ELEMS1)
    elem = ('EnCodIngs', '/usr/lib/python2.7/encodings.py', 'PYMODULE')
    toc.append(elem)
    expected = list(ELEMS1)
    expected.append(elem)
    assert toc == expected

@pytest.mark.win32
def test_append_other_case_binary():
    if False:
        while True:
            i = 10
    toc = TOC(ELEMS1)
    toc.append(('LiBrEADlInE.so.6', '/lib64/libreadline.so.6', 'BINARY'))
    expected = list(ELEMS1)
    assert toc == expected

@pytest.mark.win32
def test_insert_other_case_mixed():
    if False:
        print('Hello World!')
    toc = TOC(ELEMS1)
    elem = ('EnCodIngs', '/usr/lib/python2.7/encodings.py', 'BINARY')
    toc.insert(1, elem)
    expected = list(ELEMS1)
    assert toc == expected

@pytest.mark.win32
def test_insert_other_case_pymodule():
    if False:
        i = 10
        return i + 15
    toc = TOC(ELEMS1)
    elem = ('EnCodIngs', '/usr/lib/python2.7/encodings.py', 'PYMODULE')
    toc.insert(1, elem)
    expected = list(ELEMS1)
    expected.insert(1, elem)
    assert toc == expected

@pytest.mark.win32
def test_insert_other_case_binary():
    if False:
        for i in range(10):
            print('nop')
    toc = TOC(ELEMS1)
    toc.insert(1, ('LiBrEADlInE.so.6', '/lib64/libreadline.so.6', 'BINARY'))
    expected = list(ELEMS1)
    assert toc == expected