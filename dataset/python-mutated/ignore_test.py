import io
from xml.etree import ElementTree as ET
from pytest import raises
from hscommon.testutil import eq_
from core.ignore import IgnoreList

def test_empty():
    if False:
        for i in range(10):
            print('nop')
    il = IgnoreList()
    eq_(0, len(il))
    assert not il.are_ignored('foo', 'bar')

def test_simple():
    if False:
        print('Hello World!')
    il = IgnoreList()
    il.ignore('foo', 'bar')
    assert il.are_ignored('foo', 'bar')
    assert il.are_ignored('bar', 'foo')
    assert not il.are_ignored('foo', 'bleh')
    assert not il.are_ignored('bleh', 'bar')
    eq_(1, len(il))

def test_multiple():
    if False:
        print('Hello World!')
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('foo', 'bleh')
    il.ignore('bleh', 'bar')
    il.ignore('aybabtu', 'bleh')
    assert il.are_ignored('foo', 'bar')
    assert il.are_ignored('bar', 'foo')
    assert il.are_ignored('foo', 'bleh')
    assert il.are_ignored('bleh', 'bar')
    assert not il.are_ignored('aybabtu', 'bar')
    eq_(4, len(il))

def test_clear():
    if False:
        print('Hello World!')
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.clear()
    assert not il.are_ignored('foo', 'bar')
    assert not il.are_ignored('bar', 'foo')
    eq_(0, len(il))

def test_add_same_twice():
    if False:
        for i in range(10):
            print('nop')
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('bar', 'foo')
    eq_(1, len(il))

def test_save_to_xml():
    if False:
        while True:
            i = 10
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('foo', 'bleh')
    il.ignore('bleh', 'bar')
    f = io.BytesIO()
    il.save_to_xml(f)
    f.seek(0)
    doc = ET.parse(f)
    root = doc.getroot()
    eq_(root.tag, 'ignore_list')
    eq_(len(root), 2)
    eq_(len([c for c in root if c.tag == 'file']), 2)
    (f1, f2) = root[:]
    subchildren = [c for c in f1 if c.tag == 'file'] + [c for c in f2 if c.tag == 'file']
    eq_(len(subchildren), 3)

def test_save_then_load():
    if False:
        for i in range(10):
            print('nop')
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('foo', 'bleh')
    il.ignore('bleh', 'bar')
    il.ignore('é', 'bar')
    f = io.BytesIO()
    il.save_to_xml(f)
    f.seek(0)
    il = IgnoreList()
    il.load_from_xml(f)
    eq_(4, len(il))
    assert il.are_ignored('é', 'bar')

def test_load_xml_with_empty_file_tags():
    if False:
        print('Hello World!')
    f = io.BytesIO()
    f.write(b'<?xml version="1.0" encoding="utf-8"?><ignore_list><file><file/></file></ignore_list>')
    f.seek(0)
    il = IgnoreList()
    il.load_from_xml(f)
    eq_(0, len(il))

def test_are_ignore_works_when_a_child_is_a_key_somewhere_else():
    if False:
        i = 10
        return i + 15
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('bar', 'baz')
    assert il.are_ignored('bar', 'foo')

def test_no_dupes_when_a_child_is_a_key_somewhere_else():
    if False:
        i = 10
        return i + 15
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('bar', 'baz')
    il.ignore('bar', 'foo')
    eq_(2, len(il))

def test_iterate():
    if False:
        return 10
    il = IgnoreList()
    expected = [('foo', 'bar'), ('bar', 'baz'), ('foo', 'baz')]
    for i in expected:
        il.ignore(i[0], i[1])
    for i in il:
        expected.remove(i)
    assert not expected

def test_filter():
    if False:
        print('Hello World!')
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('bar', 'baz')
    il.ignore('foo', 'baz')
    il.filter(lambda f, s: f == 'bar')
    eq_(1, len(il))
    assert not il.are_ignored('foo', 'bar')
    assert il.are_ignored('bar', 'baz')

def test_save_with_non_ascii_items():
    if False:
        return 10
    il = IgnoreList()
    il.ignore('¬', '¿')
    f = io.BytesIO()
    try:
        il.save_to_xml(f)
    except Exception as e:
        raise AssertionError(str(e))

def test_len():
    if False:
        print('Hello World!')
    il = IgnoreList()
    eq_(0, len(il))
    il.ignore('foo', 'bar')
    eq_(1, len(il))

def test_nonzero():
    if False:
        return 10
    il = IgnoreList()
    assert not il
    il.ignore('foo', 'bar')
    assert il

def test_remove():
    if False:
        i = 10
        return i + 15
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('foo', 'baz')
    il.remove('bar', 'foo')
    eq_(len(il), 1)
    assert not il.are_ignored('foo', 'bar')

def test_remove_non_existant():
    if False:
        for i in range(10):
            print('nop')
    il = IgnoreList()
    il.ignore('foo', 'bar')
    il.ignore('foo', 'baz')
    with raises(ValueError):
        il.remove('foo', 'bleh')