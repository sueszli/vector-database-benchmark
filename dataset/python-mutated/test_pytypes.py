import contextlib
import sys
import types
import pytest
import env
from pybind11_tests import detailed_error_messages_enabled
from pybind11_tests import pytypes as m

def test_obj_class_name():
    if False:
        print('Hello World!')
    assert m.obj_class_name(None) == 'NoneType'
    assert m.obj_class_name(list) == 'list'
    assert m.obj_class_name([]) == 'list'

def test_handle_from_move_only_type_with_operator_PyObject():
    if False:
        i = 10
        return i + 15
    assert m.handle_from_move_only_type_with_operator_PyObject_ncnst()
    assert m.handle_from_move_only_type_with_operator_PyObject_const()

def test_bool(doc):
    if False:
        while True:
            i = 10
    assert doc(m.get_bool) == 'get_bool() -> bool'

def test_int(doc):
    if False:
        print('Hello World!')
    assert doc(m.get_int) == 'get_int() -> int'

def test_iterator(doc):
    if False:
        return 10
    assert doc(m.get_iterator) == 'get_iterator() -> Iterator'

@pytest.mark.parametrize(('pytype', 'from_iter_func'), [(frozenset, m.get_frozenset_from_iterable), (list, m.get_list_from_iterable), (set, m.get_set_from_iterable), (tuple, m.get_tuple_from_iterable)])
def test_from_iterable(pytype, from_iter_func):
    if False:
        while True:
            i = 10
    my_iter = iter(range(10))
    s = from_iter_func(my_iter)
    assert type(s) == pytype
    assert s == pytype(range(10))

def test_iterable(doc):
    if False:
        while True:
            i = 10
    assert doc(m.get_iterable) == 'get_iterable() -> Iterable'

def test_float(doc):
    if False:
        i = 10
        return i + 15
    assert doc(m.get_float) == 'get_float() -> float'

def test_list(capture, doc):
    if False:
        while True:
            i = 10
    assert m.list_no_args() == []
    assert m.list_ssize_t() == []
    assert m.list_size_t() == []
    lins = [1, 2]
    m.list_insert_ssize_t(lins)
    assert lins == [1, 83, 2]
    m.list_insert_size_t(lins)
    assert lins == [1, 83, 2, 57]
    with capture:
        lst = m.get_list()
        assert lst == ['inserted-0', 'overwritten', 'inserted-2']
        lst.append('value2')
        m.print_list(lst)
    assert capture.unordered == '\n        Entry at position 0: value\n        list item 0: inserted-0\n        list item 1: overwritten\n        list item 2: inserted-2\n        list item 3: value2\n    '
    assert doc(m.get_list) == 'get_list() -> list'
    assert doc(m.print_list) == 'print_list(arg0: list) -> None'

def test_none(doc):
    if False:
        i = 10
        return i + 15
    assert doc(m.get_none) == 'get_none() -> None'
    assert doc(m.print_none) == 'print_none(arg0: None) -> None'

def test_set(capture, doc):
    if False:
        i = 10
        return i + 15
    s = m.get_set()
    assert isinstance(s, set)
    assert s == {'key1', 'key2', 'key3'}
    s.add('key4')
    with capture:
        m.print_anyset(s)
    assert capture.unordered == '\n        key: key1\n        key: key2\n        key: key3\n        key: key4\n    '
    m.set_add(s, 'key5')
    assert m.anyset_size(s) == 5
    m.set_clear(s)
    assert m.anyset_empty(s)
    assert not m.anyset_contains(set(), 42)
    assert m.anyset_contains({42}, 42)
    assert m.anyset_contains({'foo'}, 'foo')
    assert doc(m.get_set) == 'get_set() -> set'
    assert doc(m.print_anyset) == 'print_anyset(arg0: anyset) -> None'

def test_frozenset(capture, doc):
    if False:
        i = 10
        return i + 15
    s = m.get_frozenset()
    assert isinstance(s, frozenset)
    assert s == frozenset({'key1', 'key2', 'key3'})
    with capture:
        m.print_anyset(s)
    assert capture.unordered == '\n        key: key1\n        key: key2\n        key: key3\n    '
    assert m.anyset_size(s) == 3
    assert not m.anyset_empty(s)
    assert not m.anyset_contains(frozenset(), 42)
    assert m.anyset_contains(frozenset({42}), 42)
    assert m.anyset_contains(frozenset({'foo'}), 'foo')
    assert doc(m.get_frozenset) == 'get_frozenset() -> frozenset'

def test_dict(capture, doc):
    if False:
        i = 10
        return i + 15
    d = m.get_dict()
    assert d == {'key': 'value'}
    with capture:
        d['key2'] = 'value2'
        m.print_dict(d)
    assert capture.unordered == '\n        key: key, value=value\n        key: key2, value=value2\n    '
    assert not m.dict_contains({}, 42)
    assert m.dict_contains({42: None}, 42)
    assert m.dict_contains({'foo': None}, 'foo')
    assert doc(m.get_dict) == 'get_dict() -> dict'
    assert doc(m.print_dict) == 'print_dict(arg0: dict) -> None'
    assert m.dict_keyword_constructor() == {'x': 1, 'y': 2, 'z': 3}

class CustomContains:
    d = {'key': None}

    def __contains__(self, m):
        if False:
            i = 10
            return i + 15
        return m in self.d

@pytest.mark.parametrize(('arg', 'func'), [(set(), m.anyset_contains), ({}, m.dict_contains), (CustomContains(), m.obj_contains)])
@pytest.mark.xfail('env.PYPY and sys.pypy_version_info < (7, 3, 10)', strict=False)
def test_unhashable_exceptions(arg, func):
    if False:
        while True:
            i = 10

    class Unhashable:
        __hash__ = None
    with pytest.raises(TypeError) as exc_info:
        func(arg, Unhashable())
    assert 'unhashable type:' in str(exc_info.value)

def test_tuple():
    if False:
        i = 10
        return i + 15
    assert m.tuple_no_args() == ()
    assert m.tuple_ssize_t() == ()
    assert m.tuple_size_t() == ()
    assert m.get_tuple() == (42, None, 'spam')

def test_simple_namespace():
    if False:
        return 10
    ns = m.get_simple_namespace()
    assert ns.attr == 42
    assert ns.x == 'foo'
    assert ns.right == 2
    assert not hasattr(ns, 'wrong')

def test_str(doc):
    if False:
        i = 10
        return i + 15
    assert m.str_from_char_ssize_t().encode().decode() == 'red'
    assert m.str_from_char_size_t().encode().decode() == 'blue'
    assert m.str_from_string().encode().decode() == 'baz'
    assert m.str_from_bytes().encode().decode() == 'boo'
    assert doc(m.str_from_bytes) == 'str_from_bytes() -> str'

    class A:

        def __str__(self):
            if False:
                i = 10
                return i + 15
            return 'this is a str'

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return 'this is a repr'
    assert m.str_from_object(A()) == 'this is a str'
    assert m.repr_from_object(A()) == 'this is a repr'
    assert m.str_from_handle(A()) == 'this is a str'
    (s1, s2) = m.str_format()
    assert s1 == '1 + 2 = 3'
    assert s1 == s2
    malformed_utf8 = b'\x80'
    if hasattr(m, 'PYBIND11_STR_LEGACY_PERMISSIVE'):
        assert m.str_from_object(malformed_utf8) is malformed_utf8
    else:
        assert m.str_from_object(malformed_utf8) == "b'\\x80'"
    assert m.str_from_handle(malformed_utf8) == "b'\\x80'"
    assert m.str_from_string_from_str('this is a str') == 'this is a str'
    ucs_surrogates_str = '\udcc3'
    with pytest.raises(UnicodeEncodeError):
        m.str_from_string_from_str(ucs_surrogates_str)

@pytest.mark.parametrize('func', [m.str_from_bytes_input, m.str_from_cstr_input, m.str_from_std_string_input])
def test_surrogate_pairs_unicode_error(func):
    if False:
        while True:
            i = 10
    input_str = '\ud83d\ude4f'.encode('utf-8', 'surrogatepass')
    with pytest.raises(UnicodeDecodeError):
        func(input_str)

def test_bytes(doc):
    if False:
        print('Hello World!')
    assert m.bytes_from_char_ssize_t().decode() == 'green'
    assert m.bytes_from_char_size_t().decode() == 'purple'
    assert m.bytes_from_string().decode() == 'foo'
    assert m.bytes_from_str().decode() == 'bar'
    assert doc(m.bytes_from_str) == 'bytes_from_str() -> bytes'

def test_bytearray():
    if False:
        for i in range(10):
            print('nop')
    assert m.bytearray_from_char_ssize_t().decode() == '$%'
    assert m.bytearray_from_char_size_t().decode() == '@$!'
    assert m.bytearray_from_string().decode() == 'foo'
    assert m.bytearray_size() == len('foo')

def test_capsule(capture):
    if False:
        while True:
            i = 10
    pytest.gc_collect()
    with capture:
        a = m.return_capsule_with_destructor()
        del a
        pytest.gc_collect()
    assert capture.unordered == '\n        creating capsule\n        destructing capsule\n    '
    with capture:
        a = m.return_renamed_capsule_with_destructor()
        del a
        pytest.gc_collect()
    assert capture.unordered == '\n        creating capsule\n        renaming capsule\n        destructing capsule\n    '
    with capture:
        a = m.return_capsule_with_destructor_2()
        del a
        pytest.gc_collect()
    assert capture.unordered == '\n        creating capsule\n        destructing capsule: 1234\n    '
    with capture:
        a = m.return_capsule_with_destructor_3()
        del a
        pytest.gc_collect()
    assert capture.unordered == '\n        creating capsule\n        destructing capsule: 1233\n        original name: oname\n    '
    with capture:
        a = m.return_renamed_capsule_with_destructor_2()
        del a
        pytest.gc_collect()
    assert capture.unordered == '\n        creating capsule\n        renaming capsule\n        destructing capsule: 1234\n    '
    with capture:
        a = m.return_capsule_with_name_and_destructor()
        del a
        pytest.gc_collect()
    assert capture.unordered == "\n        created capsule (1234, 'pointer type description')\n        destructing capsule (1234, 'pointer type description')\n    "
    with capture:
        a = m.return_capsule_with_explicit_nullptr_dtor()
        del a
        pytest.gc_collect()
    assert capture.unordered == '\n        creating capsule with explicit nullptr dtor\n    '

def test_accessors():
    if False:
        return 10

    class SubTestObject:
        attr_obj = 1
        attr_char = 2

    class TestObject:
        basic_attr = 1
        begin_end = [1, 2, 3]
        d = {'operator[object]': 1, 'operator[char *]': 2}
        sub = SubTestObject()

        def func(self, x, *args):
            if False:
                while True:
                    i = 10
            return self.basic_attr + x + sum(args)
    d = m.accessor_api(TestObject())
    assert d['basic_attr'] == 1
    assert d['begin_end'] == [1, 2, 3]
    assert d['operator[object]'] == 1
    assert d['operator[char *]'] == 2
    assert d['attr(object)'] == 1
    assert d['attr(char *)'] == 2
    assert d['missing_attr_ptr'] == 'raised'
    assert d['missing_attr_chain'] == 'raised'
    assert d['is_none'] is False
    assert d['operator()'] == 2
    assert d['operator*'] == 7
    assert d['implicit_list'] == [1, 2, 3]
    assert all((x in TestObject.__dict__ for x in d['implicit_dict']))
    assert m.tuple_accessor(()) == (0, 1, 2)
    d = m.accessor_assignment()
    assert d['get'] == 0
    assert d['deferred_get'] == 0
    assert d['set'] == 1
    assert d['deferred_set'] == 1
    assert d['var'] == 99

def test_accessor_moves():
    if False:
        i = 10
        return i + 15
    inc_refs = m.accessor_moves()
    if inc_refs:
        assert inc_refs == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    else:
        pytest.skip('Not defined: PYBIND11_HANDLE_REF_DEBUG')

def test_constructors():
    if False:
        while True:
            i = 10
    'C++ default and converting constructors are equivalent to type calls in Python'
    types = [bytes, bytearray, str, bool, int, float, tuple, list, dict, set]
    expected = {t.__name__: t() for t in types}
    assert m.default_constructors() == expected
    data = {bytes: b'41', bytearray: bytearray(b'41'), str: 42, bool: 'Not empty', int: '42', float: '+1e3', tuple: range(3), list: range(3), dict: [('two', 2), ('one', 1), ('three', 3)], set: [4, 4, 5, 6, 6, 6], frozenset: [4, 4, 5, 6, 6, 6], memoryview: b'abc'}
    inputs = {k.__name__: v for (k, v) in data.items()}
    expected = {k.__name__: k(v) for (k, v) in data.items()}
    assert m.converting_constructors(inputs) == expected
    assert m.cast_functions(inputs) == expected
    noconv1 = m.converting_constructors(expected)
    for k in noconv1:
        assert noconv1[k] is expected[k]
    noconv2 = m.cast_functions(expected)
    for k in noconv2:
        assert noconv2[k] is expected[k]

def test_non_converting_constructors():
    if False:
        while True:
            i = 10
    non_converting_test_cases = [('bytes', range(10)), ('none', 42), ('ellipsis', 42), ('type', 42)]
    for (t, v) in non_converting_test_cases:
        for move in [True, False]:
            with pytest.raises(TypeError) as excinfo:
                m.nonconverting_constructor(t, v, move)
            expected_error = f"Object of type '{type(v).__name__}' is not an instance of '{t}'"
            assert str(excinfo.value) == expected_error

def test_pybind11_str_raw_str():
    if False:
        i = 10
        return i + 15
    cvt = m.convert_to_pybind11_str
    assert cvt('Str') == 'Str'
    assert cvt(b'Bytes') == "b'Bytes'"
    assert cvt(None) == 'None'
    assert cvt(False) == 'False'
    assert cvt(True) == 'True'
    assert cvt(42) == '42'
    assert cvt(2 ** 65) == '36893488147419103232'
    assert cvt(-1.5) == '-1.5'
    assert cvt(()) == '()'
    assert cvt((18,)) == '(18,)'
    assert cvt([]) == '[]'
    assert cvt([28]) == '[28]'
    assert cvt({}) == '{}'
    assert cvt({3: 4}) == '{3: 4}'
    assert cvt(set()) == 'set()'
    assert cvt({3}) == '{3}'
    valid_orig = 'Ǳ'
    valid_utf8 = valid_orig.encode('utf-8')
    valid_cvt = cvt(valid_utf8)
    if hasattr(m, 'PYBIND11_STR_LEGACY_PERMISSIVE'):
        assert valid_cvt is valid_utf8
    else:
        assert type(valid_cvt) is str
        assert valid_cvt == "b'\\xc7\\xb1'"
    malformed_utf8 = b'\x80'
    if hasattr(m, 'PYBIND11_STR_LEGACY_PERMISSIVE'):
        assert cvt(malformed_utf8) is malformed_utf8
    else:
        malformed_cvt = cvt(malformed_utf8)
        assert type(malformed_cvt) is str
        assert malformed_cvt == "b'\\x80'"

def test_implicit_casting():
    if False:
        print('Hello World!')
    'Tests implicit casting when assigning or appending to dicts and lists.'
    z = m.get_implicit_casting()
    assert z['d'] == {'char*_i1': 'abc', 'char*_i2': 'abc', 'char*_e': 'abc', 'char*_p': 'abc', 'str_i1': 'str', 'str_i2': 'str1', 'str_e': 'str2', 'str_p': 'str3', 'int_i1': 42, 'int_i2': 42, 'int_e': 43, 'int_p': 44}
    assert z['l'] == [3, 6, 9, 12, 15]

def test_print(capture):
    if False:
        print('Hello World!')
    with capture:
        m.print_function()
    assert capture == '\n        Hello, World!\n        1 2.0 three True -- multiple args\n        *args-and-a-custom-separator\n        no new line here -- next print\n        flush\n        py::print + str.format = this\n    '
    assert capture.stderr == 'this goes to stderr'
    with pytest.raises(RuntimeError) as excinfo:
        m.print_failure()
    assert str(excinfo.value) == 'Unable to convert call argument ' + ("'1' of type 'UnregisteredType' to Python object" if detailed_error_messages_enabled else "'1' to Python object (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)")

def test_hash():
    if False:
        print('Hello World!')

    class Hashable:

        def __init__(self, value):
            if False:
                return 10
            self.value = value

        def __hash__(self):
            if False:
                i = 10
                return i + 15
            return self.value

    class Unhashable:
        __hash__ = None
    assert m.hash_function(Hashable(42)) == 42
    with pytest.raises(TypeError):
        m.hash_function(Unhashable())

def test_number_protocol():
    if False:
        i = 10
        return i + 15
    for (a, b) in [(1, 1), (3, 5)]:
        li = [a == b, a != b, a < b, a <= b, a > b, a >= b, a + b, a - b, a * b, a / b, a | b, a & b, a ^ b, a >> b, a << b]
        assert m.test_number_protocol(a, b) == li

def test_list_slicing():
    if False:
        return 10
    li = list(range(100))
    assert li[::2] == m.test_list_slicing(li)

def test_issue2361():
    if False:
        print('Hello World!')
    assert m.issue2361_str_implicit_copy_none() == 'None'
    with pytest.raises(TypeError) as excinfo:
        assert m.issue2361_dict_implicit_copy_none()
    assert 'NoneType' in str(excinfo.value)
    assert 'iterable' in str(excinfo.value)

@pytest.mark.parametrize(('method', 'args', 'fmt', 'expected_view'), [(m.test_memoryview_object, (b'red',), 'B', b'red'), (m.test_memoryview_buffer_info, (b'green',), 'B', b'green'), (m.test_memoryview_from_buffer, (False,), 'h', [3, 1, 4, 1, 5]), (m.test_memoryview_from_buffer, (True,), 'H', [2, 7, 1, 8]), (m.test_memoryview_from_buffer_nativeformat, (), '@i', [4, 7, 5])])
def test_memoryview(method, args, fmt, expected_view):
    if False:
        i = 10
        return i + 15
    view = method(*args)
    assert isinstance(view, memoryview)
    assert view.format == fmt
    assert list(view) == list(expected_view)

@pytest.mark.xfail('env.PYPY', reason='getrefcount is not available')
@pytest.mark.parametrize('method', [m.test_memoryview_object, m.test_memoryview_buffer_info])
def test_memoryview_refcount(method):
    if False:
        return 10
    buf = b'\n\x0b\x0c\r'
    ref_before = sys.getrefcount(buf)
    view = method(buf)
    ref_after = sys.getrefcount(buf)
    assert ref_before < ref_after
    assert list(view) == list(buf)

def test_memoryview_from_buffer_empty_shape():
    if False:
        while True:
            i = 10
    view = m.test_memoryview_from_buffer_empty_shape()
    assert isinstance(view, memoryview)
    assert view.format == 'B'
    assert bytes(view) == b''

def test_test_memoryview_from_buffer_invalid_strides():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):
        m.test_memoryview_from_buffer_invalid_strides()

def test_test_memoryview_from_buffer_nullptr():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        m.test_memoryview_from_buffer_nullptr()

def test_memoryview_from_memory():
    if False:
        while True:
            i = 10
    view = m.test_memoryview_from_memory()
    assert isinstance(view, memoryview)
    assert view.format == 'B'
    assert bytes(view) == b'\xff\xe1\xab7'

def test_builtin_functions():
    if False:
        while True:
            i = 10
    assert m.get_len(list(range(42))) == 42
    with pytest.raises(TypeError) as exc_info:
        m.get_len((i for i in range(42)))
    assert str(exc_info.value) in ["object of type 'generator' has no len()", "'generator' has no length"]

def test_isinstance_string_types():
    if False:
        while True:
            i = 10
    assert m.isinstance_pybind11_bytes(b'')
    assert not m.isinstance_pybind11_bytes('')
    assert m.isinstance_pybind11_str('')
    if hasattr(m, 'PYBIND11_STR_LEGACY_PERMISSIVE'):
        assert m.isinstance_pybind11_str(b'')
    else:
        assert not m.isinstance_pybind11_str(b'')

def test_pass_bytes_or_unicode_to_string_types():
    if False:
        return 10
    assert m.pass_to_pybind11_bytes(b'Bytes') == 5
    with pytest.raises(TypeError):
        m.pass_to_pybind11_bytes('Str')
    if hasattr(m, 'PYBIND11_STR_LEGACY_PERMISSIVE'):
        assert m.pass_to_pybind11_str(b'Bytes') == 5
    else:
        with pytest.raises(TypeError):
            m.pass_to_pybind11_str(b'Bytes')
    assert m.pass_to_pybind11_str('Str') == 3
    assert m.pass_to_std_string(b'Bytes') == 5
    assert m.pass_to_std_string('Str') == 3
    malformed_utf8 = b'\x80'
    if hasattr(m, 'PYBIND11_STR_LEGACY_PERMISSIVE'):
        assert m.pass_to_pybind11_str(malformed_utf8) == 1
    else:
        with pytest.raises(TypeError):
            m.pass_to_pybind11_str(malformed_utf8)

@pytest.mark.parametrize(('create_weakref', 'create_weakref_with_callback'), [(m.weakref_from_handle, m.weakref_from_handle_and_function), (m.weakref_from_object, m.weakref_from_object_and_function)])
def test_weakref(create_weakref, create_weakref_with_callback):
    if False:
        print('Hello World!')
    from weakref import getweakrefcount

    class WeaklyReferenced:
        pass
    callback_called = False

    def callback(_):
        if False:
            return 10
        nonlocal callback_called
        callback_called = True
    obj = WeaklyReferenced()
    assert getweakrefcount(obj) == 0
    wr = create_weakref(obj)
    assert getweakrefcount(obj) == 1
    obj = WeaklyReferenced()
    assert getweakrefcount(obj) == 0
    wr = create_weakref_with_callback(obj, callback)
    assert getweakrefcount(obj) == 1
    assert not callback_called
    del obj
    pytest.gc_collect()
    assert callback_called

@pytest.mark.parametrize(('create_weakref', 'has_callback'), [(m.weakref_from_handle, False), (m.weakref_from_object, False), (m.weakref_from_handle_and_function, True), (m.weakref_from_object_and_function, True)])
def test_weakref_err(create_weakref, has_callback):
    if False:
        for i in range(10):
            print('nop')

    class C:
        __slots__ = []

    def callback(_):
        if False:
            while True:
                i = 10
        pass
    ob = C()
    with pytest.raises(TypeError) if not env.PYPY else contextlib.nullcontext():
        _ = create_weakref(ob, callback) if has_callback else create_weakref(ob)

def test_cpp_iterators():
    if False:
        print('Hello World!')
    assert m.tuple_iterator() == 12
    assert m.dict_iterator() == 305 + 711
    assert m.passed_iterator(iter((-7, 3))) == -4

def test_implementation_details():
    if False:
        return 10
    lst = [39, 43, 92, 49, 22, 29, 93, 98, 26, 57, 8]
    tup = tuple(lst)
    assert m.sequence_item_get_ssize_t(lst) == 43
    assert m.sequence_item_set_ssize_t(lst) is None
    assert lst[1] == 'peppa'
    assert m.sequence_item_get_size_t(lst) == 92
    assert m.sequence_item_set_size_t(lst) is None
    assert lst[2] == 'george'
    assert m.list_item_get_ssize_t(lst) == 49
    assert m.list_item_set_ssize_t(lst) is None
    assert lst[3] == 'rebecca'
    assert m.list_item_get_size_t(lst) == 22
    assert m.list_item_set_size_t(lst) is None
    assert lst[4] == 'richard'
    assert m.tuple_item_get_ssize_t(tup) == 29
    assert m.tuple_item_set_ssize_t() == ('emely', 'edmond')
    assert m.tuple_item_get_size_t(tup) == 93
    assert m.tuple_item_set_size_t() == ('candy', 'cat')

def test_external_float_():
    if False:
        for i in range(10):
            print('nop')
    r1 = m.square_float_(2.0)
    assert r1 == 4.0

def test_tuple_rvalue_getter():
    if False:
        return 10
    pop = 1000
    tup = tuple(range(pop))
    m.tuple_rvalue_getter(tup)

def test_list_rvalue_getter():
    if False:
        for i in range(10):
            print('nop')
    pop = 1000
    my_list = list(range(pop))
    m.list_rvalue_getter(my_list)

def test_populate_dict_rvalue():
    if False:
        return 10
    pop = 1000
    my_dict = {i: i for i in range(pop)}
    assert m.populate_dict_rvalue(pop) == my_dict

def test_populate_obj_str_attrs():
    if False:
        while True:
            i = 10
    pop = 1000
    o = types.SimpleNamespace(**{str(i): i for i in range(pop)})
    new_o = m.populate_obj_str_attrs(o, pop)
    new_attrs = {k: v for (k, v) in new_o.__dict__.items() if not k.startswith('_')}
    assert all((isinstance(v, str) for v in new_attrs.values()))
    assert len(new_attrs) == pop

@pytest.mark.parametrize(('a', 'b'), [('foo', 'bar'), (1, 2), (1.0, 2.0), (list(range(3)), list(range(3, 6)))])
def test_inplace_append(a, b):
    if False:
        return 10
    expected = a + b
    assert m.inplace_append(a, b) == expected

@pytest.mark.parametrize(('a', 'b'), [(3, 2), (3.0, 2.0), (set(range(3)), set(range(2)))])
def test_inplace_subtract(a, b):
    if False:
        while True:
            i = 10
    expected = a - b
    assert m.inplace_subtract(a, b) == expected

@pytest.mark.parametrize(('a', 'b'), [(3, 2), (3.0, 2.0), ([1], 3)])
def test_inplace_multiply(a, b):
    if False:
        i = 10
        return i + 15
    expected = a * b
    assert m.inplace_multiply(a, b) == expected

@pytest.mark.parametrize(('a', 'b'), [(6, 3), (6.0, 3.0)])
def test_inplace_divide(a, b):
    if False:
        while True:
            i = 10
    expected = a / b
    assert m.inplace_divide(a, b) == expected

@pytest.mark.parametrize(('a', 'b'), [(False, True), (set(), {1})])
def test_inplace_or(a, b):
    if False:
        i = 10
        return i + 15
    expected = a | b
    assert m.inplace_or(a, b) == expected

@pytest.mark.parametrize(('a', 'b'), [(True, False), ({1, 2, 3}, {1})])
def test_inplace_and(a, b):
    if False:
        print('Hello World!')
    expected = a & b
    assert m.inplace_and(a, b) == expected

@pytest.mark.parametrize(('a', 'b'), [(8, 1), (-3, 2)])
def test_inplace_lshift(a, b):
    if False:
        while True:
            i = 10
    expected = a << b
    assert m.inplace_lshift(a, b) == expected

@pytest.mark.parametrize(('a', 'b'), [(8, 1), (-2, 2)])
def test_inplace_rshift(a, b):
    if False:
        print('Hello World!')
    expected = a >> b
    assert m.inplace_rshift(a, b) == expected

def test_tuple_nonempty_annotations(doc):
    if False:
        while True:
            i = 10
    assert doc(m.annotate_tuple_float_str) == 'annotate_tuple_float_str(arg0: tuple[float, str]) -> None'

def test_tuple_empty_annotations(doc):
    if False:
        return 10
    assert doc(m.annotate_tuple_empty) == 'annotate_tuple_empty(arg0: tuple[()]) -> None'

def test_dict_annotations(doc):
    if False:
        return 10
    assert doc(m.annotate_dict_str_int) == 'annotate_dict_str_int(arg0: dict[str, int]) -> None'

def test_list_annotations(doc):
    if False:
        return 10
    assert doc(m.annotate_list_int) == 'annotate_list_int(arg0: list[int]) -> None'

def test_set_annotations(doc):
    if False:
        return 10
    assert doc(m.annotate_set_str) == 'annotate_set_str(arg0: set[str]) -> None'

def test_iterable_annotations(doc):
    if False:
        for i in range(10):
            print('nop')
    assert doc(m.annotate_iterable_str) == 'annotate_iterable_str(arg0: Iterable[str]) -> None'

def test_iterator_annotations(doc):
    if False:
        for i in range(10):
            print('nop')
    assert doc(m.annotate_iterator_int) == 'annotate_iterator_int(arg0: Iterator[int]) -> None'

def test_fn_annotations(doc):
    if False:
        print('Hello World!')
    assert doc(m.annotate_fn) == 'annotate_fn(arg0: Callable[[list[str], str], int]) -> None'