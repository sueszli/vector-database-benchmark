import pytest
from pybind11_tests import ExamplePythonTypes, ConstructorStats, has_optional, has_exp_optional

def test_repr():
    if False:
        while True:
            i = 10
    assert 'ExamplePythonTypes__Meta' in repr(type(ExamplePythonTypes))
    assert 'ExamplePythonTypes' in repr(ExamplePythonTypes)

def test_static():
    if False:
        i = 10
        return i + 15
    ExamplePythonTypes.value = 15
    assert ExamplePythonTypes.value == 15
    assert ExamplePythonTypes.value2 == 5
    with pytest.raises(AttributeError) as excinfo:
        ExamplePythonTypes.value2 = 15
    assert str(excinfo.value) == "can't set attribute"

def test_instance(capture):
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError) as excinfo:
        ExamplePythonTypes()
    assert str(excinfo.value) == 'pybind11_tests.ExamplePythonTypes: No constructor defined!'
    instance = ExamplePythonTypes.new_instance()
    with capture:
        dict_result = instance.get_dict()
        dict_result['key2'] = 'value2'
        instance.print_dict(dict_result)
    assert capture.unordered == '\n        key: key, value=value\n        key: key2, value=value2\n    '
    with capture:
        dict_result = instance.get_dict_2()
        dict_result['key2'] = 'value2'
        instance.print_dict_2(dict_result)
    assert capture.unordered == '\n        key: key, value=value\n        key: key2, value=value2\n    '
    with capture:
        set_result = instance.get_set()
        set_result.add('key4')
        instance.print_set(set_result)
    assert capture.unordered == '\n        key: key1\n        key: key2\n        key: key3\n        key: key4\n    '
    with capture:
        set_result = instance.get_set2()
        set_result.add('key3')
        instance.print_set_2(set_result)
    assert capture.unordered == '\n        key: key1\n        key: key2\n        key: key3\n    '
    with capture:
        list_result = instance.get_list()
        list_result.append('value2')
        instance.print_list(list_result)
    assert capture.unordered == '\n        Entry at position 0: value\n        list item 0: overwritten\n        list item 1: value2\n    '
    with capture:
        list_result = instance.get_list_2()
        list_result.append('value2')
        instance.print_list_2(list_result)
    assert capture.unordered == '\n        list item 0: value\n        list item 1: value2\n    '
    with capture:
        list_result = instance.get_list_2()
        list_result.append('value2')
        instance.print_list_2(tuple(list_result))
    assert capture.unordered == '\n        list item 0: value\n        list item 1: value2\n    '
    array_result = instance.get_array()
    assert array_result == ['array entry 1', 'array entry 2']
    with capture:
        instance.print_array(array_result)
    assert capture.unordered == '\n        array item 0: array entry 1\n        array item 1: array entry 2\n    '
    varray_result = instance.get_valarray()
    assert varray_result == [1, 4, 9]
    with capture:
        instance.print_valarray(varray_result)
    assert capture.unordered == '\n        valarray item 0: 1\n        valarray item 1: 4\n        valarray item 2: 9\n    '
    with pytest.raises(RuntimeError) as excinfo:
        instance.throw_exception()
    assert str(excinfo.value) == 'This exception was intentionally thrown.'
    assert instance.pair_passthrough((True, 'test')) == ('test', True)
    assert instance.tuple_passthrough((True, 'test', 5)) == (5, 'test', True)
    assert instance.pair_passthrough([True, 'test']) == ('test', True)
    assert instance.tuple_passthrough([True, 'test', 5]) == (5, 'test', True)
    assert instance.get_bytes_from_string().decode() == 'foo'
    assert instance.get_bytes_from_str().decode() == 'bar'
    assert instance.get_str_from_string().encode().decode() == 'baz'
    assert instance.get_str_from_bytes().encode().decode() == 'boo'

    class A(object):

        def __str__(self):
            if False:
                return 10
            return 'this is a str'

        def __repr__(self):
            if False:
                print('Hello World!')
            return 'this is a repr'
    with capture:
        instance.test_print(A())
    assert capture == '\n        this is a str\n        this is a repr\n    '
    cstats = ConstructorStats.get(ExamplePythonTypes)
    assert cstats.alive() == 1
    del instance
    assert cstats.alive() == 0

def test_class_docs(doc):
    if False:
        print('Hello World!')
    assert doc(ExamplePythonTypes) == 'Example 2 documentation'

def test_method_docs(doc):
    if False:
        for i in range(10):
            print('nop')
    assert doc(ExamplePythonTypes.get_dict) == '\n        get_dict(self: m.ExamplePythonTypes) -> dict\n\n        Return a Python dictionary\n    '
    assert doc(ExamplePythonTypes.get_dict_2) == '\n        get_dict_2(self: m.ExamplePythonTypes) -> Dict[str, str]\n\n        Return a C++ dictionary\n    '
    assert doc(ExamplePythonTypes.get_list) == '\n        get_list(self: m.ExamplePythonTypes) -> list\n\n        Return a Python list\n    '
    assert doc(ExamplePythonTypes.get_list_2) == '\n        get_list_2(self: m.ExamplePythonTypes) -> List[str]\n\n        Return a C++ list\n    '
    assert doc(ExamplePythonTypes.get_dict) == '\n        get_dict(self: m.ExamplePythonTypes) -> dict\n\n        Return a Python dictionary\n    '
    assert doc(ExamplePythonTypes.get_set) == '\n        get_set(self: m.ExamplePythonTypes) -> set\n\n        Return a Python set\n    '
    assert doc(ExamplePythonTypes.get_set2) == '\n        get_set2(self: m.ExamplePythonTypes) -> Set[str]\n\n        Return a C++ set\n    '
    assert doc(ExamplePythonTypes.get_array) == '\n        get_array(self: m.ExamplePythonTypes) -> List[str[2]]\n\n        Return a C++ array\n    '
    assert doc(ExamplePythonTypes.get_valarray) == '\n        get_valarray(self: m.ExamplePythonTypes) -> List[int]\n\n        Return a C++ valarray\n    '
    assert doc(ExamplePythonTypes.print_dict) == '\n        print_dict(self: m.ExamplePythonTypes, arg0: dict) -> None\n\n        Print entries of a Python dictionary\n    '
    assert doc(ExamplePythonTypes.print_dict_2) == '\n        print_dict_2(self: m.ExamplePythonTypes, arg0: Dict[str, str]) -> None\n\n        Print entries of a C++ dictionary\n    '
    assert doc(ExamplePythonTypes.print_set) == '\n        print_set(self: m.ExamplePythonTypes, arg0: set) -> None\n\n        Print entries of a Python set\n    '
    assert doc(ExamplePythonTypes.print_set_2) == '\n        print_set_2(self: m.ExamplePythonTypes, arg0: Set[str]) -> None\n\n        Print entries of a C++ set\n    '
    assert doc(ExamplePythonTypes.print_list) == '\n        print_list(self: m.ExamplePythonTypes, arg0: list) -> None\n\n        Print entries of a Python list\n    '
    assert doc(ExamplePythonTypes.print_list_2) == '\n        print_list_2(self: m.ExamplePythonTypes, arg0: List[str]) -> None\n\n        Print entries of a C++ list\n    '
    assert doc(ExamplePythonTypes.print_array) == '\n        print_array(self: m.ExamplePythonTypes, arg0: List[str[2]]) -> None\n\n        Print entries of a C++ array\n    '
    assert doc(ExamplePythonTypes.pair_passthrough) == '\n        pair_passthrough(self: m.ExamplePythonTypes, arg0: Tuple[bool, str]) -> Tuple[str, bool]\n\n        Return a pair in reversed order\n    '
    assert doc(ExamplePythonTypes.tuple_passthrough) == '\n        tuple_passthrough(self: m.ExamplePythonTypes, arg0: Tuple[bool, str, int]) -> Tuple[int, str, bool]\n\n        Return a triple in reversed order\n    '
    assert doc(ExamplePythonTypes.throw_exception) == '\n        throw_exception(self: m.ExamplePythonTypes) -> None\n\n        Throw an exception\n    '
    assert doc(ExamplePythonTypes.new_instance) == '\n        new_instance() -> m.ExamplePythonTypes\n\n        Return an instance\n    '

def test_module():
    if False:
        print('Hello World!')
    import pybind11_tests
    assert pybind11_tests.__name__ == 'pybind11_tests'
    assert ExamplePythonTypes.__name__ == 'ExamplePythonTypes'
    assert ExamplePythonTypes.__module__ == 'pybind11_tests'
    assert ExamplePythonTypes.get_set.__name__ == 'get_set'
    assert ExamplePythonTypes.get_set.__module__ == 'pybind11_tests'

def test_print(capture):
    if False:
        print('Hello World!')
    from pybind11_tests import test_print_function
    with capture:
        test_print_function()
    assert capture == '\n        Hello, World!\n        1 2.0 three True -- multiple args\n        *args-and-a-custom-separator\n        no new line here -- next print\n        flush\n        py::print + str.format = this\n    '
    assert capture.stderr == 'this goes to stderr'

def test_str_api():
    if False:
        while True:
            i = 10
    from pybind11_tests import test_str_format
    (s1, s2) = test_str_format()
    assert s1 == '1 + 2 = 3'
    assert s1 == s2

def test_dict_api():
    if False:
        print('Hello World!')
    from pybind11_tests import test_dict_keyword_constructor
    assert test_dict_keyword_constructor() == {'x': 1, 'y': 2, 'z': 3}

def test_accessors():
    if False:
        while True:
            i = 10
    from pybind11_tests import test_accessor_api, test_tuple_accessor, test_accessor_assignment

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
                return 10
            return self.basic_attr + x + sum(args)
    d = test_accessor_api(TestObject())
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
    assert test_tuple_accessor(tuple()) == (0, 1, 2)
    d = test_accessor_assignment()
    assert d['get'] == 0
    assert d['deferred_get'] == 0
    assert d['set'] == 1
    assert d['deferred_set'] == 1
    assert d['var'] == 99

@pytest.mark.skipif(not has_optional, reason='no <optional>')
def test_optional():
    if False:
        for i in range(10):
            print('nop')
    from pybind11_tests import double_or_zero, half_or_none, test_nullopt
    assert double_or_zero(None) == 0
    assert double_or_zero(42) == 84
    pytest.raises(TypeError, double_or_zero, 'foo')
    assert half_or_none(0) is None
    assert half_or_none(42) == 21
    pytest.raises(TypeError, half_or_none, 'foo')
    assert test_nullopt() == 42
    assert test_nullopt(None) == 42
    assert test_nullopt(42) == 42
    assert test_nullopt(43) == 43

@pytest.mark.skipif(not has_exp_optional, reason='no <experimental/optional>')
def test_exp_optional():
    if False:
        i = 10
        return i + 15
    from pybind11_tests import double_or_zero_exp, half_or_none_exp, test_nullopt_exp
    assert double_or_zero_exp(None) == 0
    assert double_or_zero_exp(42) == 84
    pytest.raises(TypeError, double_or_zero_exp, 'foo')
    assert half_or_none_exp(0) is None
    assert half_or_none_exp(42) == 21
    pytest.raises(TypeError, half_or_none_exp, 'foo')
    assert test_nullopt_exp() == 42
    assert test_nullopt_exp(None) == 42
    assert test_nullopt_exp(42) == 42
    assert test_nullopt_exp(43) == 43

def test_constructors():
    if False:
        for i in range(10):
            print('nop')
    'C++ default and converting constructors are equivalent to type calls in Python'
    from pybind11_tests import test_default_constructors, test_converting_constructors, test_cast_functions
    types = [str, bool, int, float, tuple, list, dict, set]
    expected = {t.__name__: t() for t in types}
    assert test_default_constructors() == expected
    data = {str: 42, bool: 'Not empty', int: '42', float: '+1e3', tuple: range(3), list: range(3), dict: [('two', 2), ('one', 1), ('three', 3)], set: [4, 4, 5, 6, 6, 6], memoryview: b'abc'}
    inputs = {k.__name__: v for (k, v) in data.items()}
    expected = {k.__name__: k(v) for (k, v) in data.items()}
    assert test_converting_constructors(inputs) == expected
    assert test_cast_functions(inputs) == expected

def test_move_out_container():
    if False:
        print('Hello World!')
    'Properties use the `reference_internal` policy by default. If the underlying function\n    returns an rvalue, the policy is automatically changed to `move` to avoid referencing\n    a temporary. In case the return value is a container of user-defined types, the policy\n    also needs to be applied to the elements, not just the container.'
    from pybind11_tests import MoveOutContainer
    c = MoveOutContainer()
    moved_out_list = c.move_list
    assert [x.value for x in moved_out_list] == [0, 1, 2]

def test_implicit_casting():
    if False:
        return 10
    'Tests implicit casting when assigning or appending to dicts and lists.'
    from pybind11_tests import get_implicit_casting
    z = get_implicit_casting()
    assert z['d'] == {'char*_i1': 'abc', 'char*_i2': 'abc', 'char*_e': 'abc', 'char*_p': 'abc', 'str_i1': 'str', 'str_i2': 'str1', 'str_e': 'str2', 'str_p': 'str3', 'int_i1': 42, 'int_i2': 42, 'int_e': 43, 'int_p': 44}
    assert z['l'] == [3, 6, 9, 12, 15]

def test_unicode_conversion():
    if False:
        i = 10
        return i + 15
    'Tests unicode conversion and error reporting.'
    import pybind11_tests
    from pybind11_tests import good_utf8_string, bad_utf8_string, good_utf16_string, bad_utf16_string, good_utf32_string, good_wchar_string, u8_Z, u8_eacute, u16_ibang, u32_mathbfA, wchar_heart
    assert good_utf8_string() == u'Say utf8‽ 🎂 𝐀'
    assert good_utf16_string() == u'b‽🎂𝐀z'
    assert good_utf32_string() == u'a𝐀🎂‽z'
    assert good_wchar_string() == u'a⸘𝐀z'
    with pytest.raises(UnicodeDecodeError):
        bad_utf8_string()
    with pytest.raises(UnicodeDecodeError):
        bad_utf16_string()
    if hasattr(pybind11_tests, 'bad_utf32_string'):
        with pytest.raises(UnicodeDecodeError):
            pybind11_tests.bad_utf32_string()
    if hasattr(pybind11_tests, 'bad_wchar_string'):
        with pytest.raises(UnicodeDecodeError):
            pybind11_tests.bad_wchar_string()
    assert u8_Z() == 'Z'
    assert u8_eacute() == u'é'
    assert u16_ibang() == u'‽'
    assert u32_mathbfA() == u'𝐀'
    assert wchar_heart() == u'♥'

def test_single_char_arguments():
    if False:
        return 10
    'Tests failures for passing invalid inputs to char-accepting functions'
    from pybind11_tests import ord_char, ord_char16, ord_char32, ord_wchar, wchar_size

    def toobig_message(r):
        if False:
            i = 10
            return i + 15
        return 'Character code point not in range({0:#x})'.format(r)
    toolong_message = 'Expected a character, but multi-character string found'
    assert ord_char(u'a') == 97
    assert ord_char(u'é') == 233
    with pytest.raises(ValueError) as excinfo:
        assert ord_char(u'Ā') == 256
    assert str(excinfo.value) == toobig_message(256)
    with pytest.raises(ValueError) as excinfo:
        assert ord_char(u'ab')
    assert str(excinfo.value) == toolong_message
    assert ord_char16(u'a') == 97
    assert ord_char16(u'é') == 233
    assert ord_char16(u'Ā') == 256
    assert ord_char16(u'‽') == 8253
    assert ord_char16(u'♥') == 9829
    with pytest.raises(ValueError) as excinfo:
        assert ord_char16(u'🎂') == 127874
    assert str(excinfo.value) == toobig_message(65536)
    with pytest.raises(ValueError) as excinfo:
        assert ord_char16(u'aa')
    assert str(excinfo.value) == toolong_message
    assert ord_char32(u'a') == 97
    assert ord_char32(u'é') == 233
    assert ord_char32(u'Ā') == 256
    assert ord_char32(u'‽') == 8253
    assert ord_char32(u'♥') == 9829
    assert ord_char32(u'🎂') == 127874
    with pytest.raises(ValueError) as excinfo:
        assert ord_char32(u'aa')
    assert str(excinfo.value) == toolong_message
    assert ord_wchar(u'a') == 97
    assert ord_wchar(u'é') == 233
    assert ord_wchar(u'Ā') == 256
    assert ord_wchar(u'‽') == 8253
    assert ord_wchar(u'♥') == 9829
    if wchar_size == 2:
        with pytest.raises(ValueError) as excinfo:
            assert ord_wchar(u'🎂') == 127874
        assert str(excinfo.value) == toobig_message(65536)
    else:
        assert ord_wchar(u'🎂') == 127874
    with pytest.raises(ValueError) as excinfo:
        assert ord_wchar(u'aa')
    assert str(excinfo.value) == toolong_message