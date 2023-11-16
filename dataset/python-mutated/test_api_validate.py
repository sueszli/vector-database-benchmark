import re
from textwrap import dedent
import pytest
from lxml.etree import Element, tostring as etree_tostring
from streamlink.exceptions import PluginError
from streamlink.plugin.api import validate
from streamlink.plugin.api.validate._exception import ValidationError

def assert_validationerror(exception, expected):
    if False:
        i = 10
        return i + 15
    assert str(exception) == dedent(expected).strip('\n')

class TestSchema:

    @pytest.fixture(scope='class')
    def schema(self):
        if False:
            print('Hello World!')
        return validate.Schema(str, 'foo')

    @pytest.fixture(scope='class')
    def schema_nested(self, schema: validate.Schema):
        if False:
            for i in range(10):
                print('nop')
        return validate.Schema(schema)

    def test_validate_success(self, schema: validate.Schema):
        if False:
            return 10
        assert schema.validate('foo') == 'foo'

    def test_validate_failure(self, schema: validate.Schema):
        if False:
            while True:
                i = 10
        with pytest.raises(PluginError) as cm:
            schema.validate('bar')
        assert_validationerror(cm.value, "\n            Unable to validate result: ValidationError(equality):\n              'bar' does not equal 'foo'\n        ")

    def test_validate_failure_custom(self, schema: validate.Schema):
        if False:
            i = 10
            return i + 15

        class CustomError(PluginError):
            pass
        with pytest.raises(CustomError) as cm:
            schema.validate('bar', name='data', exception=CustomError)
        assert_validationerror(cm.value, "\n            Unable to validate data: ValidationError(equality):\n              'bar' does not equal 'foo'\n        ")

    def test_nested_success(self, schema_nested: validate.Schema):
        if False:
            return 10
        assert schema_nested.validate('foo') == 'foo'

    def test_nested_failure(self, schema_nested: validate.Schema):
        if False:
            while True:
                i = 10
        with pytest.raises(PluginError) as cm:
            schema_nested.validate('bar')
        assert_validationerror(cm.value, "\n            Unable to validate result: ValidationError(equality):\n              'bar' does not equal 'foo'\n        ")

class TestEquality:

    def test_success(self):
        if False:
            while True:
                i = 10
        assert validate.validate('foo', 'foo') == 'foo'

    def test_failure(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate('foo', 'bar')
        assert_validationerror(cm.value, "\n            ValidationError(equality):\n              'bar' does not equal 'foo'\n        ")

class TestType:

    def test_success(self):
        if False:
            i = 10
            return i + 15

        class A:
            pass

        class B(A):
            pass
        a = A()
        b = B()
        assert validate.validate(A, a) is a
        assert validate.validate(B, b) is b
        assert validate.validate(A, b) is b

    def test_failure(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(int, '1')
        assert_validationerror(cm.value, "\n            ValidationError(type):\n              Type of '1' should be int, but is str\n        ")

class TestSequence:

    @pytest.mark.parametrize(('schema', 'value'), [([3, 2, 1, 0], [1, 2]), ((3, 2, 1, 0), (1, 2)), ({3, 2, 1, 0}, {1, 2}), (frozenset((3, 2, 1, 0)), frozenset((1, 2)))], ids=['list', 'tuple', 'set', 'frozenset'])
    def test_sequences(self, schema, value):
        if False:
            while True:
                i = 10
        result = validate.validate(schema, value)
        assert result == value
        assert result is not value

    def test_empty(self):
        if False:
            return 10
        assert validate.validate([1, 2, 3], []) == []

    def test_failure_items(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate([1, 2, 3], [3, 4, 5])
        assert_validationerror(cm.value, '\n            ValidationError(AnySchema):\n              ValidationError(equality):\n                4 does not equal 1\n              ValidationError(equality):\n                4 does not equal 2\n              ValidationError(equality):\n                4 does not equal 3\n        ')

    def test_failure_schema(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate([1, 2, 3], {1, 2, 3})
        assert_validationerror(cm.value, '\n            ValidationError(type):\n              Type of {1, 2, 3} should be list, but is set\n        ')

class TestDict:

    def test_simple(self):
        if False:
            while True:
                i = 10
        schema = {'foo': 'FOO', 'bar': str}
        value = {'foo': 'FOO', 'bar': 'BAR', 'baz': 'BAZ'}
        result = validate.validate(schema, value)
        assert result == {'foo': 'FOO', 'bar': 'BAR'}
        assert result is not value

    @pytest.mark.parametrize(('value', 'expected'), [({'foo': 'foo'}, {'foo': 'foo'}), ({'bar': 'bar'}, {})], ids=['existing', 'missing'])
    def test_optional(self, value, expected):
        if False:
            while True:
                i = 10
        assert validate.validate({validate.optional('foo'): 'foo'}, value) == expected

    @pytest.mark.parametrize(('schema', 'value', 'expected'), [({str: {int: str}}, {'foo': {1: 'foo'}}, {'foo': {1: 'foo'}}), ({validate.all(str, 'foo'): str}, {'foo': 'foo'}, {'foo': 'foo'}), ({validate.any(int, str): str}, {'foo': 'foo'}, {'foo': 'foo'}), ({validate.transform(lambda s: s.upper()): str}, {'foo': 'foo'}, {'FOO': 'foo'}), ({validate.union((str,)): str}, {'foo': 'foo'}, {('foo',): 'foo'})], ids=['type', 'AllSchema', 'AnySchema', 'TransformSchema', 'UnionSchema'])
    def test_keys(self, schema, value, expected):
        if False:
            while True:
                i = 10
        assert validate.validate(schema, value) == expected

    def test_failure_key(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate({str: int}, {'foo': 1, 2: 3})
        assert_validationerror(cm.value, '\n            ValidationError(dict):\n              Unable to validate key\n              Context(type):\n                Type of 2 should be str, but is int\n        ')

    def test_failure_key_value(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate({str: int}, {'foo': 'bar'})
        assert_validationerror(cm.value, "\n            ValidationError(dict):\n              Unable to validate value\n              Context(type):\n                Type of 'bar' should be int, but is str\n        ")

    def test_failure_notfound(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate({'foo': 'bar'}, {'baz': 'qux'})
        assert_validationerror(cm.value, "\n            ValidationError(dict):\n              Key 'foo' not found in {'baz': 'qux'}\n        ")

    def test_failure_value(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate({'foo': 'bar'}, {'foo': 1})
        assert_validationerror(cm.value, "\n            ValidationError(dict):\n              Unable to validate value of key 'foo'\n              Context(equality):\n                1 does not equal 'bar'\n        ")

    def test_failure_schema(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate({}, 1)
        assert_validationerror(cm.value, '\n            ValidationError(type):\n              Type of 1 should be dict, but is int\n        ')

class TestCallable:

    @staticmethod
    def subject(v):
        if False:
            for i in range(10):
                print('nop')
        return v is not None

    def test_success(self):
        if False:
            i = 10
            return i + 15
        value = object()
        assert validate.validate(self.subject, value) is value

    def test_failure(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(self.subject, None)
        assert_validationerror(cm.value, '\n            ValidationError(Callable):\n              subject(None) is not true\n        ')

class TestPattern:

    @pytest.mark.parametrize(('pattern', 'data', 'expected'), [('\\s(?P<bar>\\S+)\\s', 'foo bar baz', {'bar': 'bar'}), (b'\\s(?P<bar>\\S+)\\s', b'foo bar baz', {'bar': b'bar'})])
    def test_success(self, pattern, data, expected):
        if False:
            while True:
                i = 10
        result = validate.validate(re.compile(pattern), data)
        assert type(result) is re.Match
        assert result.groupdict() == expected

    def test_stringsubclass(self):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.all(validate.xml_xpath_string('.//@bar'), re.compile('.+'), validate.get(0)), Element('foo', {'bar': 'baz'})) == 'baz'

    def test_failure(self):
        if False:
            while True:
                i = 10
        assert validate.validate(re.compile('foo'), 'bar') is None

    def test_failure_type(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(re.compile('foo'), b'foo')
        assert_validationerror(cm.value, '\n            ValidationError(Pattern):\n              cannot use a string pattern on a bytes-like object\n        ')

    def test_failure_schema(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(re.compile('foo'), 123)
        assert_validationerror(cm.value, '\n            ValidationError(Pattern):\n              Type of 123 should be str or bytes, but is int\n        ')

class TestAllSchema:

    @pytest.fixture(scope='class')
    def schema(self):
        if False:
            return 10
        return validate.all(str, lambda string: string.startswith('f'), 'foo')

    def test_success(self, schema):
        if False:
            for i in range(10):
                print('nop')
        assert validate.validate(schema, 'foo') == 'foo'

    @pytest.mark.parametrize(('value', 'error'), [(123, '\n                    ValidationError(type):\n                      Type of 123 should be str, but is int\n                '), ('bar', "\n                    ValidationError(Callable):\n                      <lambda>('bar') is not true\n                "), ('failure', "\n                    ValidationError(equality):\n                      'failure' does not equal 'foo'\n                ")], ids=['first', 'second', 'third'])
    def test_failure(self, schema, value, error):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(schema, value)
        assert_validationerror(cm.value, error)

class TestAnySchema:

    @pytest.fixture(scope='class')
    def schema(self):
        if False:
            i = 10
            return i + 15
        return validate.any('foo', str, lambda data: data is not None)

    @pytest.mark.parametrize('value', ['foo', 'success', object()], ids=['first', 'second', 'third'])
    def test_success(self, schema, value):
        if False:
            return 10
        assert validate.validate(schema, value) is value

    def test_failure(self, schema):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(schema, None)
        assert_validationerror(cm.value, "\n            ValidationError(AnySchema):\n              ValidationError(equality):\n                None does not equal 'foo'\n              ValidationError(type):\n                Type of None should be str, but is NoneType\n              ValidationError(Callable):\n                <lambda>(None) is not true\n        ")

class TestNoneOrAllSchema:

    @pytest.mark.parametrize(('data', 'expected'), [('foo', 'FOO'), ('bar', None)])
    def test_success(self, data, expected):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.Schema(re.compile('foo'), validate.none_or_all(validate.get(0), validate.transform(str.upper))), data) == expected

    def test_failure(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.none_or_all(str, int), 'foo')
        assert_validationerror(cm.value, "\n            ValidationError(NoneOrAllSchema):\n              ValidationError(type):\n                Type of 'foo' should be int, but is str\n        ")

class TestListSchema:

    def test_success(self):
        if False:
            for i in range(10):
                print('nop')
        data = [1, 3.14, 'foo']
        result = validate.validate(validate.list(int, float, 'foo'), data)
        assert result is not data
        assert result == [1, 3.14, 'foo']
        assert type(result) is type(data)
        assert len(result) == len(data)

    @pytest.mark.parametrize('data', [[1, 'foo'], [1.2, 'foo'], [1, 'bar'], [1.2, 'bar']])
    def test_success_subschemas(self, data):
        if False:
            print('Hello World!')
        schema = validate.list(validate.any(int, float), validate.all(validate.any('foo', 'bar'), validate.transform(str.upper)))
        result = validate.validate(schema, data)
        assert result is not data
        assert result[0] is data[0]
        assert result[1] is not data[1]
        assert result[1].isupper()

    def test_failure(self):
        if False:
            for i in range(10):
                print('nop')
        data = [1, 3.14, 'foo']
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.list('foo', int, float), data)
        assert_validationerror(cm.value, "\n            ValidationError(ListSchema):\n              ValidationError(equality):\n                1 does not equal 'foo'\n              ValidationError(type):\n                Type of 3.14 should be int, but is float\n              ValidationError(type):\n                Type of 'foo' should be float, but is str\n        ")

    def test_failure_type(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.list(), {})
        assert_validationerror(cm.value, '\n            ValidationError(ListSchema):\n              Type of {} should be list, but is dict\n        ')

    def test_failure_length(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.list('foo', 'bar', 'baz'), ['foo', 'bar'])
        assert_validationerror(cm.value, '\n            ValidationError(ListSchema):\n              Length of list (2) does not match expectation (3)\n        ')

class TestRegexSchema:

    @pytest.mark.parametrize(('pattern', 'data', 'expected'), [('\\s(?P<bar>\\S+)\\s', 'foo bar baz', {'bar': 'bar'}), (b'\\s(?P<bar>\\S+)\\s', b'foo bar baz', {'bar': b'bar'})])
    def test_success(self, pattern, data, expected):
        if False:
            while True:
                i = 10
        result = validate.validate(validate.regex(re.compile(pattern)), data)
        assert type(result) is re.Match
        assert result.groupdict() == expected

    def test_findall(self):
        if False:
            print('Hello World!')
        assert validate.validate(validate.regex(re.compile('\\w+'), 'findall'), 'foo bar baz') == ['foo', 'bar', 'baz']

    def test_split(self):
        if False:
            for i in range(10):
                print('nop')
        assert validate.validate(validate.regex(re.compile('\\s+'), 'split'), 'foo bar baz') == ['foo', 'bar', 'baz']

    def test_failure(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.regex(re.compile('foo')), 'bar')
        assert_validationerror(cm.value, "\n            ValidationError(RegexSchema):\n              Pattern 'foo' did not match 'bar'\n        ")

    def test_failure_type(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.regex(re.compile('foo')), b'foo')
        assert_validationerror(cm.value, '\n            ValidationError(RegexSchema):\n              cannot use a string pattern on a bytes-like object\n        ')

    def test_failure_schema(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.regex(re.compile('foo')), 123)
        assert_validationerror(cm.value, '\n            ValidationError(RegexSchema):\n              Type of 123 should be str or bytes, but is int\n        ')

class TestTransformSchema:

    def test_success(self):
        if False:
            i = 10
            return i + 15

        def callback(string: str, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return string.format(*args, **kwargs)
        assert validate.validate(validate.transform(callback, 'foo', 'bar', baz='qux'), '{0} {1} {baz}') == 'foo bar qux'

    def test_failure_signature(self):
        if False:
            print('Hello World!')

        def callback():
            if False:
                i = 10
                return i + 15
            pass
        with pytest.raises(TypeError) as cm:
            validate.validate(validate.transform(callback), 'foo')
        assert str(cm.value).endswith('takes 0 positional arguments but 1 was given')

    def test_failure_schema(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.transform('not a callable'), 'foo')
        assert_validationerror(cm.value, "\n            ValidationError(type):\n              Type of 'not a callable' should be Callable, but is str\n        ")

class TestGetItemSchema:

    class Container:

        def __init__(self, exception):
            if False:
                return 10
            self.exception = exception

        def __getitem__(self, item):
            if False:
                for i in range(10):
                    print('nop')
            raise self.exception

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.__class__.__name__

    @pytest.mark.parametrize('obj', [{'foo': 'bar'}, Element('elem', {'foo': 'bar'}), re.match('(?P<foo>.+)', 'bar')], ids=['dict', 'lxml.etree.Element', 're.Match'])
    def test_simple(self, obj):
        if False:
            return 10
        assert validate.validate(validate.get('foo'), obj) == 'bar'

    @pytest.mark.parametrize('exception', [KeyError, IndexError])
    def test_getitem_no_default(self, exception):
        if False:
            i = 10
            return i + 15
        container = self.Container(exception())
        assert validate.validate(validate.get('foo'), container) is None

    @pytest.mark.parametrize('exception', [KeyError, IndexError])
    def test_getitem_default(self, exception):
        if False:
            print('Hello World!')
        container = self.Container(exception('failure'))
        assert validate.validate(validate.get('foo', default='default'), container) == 'default'

    @pytest.mark.parametrize('exception', [TypeError, AttributeError])
    def test_getitem_error(self, exception):
        if False:
            while True:
                i = 10
        container = self.Container(exception('failure'))
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.get('foo', default='default'), container)
        assert_validationerror(cm.value, "\n            ValidationError(GetItemSchema):\n              Could not get key 'foo' from object Container\n              Context:\n                failure\n        ")

    def test_nested(self):
        if False:
            i = 10
            return i + 15
        dictionary = {'foo': {'bar': {'baz': 'qux'}}}
        assert validate.validate(validate.get(('foo', 'bar', 'baz')), dictionary) == 'qux'

    def test_nested_default(self):
        if False:
            i = 10
            return i + 15
        dictionary = {'foo': {'bar': {'baz': 'qux'}}}
        assert validate.validate(validate.get(('foo', 'bar', 'qux'), default='default'), dictionary) == 'default'

    def test_nested_failure(self):
        if False:
            i = 10
            return i + 15
        dictionary = {'foo': {'bar': {'baz': 'qux'}}}
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.get(('foo', 'qux', 'baz'), default='default'), dictionary)
        assert_validationerror(cm.value, "\n            ValidationError(GetItemSchema):\n              Item 'qux' was not found in object {'bar': {'baz': 'qux'}}\n        ")

    def test_strict(self):
        if False:
            print('Hello World!')
        dictionary = {('foo', 'bar', 'baz'): 'foo-bar-baz', 'foo': {'bar': {'baz': 'qux'}}}
        assert validate.validate(validate.get(('foo', 'bar', 'baz'), strict=True), dictionary) == 'foo-bar-baz'

class TestAttrSchema:

    class Subject:
        foo = 1
        bar = 2

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return self.__class__.__name__

    @pytest.fixture()
    def obj(self):
        if False:
            return 10
        obj1 = self.Subject()
        obj2 = self.Subject()
        obj1.bar = obj2
        return obj1

    def test_success(self, obj):
        if False:
            return 10
        schema = validate.attr({'foo': validate.transform(lambda num: num + 1)})
        newobj = validate.validate(schema, obj)
        assert obj.foo == 1
        assert newobj is not obj
        assert newobj.foo == 2
        assert newobj.bar is obj.bar

    def test_failure_missing(self, obj):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.attr({'missing': int}), obj)
        assert_validationerror(cm.value, "\n            ValidationError(AttrSchema):\n              Attribute 'missing' not found on object Subject\n        ")

    def test_failure_subschema(self, obj):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.attr({'foo': str}), obj)
        assert_validationerror(cm.value, "\n            ValidationError(AttrSchema):\n              Could not validate attribute 'foo'\n              Context(type):\n                Type of 1 should be str, but is int\n        ")

class TestXmlElementSchema:
    upper = validate.transform(str.upper)

    @pytest.fixture()
    def element(self):
        if False:
            return 10
        childA = Element('childA', {'a': '1'})
        childB = Element('childB', {'b': '2'})
        childC = Element('childC')
        childA.text = 'childAtext'
        childA.tail = 'childAtail'
        childB.text = 'childBtext'
        childB.tail = 'childBtail'
        childB.append(childC)
        parent = Element('parent', {'attrkey1': 'attrval1', 'attrkey2': 'attrval2'})
        parent.text = 'parenttext'
        parent.tail = 'parenttail'
        parent.append(childA)
        parent.append(childB)
        return parent

    @pytest.mark.parametrize(('schema', 'expected'), [(validate.xml_element(), '<parent attrkey1="attrval1" attrkey2="attrval2">' + 'parenttext' + '<childA a="1">childAtext</childA>' + 'childAtail' + '<childB b="2">childBtext<childC/></childB>' + 'childBtail' + '</parent>' + 'parenttail'), (validate.xml_element(tag=upper, attrib={upper: upper}, text=upper, tail=upper), '<PARENT ATTRKEY1="ATTRVAL1" ATTRKEY2="ATTRVAL2">' + 'PARENTTEXT' + '<childA a="1">childAtext</childA>' + 'childAtail' + '<childB b="2">childBtext<childC/></childB>' + 'childBtail' + '</PARENT>' + 'PARENTTAIL')], ids=['empty', 'subschemas'])
    def test_success(self, element, schema, expected):
        if False:
            i = 10
            return i + 15
        newelement = validate.validate(schema, element)
        assert etree_tostring(newelement).decode('utf-8') == expected
        assert newelement is not element
        assert newelement[0] is not element[0]
        assert newelement[1] is not element[1]
        assert newelement[1][0] is not element[1][0]

    @pytest.mark.parametrize(('schema', 'error'), [(validate.xml_element(tag='invalid'), "\n                    ValidationError(XmlElementSchema):\n                      Unable to validate XML tag\n                      Context(equality):\n                        'parent' does not equal 'invalid'\n                "), (validate.xml_element(attrib={'invalid': 'invalid'}), "\n                    ValidationError(XmlElementSchema):\n                      Unable to validate XML attributes\n                      Context(dict):\n                        Key 'invalid' not found in {'attrkey1': 'attrval1', 'attrkey2': 'attrval2'}\n                "), (validate.xml_element(text='invalid'), "\n                    ValidationError(XmlElementSchema):\n                      Unable to validate XML text\n                      Context(equality):\n                        'parenttext' does not equal 'invalid'\n                "), (validate.xml_element(tail='invalid'), "\n                    ValidationError(XmlElementSchema):\n                      Unable to validate XML tail\n                      Context(equality):\n                        'parenttail' does not equal 'invalid'\n                ")], ids=['tag', 'attrib', 'text', 'tail'])
    def test_failure(self, element, schema, error):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(schema, element)
        assert_validationerror(cm.value, error)

    def test_failure_schema(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_element(), 'not-an-element')
        assert_validationerror(cm.value, "\n            ValidationError(Callable):\n              iselement('not-an-element') is not true\n        ")

class TestUnionGetSchema:

    def test_simple(self):
        if False:
            return 10
        assert validate.validate(validate.union_get('foo', 'bar'), {'foo': 1, 'bar': 2}) == (1, 2)

    def test_sequence_type(self):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.union_get('foo', 'bar', seq=list), {'foo': 1, 'bar': 2}) == [1, 2]

    def test_nested(self):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.union_get(('foo', 'bar'), ('baz', 'qux')), {'foo': {'bar': 1}, 'baz': {'qux': 2}}) == (1, 2)

class TestUnionSchema:
    upper = validate.transform(str.upper)

    def test_dict_success(self):
        if False:
            return 10
        schema = validate.union({'foo': str, 'bar': self.upper, validate.optional('baz'): int})
        assert validate.validate(schema, 'value') == {'foo': 'value', 'bar': 'VALUE'}

    def test_dict_failure(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.union({'foo': int}), 'value')
        assert_validationerror(cm.value, "\n            ValidationError(UnionSchema):\n              Could not validate union\n              Context(dict):\n                Unable to validate union 'foo'\n                Context(type):\n                  Type of 'value' should be int, but is str\n        ")

    @pytest.mark.parametrize(('schema', 'expected'), [(validate.union([str, upper]), ['value', 'VALUE']), (validate.union((str, upper)), ('value', 'VALUE')), (validate.union({str, upper}), {'value', 'VALUE'}), (validate.union(frozenset((str, upper))), frozenset(('value', 'VALUE')))], ids=['list', 'tuple', 'set', 'frozenset'])
    def test_sequence(self, schema, expected):
        if False:
            print('Hello World!')
        result = validate.validate(schema, 'value')
        assert result == expected

    def test_failure_schema(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.union(None), None)
        assert_validationerror(cm.value, '\n            ValidationError(UnionSchema):\n              Could not validate union\n              Context:\n                Invalid union type: NoneType\n        ')

class TestLengthValidator:

    @pytest.mark.parametrize(('args', 'value'), [((3,), 'abc'), ((3,), [1, 2, 3]), ((3,), 'abcd'), ((3,), [1, 2, 3, 4]), ((3, 'lt'), 'ab'), ((3, 'lt'), [1, 2]), ((3, 'le'), 'ab'), ((3, 'le'), [1, 2]), ((3, 'le'), 'abc'), ((3, 'le'), [1, 2, 3]), ((3, 'eq'), 'abc'), ((3, 'eq'), [1, 2, 3]), ((3, 'ge'), 'abc'), ((3, 'ge'), [1, 2, 3]), ((3, 'ge'), 'abcd'), ((3, 'ge'), [1, 2, 3, 4]), ((3, 'gt'), 'abcd'), ((3, 'gt'), [1, 2, 3, 4])])
    def test_success(self, args, value):
        if False:
            for i in range(10):
                print('nop')
        assert validate.validate(validate.length(*args), value) == value

    @pytest.mark.parametrize(('args', 'value', 'error'), [((3,), 'ab', 'Length must be >=3, but value is 2'), ((3,), [1, 2], 'Length must be >=3, but value is 2'), ((3, 'lt'), 'abc', 'Length must be <3, but value is 3'), ((3, 'lt'), [1, 2, 3], 'Length must be <3, but value is 3'), ((3, 'le'), 'abcd', 'Length must be <=3, but value is 4'), ((3, 'le'), [1, 2, 3, 4], 'Length must be <=3, but value is 4'), ((3, 'eq'), 'ab', 'Length must be ==3, but value is 2'), ((3, 'eq'), [1, 2], 'Length must be ==3, but value is 2'), ((3, 'ge'), 'ab', 'Length must be >=3, but value is 2'), ((3, 'ge'), [1, 2], 'Length must be >=3, but value is 2'), ((3, 'gt'), 'abc', 'Length must be >3, but value is 3'), ((3, 'gt'), [1, 2, 3], 'Length must be >3, but value is 3')])
    def test_failure(self, args, value, error):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.length(*args), value)
        assert_validationerror(cm.value, f'\n            ValidationError(length):\n              {error}\n        ')

class TestStartsWithValidator:

    def test_success(self):
        if False:
            for i in range(10):
                print('nop')
        assert validate.validate(validate.startswith('foo'), 'foo bar baz')

    def test_failure(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.startswith('invalid'), 'foo bar baz')
        assert_validationerror(cm.value, "\n            ValidationError(startswith):\n              'foo bar baz' does not start with 'invalid'\n        ")

    def test_failure_schema(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.startswith('invalid'), 1)
        assert_validationerror(cm.value, '\n            ValidationError(type):\n              Type of 1 should be str, but is int\n        ')

class TestEndsWithValidator:

    def test_success(self):
        if False:
            print('Hello World!')
        assert validate.validate(validate.endswith('baz'), 'foo bar baz')

    def test_failure(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.endswith('invalid'), 'foo bar baz')
        assert_validationerror(cm.value, "\n            ValidationError(endswith):\n              'foo bar baz' does not end with 'invalid'\n        ")

    def test_failure_schema(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.endswith('invalid'), 1)
        assert_validationerror(cm.value, '\n            ValidationError(type):\n              Type of 1 should be str, but is int\n        ')

class TestContainsValidator:

    def test_success(self):
        if False:
            i = 10
            return i + 15
        assert validate.validate(validate.contains('bar'), 'foo bar baz')

    def test_failure(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.contains('invalid'), 'foo bar baz')
        assert_validationerror(cm.value, "\n            ValidationError(contains):\n              'foo bar baz' does not contain 'invalid'\n        ")

    def test_failure_schema(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.contains('invalid'), 1)
        assert_validationerror(cm.value, '\n            ValidationError(type):\n              Type of 1 should be str, but is int\n        ')

class TestUrlValidator:
    url = 'https://user:pass@sub.host.tld:1234/path.m3u8?query#fragment'

    @pytest.mark.parametrize('params', [dict(scheme='http'), dict(scheme='https'), dict(netloc='user:pass@sub.host.tld:1234', username='user', password='pass', hostname='sub.host.tld', port=1234), dict(path=validate.endswith('.m3u8'))], ids=['implicit https', 'explicit https', 'multiple attributes', 'subschemas'])
    def test_success(self, params):
        if False:
            i = 10
            return i + 15
        assert validate.validate(validate.url(**params), self.url)

    def test_failure_valid_url(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.url(), 'foo')
        assert_validationerror(cm.value, "\n            ValidationError(url):\n              'foo' is not a valid URL\n        ")

    def test_failure_url_attribute(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.url(invalid=str), self.url)
        assert_validationerror(cm.value, "\n            ValidationError(url):\n              Invalid URL attribute 'invalid'\n        ")

    def test_failure_subschema(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.url(hostname='invalid'), self.url)
        assert_validationerror(cm.value, "\n            ValidationError(url):\n              Unable to validate URL attribute 'hostname'\n              Context(equality):\n                'sub.host.tld' does not equal 'invalid'\n        ")

    def test_failure_schema(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.url(), 1)
        assert_validationerror(cm.value, '\n            ValidationError(type):\n              Type of 1 should be str, but is int\n        ')

class TestGetAttrValidator:

    @pytest.fixture(scope='class')
    def subject(self):
        if False:
            print('Hello World!')

        class Subject:
            foo = 1
        return Subject()

    def test_simple(self, subject):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.getattr('foo'), subject) == 1

    def test_default(self, subject):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.getattr('bar', 2), subject) == 2

    def test_no_default(self, subject):
        if False:
            print('Hello World!')
        assert validate.validate(validate.getattr('bar'), subject) is None
        assert validate.validate(validate.getattr('baz'), None) is None

class TestHasAttrValidator:

    class Subject:
        foo = 1

        def __repr__(self):
            if False:
                while True:
                    i = 10
            return self.__class__.__name__

    def test_success(self):
        if False:
            return 10
        assert validate.validate(validate.hasattr('foo'), self.Subject())

    def test_failure(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.hasattr('bar'), self.Subject())
        assert_validationerror(cm.value, '\n            ValidationError(Callable):\n              getter(Subject) is not true\n        ')

class TestFilterValidator:

    def test_dict(self):
        if False:
            while True:
                i = 10
        schema = validate.filter(lambda k, v: k < 2 and v > 0)
        value = {0: 0, 1: 1, 2: 0, 3: 1}
        assert validate.validate(schema, value) == {1: 1}

    def test_sequence(self):
        if False:
            while True:
                i = 10
        schema = validate.filter(lambda k: k < 2)
        value = (0, 1, 2, 3)
        assert validate.validate(schema, value) == (0, 1)

class TestMapValidator:

    def test_dict(self):
        if False:
            while True:
                i = 10
        schema = validate.map(lambda k, v: (k + 1, v + 1))
        value = {0: 0, 1: 1, 2: 0, 3: 1}
        assert validate.validate(schema, value) == {1: 1, 2: 2, 3: 1, 4: 2}

    def test_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        schema = validate.map(lambda k: k + 1)
        value = (0, 1, 2, 3)
        assert validate.validate(schema, value) == (1, 2, 3, 4)

class TestXmlFindValidator:

    def test_success(self):
        if False:
            return 10
        element = Element('foo')
        assert validate.validate(validate.xml_find('.'), element) is element

    def test_namespaces(self):
        if False:
            for i in range(10):
                print('nop')
        root = Element('root')
        child = Element('{http://a}foo')
        root.append(child)
        assert validate.validate(validate.xml_find('./a:foo', namespaces={'a': 'http://a'}), root) is child

    def test_failure_no_element(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_find('*'), Element('foo'))
        assert_validationerror(cm.value, "\n            ValidationError(xml_find):\n              ElementPath query '*' did not return an element\n        ")

    def test_failure_not_found(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_find('invalid'), Element('foo'))
        assert_validationerror(cm.value, "\n            ValidationError(xml_find):\n              ElementPath query 'invalid' did not return an element\n        ")

    def test_failure_schema(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_find('.'), 'not-an-element')
        assert_validationerror(cm.value, "\n            ValidationError(Callable):\n              iselement('not-an-element') is not true\n        ")

    def test_failure_syntax(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_find('['), Element('foo'))
        assert_validationerror(cm.value, "\n            ValidationError(xml_find):\n              ElementPath syntax error: '['\n              Context:\n                invalid path\n        ")

class TestXmlFindallValidator:

    @pytest.fixture(scope='class')
    def element(self):
        if False:
            while True:
                i = 10
        element = Element('root')
        for child in (Element('foo'), Element('bar'), Element('baz')):
            element.append(child)
        return element

    def test_simple(self, element):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.xml_findall('*'), element) == [element[0], element[1], element[2]]

    def test_empty(self, element):
        if False:
            i = 10
            return i + 15
        assert validate.validate(validate.xml_findall('missing'), element) == []

    def test_namespaces(self):
        if False:
            for i in range(10):
                print('nop')
        root = Element('root')
        for child in (Element('{http://a}foo'), Element('{http://unknown}bar'), Element('{http://a}baz')):
            root.append(child)
        assert validate.validate(validate.xml_findall('./a:*', namespaces={'a': 'http://a'}), root) == [root[0], root[2]]

    def test_failure_schema(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_findall('*'), 'not-an-element')
        assert_validationerror(cm.value, "\n            ValidationError(Callable):\n              iselement('not-an-element') is not true\n        ")

class TestXmlFindtextValidator:

    def test_simple(self):
        if False:
            print('Hello World!')
        element = Element('foo')
        element.text = 'bar'
        assert validate.validate(validate.xml_findtext('.'), element) == 'bar'

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        element = Element('foo')
        assert validate.validate(validate.xml_findtext('.'), element) is None

    def test_namespaces(self):
        if False:
            i = 10
            return i + 15
        root = Element('root')
        child = Element('{http://a}foo')
        child.text = 'bar'
        root.append(child)
        assert validate.validate(validate.xml_findtext('./a:foo', namespaces={'a': 'http://a'}), root) == 'bar'

    def test_failure_schema(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_findtext('.'), 'not-an-element')
        assert_validationerror(cm.value, "\n            ValidationError(Callable):\n              iselement('not-an-element') is not true\n        ")

class TestXmlXpathValidator:

    @pytest.fixture(scope='class')
    def element(self):
        if False:
            while True:
                i = 10
        element = Element('root')
        for child in (Element('foo'), Element('bar'), Element('baz')):
            child.text = child.tag.upper()
            element.append(child)
        return element

    def test_simple(self, element):
        if False:
            print('Hello World!')
        assert validate.validate(validate.xml_xpath('*'), element) == [element[0], element[1], element[2]]
        assert validate.validate(validate.xml_xpath('*/text()'), element) == ['FOO', 'BAR', 'BAZ']

    def test_empty(self, element):
        if False:
            for i in range(10):
                print('nop')
        assert validate.validate(validate.xml_xpath('invalid'), element) is None

    def test_other(self, element):
        if False:
            print('Hello World!')
        assert validate.validate(validate.xml_xpath('local-name(.)'), element) == 'root'

    def test_namespaces(self):
        if False:
            for i in range(10):
                print('nop')
        nsmap = {'a': 'http://a', 'b': 'http://b'}
        root = Element('root', nsmap=nsmap)
        for child in (Element('{http://a}child'), Element('{http://b}child')):
            root.append(child)
        assert validate.validate(validate.xml_xpath('./b:child', namespaces=nsmap), root)[0] is root[1]

    def test_extensions(self, element):
        if False:
            for i in range(10):
                print('nop')

        def foo(context, a, b):
            if False:
                while True:
                    i = 10
            return int(context.context_node.attrib.get('val')) + a + b
        element = Element('root', attrib={'val': '3'})
        assert validate.validate(validate.xml_xpath('foo(5, 7)', extensions={(None, 'foo'): foo}), element) == 15.0

    def test_smart_strings(self, element):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.xml_xpath('*/text()'), element)[0].getparent().tag == 'foo'
        assert not hasattr(validate.validate(validate.xml_xpath('*/text()', smart_strings=False), element)[0], 'getparent')

    def test_variables(self, element):
        if False:
            print('Hello World!')
        assert validate.validate(validate.xml_xpath('*[local-name() = $name]/text()', name='foo'), element) == ['FOO']

    def test_failure_schema(self):
        if False:
            return 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_xpath('.'), 'not-an-element')
        assert_validationerror(cm.value, "\n            ValidationError(Callable):\n              iselement('not-an-element') is not true\n        ")

    def test_failure_evaluation(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_xpath('?'), Element('root'))
        assert_validationerror(cm.value, "\n            ValidationError(xml_xpath):\n              XPath evaluation error: '?'\n              Context:\n                Invalid expression\n        ")

class TestXmlXpathStringValidator:

    @pytest.fixture(scope='class')
    def element(self):
        if False:
            for i in range(10):
                print('nop')
        element = Element('root')
        for child in (Element('foo'), Element('bar'), Element('baz')):
            child.text = child.tag.upper()
            element.append(child)
        return element

    def test_simple(self, element):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.xml_xpath_string('./foo/text()'), element) == 'FOO'

    def test_empty(self, element):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.xml_xpath_string('./text()'), element) is None

    def test_smart_strings(self, element):
        if False:
            i = 10
            return i + 15
        assert not hasattr(validate.validate(validate.xml_xpath_string('./foo/text()'), element)[0], 'getparent')

    def test_failure_schema(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.xml_xpath_string('.'), 'not-an-element')
        assert_validationerror(cm.value, "\n            ValidationError(Callable):\n              iselement('not-an-element') is not true\n        ")

class TestParseJsonValidator:

    def test_success(self):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.parse_json(), '{"a": ["b", true, false, null, 1, 2.3]}') == {'a': ['b', True, False, None, 1, 2.3]}

    def test_failure(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.parse_json(), 'invalid')
        assert_validationerror(cm.value, "\n            ValidationError:\n              Unable to parse JSON: Expecting value: line 1 column 1 (char 0) ('invalid')\n        ")

class TestParseHtmlValidator:

    def test_success(self):
        if False:
            print('Hello World!')
        assert validate.validate(validate.parse_html(), '<!DOCTYPE html><body>&quot;perfectly&quot;<a>valid<div>HTML').tag == 'html'

    def test_failure(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.parse_html(), None)
        assert_validationerror(cm.value, '\n            ValidationError:\n              Unable to parse HTML: can only parse strings (None)\n        ')

class TestParseXmlValidator:

    def test_success(self):
        if False:
            while True:
                i = 10
        assert validate.validate(validate.parse_xml(), '<?xml version="1.0" encoding="utf-8"?><root></root>').tag == 'root'

    def test_failure(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.parse_xml(), None)
        assert_validationerror(cm.value, '\n            ValidationError:\n              Unable to parse XML: can only parse strings (None)\n        ')

class TestParseQsdValidator:

    def test_success(self):
        if False:
            return 10
        assert validate.validate(validate.parse_qsd(), 'foo=bar&foo=baz&qux=quux') == {'foo': 'baz', 'qux': 'quux'}

    def test_failure(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValidationError) as cm:
            validate.validate(validate.parse_qsd(), 123)
        assert_validationerror(cm.value, "\n            ValidationError:\n              Unable to parse query string: 'int' object has no attribute 'decode' (123)\n        ")

class TestValidationError:

    def test_subclass(self):
        if False:
            i = 10
            return i + 15
        assert issubclass(ValidationError, ValueError)

    def test_empty(self):
        if False:
            while True:
                i = 10
        assert str(ValidationError()) == 'ValidationError:'
        assert str(ValidationError('')) == 'ValidationError:'
        assert str(ValidationError(ValidationError())) == 'ValidationError:\n  ValidationError:'
        assert str(ValidationError(ValidationError(''))) == 'ValidationError:\n  ValidationError:'

    def test_single(self):
        if False:
            i = 10
            return i + 15
        assert str(ValidationError('foo')) == 'ValidationError:\n  foo'
        assert str(ValidationError(ValueError('bar'))) == 'ValidationError:\n  bar'

    def test_single_nested(self):
        if False:
            print('Hello World!')
        err = ValidationError(ValidationError('baz'))
        assert_validationerror(err, '\n            ValidationError:\n              ValidationError:\n                baz\n        ')

    def test_multiple_nested(self):
        if False:
            i = 10
            return i + 15
        err = ValidationError('a', ValidationError('b', 'c'), 'd', ValidationError('e'), 'f')
        assert_validationerror(err, '\n            ValidationError:\n              a\n              ValidationError:\n                b\n                c\n              d\n              ValidationError:\n                e\n              f\n        ')

    def test_context(self):
        if False:
            while True:
                i = 10
        errA = ValidationError('a')
        errB = ValidationError('b')
        errC = ValidationError('c')
        errA.__cause__ = errB
        errB.__cause__ = errC
        assert_validationerror(errA, '\n            ValidationError:\n              a\n              Context:\n                b\n                Context:\n                  c\n        ')

    def test_multiple_nested_context(self):
        if False:
            i = 10
            return i + 15
        errAB = ValidationError('a', 'b')
        errC = ValidationError('c')
        errDE = ValidationError('d', 'e')
        errF = ValidationError('f')
        errG = ValidationError('g')
        errHI = ValidationError('h', 'i')
        errCF = ValidationError(errC, errF)
        errAB.__cause__ = errCF
        errC.__cause__ = errDE
        errF.__cause__ = errG
        errCF.__cause__ = errHI
        assert_validationerror(errAB, '\n            ValidationError:\n              a\n              b\n              Context:\n                ValidationError:\n                  c\n                  Context:\n                    d\n                    e\n                ValidationError:\n                  f\n                  Context:\n                    g\n                Context:\n                  h\n                  i\n        ')

    def test_schema(self):
        if False:
            while True:
                i = 10
        err = ValidationError(ValidationError('foo', schema=dict), ValidationError('bar', schema='something'), schema=validate.any)
        assert_validationerror(err, '\n            ValidationError(AnySchema):\n              ValidationError(dict):\n                foo\n              ValidationError(something):\n                bar\n        ')

    def test_recursion(self):
        if False:
            i = 10
            return i + 15
        err1 = ValidationError('foo')
        err2 = ValidationError('bar')
        err2.__cause__ = err1
        err1.__cause__ = err2
        assert_validationerror(err1, '\n            ValidationError:\n              foo\n              Context:\n                bar\n                Context:\n                  ...\n        ')

    def test_truncate(self):
        if False:
            i = 10
            return i + 15
        err = ValidationError('foo {foo} bar {bar} baz', foo='Some really long error message that exceeds the maximum error message length', bar=repr('Some really long error message that exceeds the maximum error message length'))
        assert_validationerror(err, "\n            ValidationError:\n              foo <Some really long error message that exceeds the maximum...> bar <'Some really long error message that exceeds the maximu...> baz\n        ")