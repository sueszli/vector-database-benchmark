from jsonschema import Draft4Validator
from jsonschema.exceptions import ValidationError
from helpers import unittest, in_parse
import luigi
import luigi.interface
import json
import mock
import collections
import pytest

class DictParameterTask(luigi.Task):
    param = luigi.DictParameter()

class DictParameterTest(unittest.TestCase):
    _dict = collections.OrderedDict([('username', 'me'), ('password', 'secret')])

    def test_parse(self):
        if False:
            for i in range(10):
                print('nop')
        d = luigi.DictParameter().parse(json.dumps(DictParameterTest._dict))
        self.assertEqual(d, DictParameterTest._dict)

    def test_serialize(self):
        if False:
            while True:
                i = 10
        d = luigi.DictParameter().serialize(DictParameterTest._dict)
        self.assertEqual(d, '{"username": "me", "password": "secret"}')

    def test_parse_and_serialize(self):
        if False:
            print('Hello World!')
        inputs = ['{"username": "me", "password": "secret"}', '{"password": "secret", "username": "me"}']
        for json_input in inputs:
            _dict = luigi.DictParameter().parse(json_input)
            self.assertEqual(json_input, luigi.DictParameter().serialize(_dict))

    def test_parse_interface(self):
        if False:
            return 10
        in_parse(['DictParameterTask', '--param', '{"username": "me", "password": "secret"}'], lambda task: self.assertEqual(task.param, DictParameterTest._dict))

    def test_serialize_task(self):
        if False:
            return 10
        t = DictParameterTask(DictParameterTest._dict)
        self.assertEqual(str(t), 'DictParameterTask(param={"username": "me", "password": "secret"})')

    def test_parse_invalid_input(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, lambda : luigi.DictParameter().parse('{"invalid"}'))

    def test_hash_normalize(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : hash(luigi.DictParameter().parse('{"a": {"b": []}}')))
        a = luigi.DictParameter().normalize({'a': [{'b': []}]})
        b = luigi.DictParameter().normalize({'a': [{'b': []}]})
        self.assertEqual(hash(a), hash(b))

    def test_schema(self):
        if False:
            return 10
        a = luigi.parameter.DictParameter(schema={'type': 'object', 'properties': {'an_int': {'type': 'integer'}, 'an_optional_str': {'type': 'string'}}, 'additionalProperties': False, 'required': ['an_int']})
        with pytest.raises(ValidationError, match="Additional properties are not allowed \\('INVALID_ATTRIBUTE' was unexpected\\)"):
            a.normalize({'INVALID_ATTRIBUTE': 0})
        with pytest.raises(ValidationError, match="'an_int' is a required property"):
            a.normalize({})
        a.normalize({'an_int': 1})
        a.normalize({'an_int': 1, 'an_optional_str': 'hello'})
        with pytest.raises(ValidationError, match="'999' is not of type 'integer'"):
            a.normalize({'an_int': '999'})
        with pytest.raises(ValidationError, match="999 is not of type 'string'"):
            a.normalize({'an_int': 1, 'an_optional_str': 999})
        b = luigi.DictParameter(schema={'type': 'object', 'patternProperties': {'.*': {'type': 'string', 'enum': ['web', 'staging']}}})
        b.normalize({'role': 'web', 'env': 'staging'})
        with pytest.raises(ValidationError, match="'UNKNOWN_VALUE' is not one of \\['web', 'staging'\\]"):
            b.normalize({'role': 'UNKNOWN_VALUE', 'env': 'staging'})
        with mock.patch('luigi.parameter._JSONSCHEMA_ENABLED', False):
            with pytest.warns(UserWarning, match="The 'jsonschema' package is not installed so the parameter can not be validated even though a schema is given."):
                luigi.ListParameter(schema={'type': 'object'})
        validator = Draft4Validator(schema={'type': 'object', 'patternProperties': {'.*': {'type': 'string', 'enum': ['web', 'staging']}}})
        c = luigi.DictParameter(schema=validator)
        c.normalize({'role': 'web', 'env': 'staging'})
        with pytest.raises(ValidationError, match="'UNKNOWN_VALUE' is not one of \\['web', 'staging'\\]"):
            c.normalize({'role': 'UNKNOWN_VALUE', 'env': 'staging'})
        frozen_data = luigi.freezing.recursively_freeze({'role': 'web', 'env': 'staging'})
        c.normalize(frozen_data)