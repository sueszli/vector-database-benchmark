from jsonschema import Draft4Validator
from jsonschema.exceptions import ValidationError
from helpers import unittest, in_parse
import luigi
import json
import mock
import pytest

class ListParameterTask(luigi.Task):
    param = luigi.ListParameter()

class ListParameterTest(unittest.TestCase):
    _list = [1, 'one', True]

    def test_parse(self):
        if False:
            i = 10
            return i + 15
        d = luigi.ListParameter().parse(json.dumps(ListParameterTest._list))
        self.assertEqual(d, ListParameterTest._list)

    def test_serialize(self):
        if False:
            print('Hello World!')
        d = luigi.ListParameter().serialize(ListParameterTest._list)
        self.assertEqual(d, '[1, "one", true]')

    def test_list_serialize_parse(self):
        if False:
            while True:
                i = 10
        a = luigi.ListParameter()
        b_list = [1, 2, 3]
        self.assertEqual(b_list, a.parse(a.serialize(b_list)))

    def test_parse_interface(self):
        if False:
            for i in range(10):
                print('nop')
        in_parse(['ListParameterTask', '--param', '[1, "one", true]'], lambda task: self.assertEqual(task.param, tuple(ListParameterTest._list)))

    def test_serialize_task(self):
        if False:
            i = 10
            return i + 15
        t = ListParameterTask(ListParameterTest._list)
        self.assertEqual(str(t), 'ListParameterTask(param=[1, "one", true])')

    def test_parse_invalid_input(self):
        if False:
            return 10
        self.assertRaises(ValueError, lambda : luigi.ListParameter().parse('{"invalid"}'))

    def test_hash_normalize(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, lambda : hash(luigi.ListParameter().parse('"NOT A LIST"')))
        a = luigi.ListParameter().normalize([0])
        b = luigi.ListParameter().normalize([0])
        self.assertEqual(hash(a), hash(b))

    def test_schema(self):
        if False:
            i = 10
            return i + 15
        a = luigi.ListParameter(schema={'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 10}, 'minItems': 1})
        with pytest.raises(ValidationError, match="'INVALID_ATTRIBUTE' is not of type 'number'"):
            a.normalize(['INVALID_ATTRIBUTE'])
        with pytest.raises(ValidationError, match='\\[\\] is too short'):
            a.normalize([])
        valid_list = [1, 2, 3]
        a.normalize(valid_list)
        invalid_list_type = ['NOT AN INT']
        invalid_list_value = [-999, 999]
        with pytest.raises(ValidationError, match="'NOT AN INT' is not of type 'number'"):
            a.normalize(invalid_list_type)
        with pytest.raises(ValidationError, match='-999 is less than the minimum of 0'):
            a.normalize(invalid_list_value)
        with mock.patch('luigi.parameter._JSONSCHEMA_ENABLED', False):
            with pytest.warns(UserWarning, match="The 'jsonschema' package is not installed so the parameter can not be validated even though a schema is given."):
                luigi.ListParameter(schema={'type': 'array', 'items': {'type': 'number'}})
        validator = Draft4Validator(schema={'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 10}, 'minItems': 1})
        c = luigi.DictParameter(schema=validator)
        c.normalize(valid_list)
        with pytest.raises(ValidationError, match="'INVALID_ATTRIBUTE' is not of type 'number'"):
            c.normalize(['INVALID_ATTRIBUTE'])
        frozen_data = luigi.freezing.recursively_freeze(valid_list)
        c.normalize(frozen_data)