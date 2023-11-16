import json
import re
import unittest
from datetime import datetime
from coalib.output.JSONEncoder import create_json_encoder

class TestClass1(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.a = 0

class TestClass2(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.a = 0
        self.b = TestClass1()

class TestClass3(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = 0
        self.b = TestClass1()

    @staticmethod
    def __getitem__(key):
        if False:
            i = 10
            return i + 15
        return 'val'

    @staticmethod
    def keys():
        if False:
            i = 10
            return i + 15
        return ['key']

class PropertiedClass(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._a = 5

    @property
    def prop(self):
        if False:
            i = 10
            return i + 15
        return self._a

class JSONAbleClass(object):

    @staticmethod
    def __json__():
        if False:
            for i in range(10):
                print('nop')
        return ['dont', 'panic']

class JSONEncoderTest(unittest.TestCase):
    JSONEncoder = create_json_encoder(use_relpath=True)
    kw = {'cls': JSONEncoder, 'sort_keys': True}

    def test_builtins(self):
        if False:
            print('Hello World!')
        self.assertEqual('"test"', json.dumps('test', **self.kw))
        self.assertEqual('1', json.dumps(1, **self.kw))
        self.assertEqual('true', json.dumps(True, **self.kw))
        self.assertEqual('null', json.dumps(None, **self.kw))

    def test_iter(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('[0, 1]', json.dumps([0, 1], **self.kw))
        self.assertEqual('[0, 1]', json.dumps((0, 1), **self.kw))
        self.assertEqual('[0, 1]', json.dumps(range(2), **self.kw))

    def test_dict(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual('{"0": 1}', json.dumps({0: 1}, **self.kw))
        self.assertEqual('{"0": 1}', json.dumps({'0': 1}, **self.kw))
        self.assertEqual('{"0": "1"}', json.dumps({'0': '1'}, **self.kw))

    def test_time(self):
        if False:
            print('Hello World!')
        tf = datetime.today()
        self.assertEqual('"' + tf.isoformat() + '"', json.dumps(tf, **self.kw))

    def test_re_object(self):
        if False:
            for i in range(10):
                print('nop')
        uut = re.compile('x')
        self.assertEqual('"' + uut.pattern + '"', json.dumps(uut, **self.kw))

    def test_class1(self):
        if False:
            i = 10
            return i + 15
        tc1 = TestClass1()
        self.assertEqual('{"a": 0}', json.dumps(tc1, **self.kw))
        self.assertEqual('[{"a": 0}]', json.dumps([tc1], **self.kw))
        self.assertEqual('{"0": {"a": 0}}', json.dumps({0: tc1}, **self.kw))

    def test_class2(self):
        if False:
            return 10
        tc2 = TestClass2()
        self.assertEqual('{"a": 0, "b": {"a": 0}}', json.dumps(tc2, **self.kw))

    def test_class3(self):
        if False:
            while True:
                i = 10
        tc3 = TestClass3()
        self.assertEqual('{"key": "val"}', json.dumps(tc3, **self.kw))

    def test_propertied_class(self):
        if False:
            print('Hello World!')
        uut = PropertiedClass()
        self.assertEqual('{"prop": 5}', json.dumps(uut, **self.kw))

    def test_jsonable_class(self):
        if False:
            return 10
        uut = JSONAbleClass()
        self.assertEqual('["dont", "panic"]', json.dumps(uut, **self.kw))

    def test_type_error(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            json.dumps(1j, **self.kw)