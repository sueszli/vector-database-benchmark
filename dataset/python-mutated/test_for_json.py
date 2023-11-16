import unittest
import simplejson as json

class ForJson(object):

    def for_json(self):
        if False:
            print('Hello World!')
        return {'for_json': 1}

class NestedForJson(object):

    def for_json(self):
        if False:
            i = 10
            return i + 15
        return {'nested': ForJson()}

class ForJsonList(object):

    def for_json(self):
        if False:
            for i in range(10):
                print('nop')
        return ['list']

class DictForJson(dict):

    def for_json(self):
        if False:
            i = 10
            return i + 15
        return {'alpha': 1}

class ListForJson(list):

    def for_json(self):
        if False:
            print('Hello World!')
        return ['list']

class TestForJson(unittest.TestCase):

    def assertRoundTrip(self, obj, other, for_json=True):
        if False:
            for i in range(10):
                print('nop')
        if for_json is None:
            s = json.dumps(obj)
        else:
            s = json.dumps(obj, for_json=for_json)
        self.assertEqual(json.loads(s), other)

    def test_for_json_encodes_stand_alone_object(self):
        if False:
            return 10
        self.assertRoundTrip(ForJson(), ForJson().for_json())

    def test_for_json_encodes_object_nested_in_dict(self):
        if False:
            print('Hello World!')
        self.assertRoundTrip({'hooray': ForJson()}, {'hooray': ForJson().for_json()})

    def test_for_json_encodes_object_nested_in_list_within_dict(self):
        if False:
            print('Hello World!')
        self.assertRoundTrip({'list': [0, ForJson(), 2, 3]}, {'list': [0, ForJson().for_json(), 2, 3]})

    def test_for_json_encodes_object_nested_within_object(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRoundTrip(NestedForJson(), {'nested': {'for_json': 1}})

    def test_for_json_encodes_list(self):
        if False:
            print('Hello World!')
        self.assertRoundTrip(ForJsonList(), ForJsonList().for_json())

    def test_for_json_encodes_list_within_object(self):
        if False:
            print('Hello World!')
        self.assertRoundTrip({'nested': ForJsonList()}, {'nested': ForJsonList().for_json()})

    def test_for_json_encodes_dict_subclass(self):
        if False:
            i = 10
            return i + 15
        self.assertRoundTrip(DictForJson(a=1), DictForJson(a=1).for_json())

    def test_for_json_encodes_list_subclass(self):
        if False:
            i = 10
            return i + 15
        self.assertRoundTrip(ListForJson(['l']), ListForJson(['l']).for_json())

    def test_for_json_ignored_if_not_true_with_dict_subclass(self):
        if False:
            while True:
                i = 10
        for for_json in (None, False):
            self.assertRoundTrip(DictForJson(a=1), {'a': 1}, for_json=for_json)

    def test_for_json_ignored_if_not_true_with_list_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        for for_json in (None, False):
            self.assertRoundTrip(ListForJson(['l']), ['l'], for_json=for_json)

    def test_raises_typeerror_if_for_json_not_true_with_object(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, json.dumps, ForJson())
        self.assertRaises(TypeError, json.dumps, ForJson(), for_json=False)