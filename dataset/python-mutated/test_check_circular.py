from unittest import TestCase
import simplejson as json

def default_iterable(obj):
    if False:
        while True:
            i = 10
    return list(obj)

class TestCheckCircular(TestCase):

    def test_circular_dict(self):
        if False:
            print('Hello World!')
        dct = {}
        dct['a'] = dct
        self.assertRaises(ValueError, json.dumps, dct)

    def test_circular_list(self):
        if False:
            while True:
                i = 10
        lst = []
        lst.append(lst)
        self.assertRaises(ValueError, json.dumps, lst)

    def test_circular_composite(self):
        if False:
            print('Hello World!')
        dct2 = {}
        dct2['a'] = []
        dct2['a'].append(dct2)
        self.assertRaises(ValueError, json.dumps, dct2)

    def test_circular_default(self):
        if False:
            i = 10
            return i + 15
        json.dumps([set()], default=default_iterable)
        self.assertRaises(TypeError, json.dumps, [set()])

    def test_circular_off_default(self):
        if False:
            i = 10
            return i + 15
        json.dumps([set()], default=default_iterable, check_circular=False)
        self.assertRaises(TypeError, json.dumps, [set()], check_circular=False)