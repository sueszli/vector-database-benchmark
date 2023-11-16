from unittest import TestCase
import simplejson as json

class TestBigintAsString(TestCase):
    values = [(200, 200), ((1 << 53) - 1, 9007199254740991), (1 << 53, '9007199254740992'), ((1 << 53) + 1, '9007199254740993'), (-100, -100), (-1 << 53, '-9007199254740992'), ((-1 << 53) - 1, '-9007199254740993'), ((-1 << 53) + 1, -9007199254740991)]
    options = ({'bigint_as_string': True}, {'int_as_string_bitcount': 53})

    def test_ints(self):
        if False:
            return 10
        for opts in self.options:
            for (val, expect) in self.values:
                self.assertEqual(val, json.loads(json.dumps(val)))
                self.assertEqual(expect, json.loads(json.dumps(val, **opts)))

    def test_lists(self):
        if False:
            return 10
        for opts in self.options:
            for (val, expect) in self.values:
                val = [val, val]
                expect = [expect, expect]
                self.assertEqual(val, json.loads(json.dumps(val)))
                self.assertEqual(expect, json.loads(json.dumps(val, **opts)))

    def test_dicts(self):
        if False:
            print('Hello World!')
        for opts in self.options:
            for (val, expect) in self.values:
                val = {'k': val}
                expect = {'k': expect}
                self.assertEqual(val, json.loads(json.dumps(val)))
                self.assertEqual(expect, json.loads(json.dumps(val, **opts)))

    def test_dict_keys(self):
        if False:
            print('Hello World!')
        for opts in self.options:
            for (val, _) in self.values:
                expect = {str(val): 'value'}
                val = {val: 'value'}
                self.assertEqual(expect, json.loads(json.dumps(val)))
                self.assertEqual(expect, json.loads(json.dumps(val, **opts)))