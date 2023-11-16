import unittest

class Dict(dict):

    def __init__(self, **kw):
        if False:
            for i in range(10):
                print('nop')
        super(Dict, self).__init__(**kw)

    def __getattr__(self, key):
        if False:
            return 10
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        if False:
            i = 10
            return i + 15
        self[key] = value

class TestDict(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        print('setUp...')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        print('tearDown...')

    def test_init(self):
        if False:
            i = 10
            return i + 15
        d = Dict(a=1, b='test')
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 'test')
        self.assertTrue(isinstance(d, dict))

    def test_key(self):
        if False:
            return 10
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key, 'value')

    def test_attr(self):
        if False:
            for i in range(10):
                print('nop')
        d = Dict()
        d.key = 'value'
        self.assertTrue('key' in d)
        self.assertEqual(d['key'], 'value')

    def test_keyerror(self):
        if False:
            for i in range(10):
                print('nop')
        d = Dict()
        with self.assertRaises(KeyError):
            value = d['empty']

    def test_attrerror(self):
        if False:
            return 10
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty
if __name__ == '__main__':
    unittest.main()