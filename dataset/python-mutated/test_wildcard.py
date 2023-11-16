"""Some tests for the wildcard utilities."""
import unittest
from IPython.utils import wildcard

class obj_t(object):
    pass
root = obj_t()
l = ['arna', 'abel', 'ABEL', 'active', 'bob', 'bark', 'abbot']
q = ['kate', 'loop', 'arne', 'vito', 'lucifer', 'koppel']
for x in l:
    o = obj_t()
    setattr(root, x, o)
    for y in q:
        p = obj_t()
        setattr(o, y, p)
root._apan = obj_t()
root._apan.a = 10
root._apan._a = 20
root._apan.__a = 20
root.__anka = obj_t()
root.__anka.a = 10
root.__anka._a = 20
root.__anka.__a = 20
root._APAN = obj_t()
root._APAN.a = 10
root._APAN._a = 20
root._APAN.__a = 20
root.__ANKA = obj_t()
root.__ANKA.a = 10
root.__ANKA._a = 20
root.__ANKA.__a = 20

class Tests(unittest.TestCase):

    def test_case(self):
        if False:
            for i in range(10):
                print('nop')
        ns = root.__dict__
        tests = [('a*', ['abbot', 'abel', 'active', 'arna']), ('?b*.?o*', ['abbot.koppel', 'abbot.loop', 'abel.koppel', 'abel.loop']), ('_a*', []), ('_*anka', ['__anka']), ('_*a*', ['__anka'])]
        for (pat, res) in tests:
            res.sort()
            a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=False, show_all=False).keys())
            self.assertEqual(a, res)

    def test_case_showall(self):
        if False:
            i = 10
            return i + 15
        ns = root.__dict__
        tests = [('a*', ['abbot', 'abel', 'active', 'arna']), ('?b*.?o*', ['abbot.koppel', 'abbot.loop', 'abel.koppel', 'abel.loop']), ('_a*', ['_apan']), ('_*anka', ['__anka']), ('_*a*', ['__anka', '_apan'])]
        for (pat, res) in tests:
            res.sort()
            a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=False, show_all=True).keys())
            self.assertEqual(a, res)

    def test_nocase(self):
        if False:
            i = 10
            return i + 15
        ns = root.__dict__
        tests = [('a*', ['abbot', 'abel', 'ABEL', 'active', 'arna']), ('?b*.?o*', ['abbot.koppel', 'abbot.loop', 'abel.koppel', 'abel.loop', 'ABEL.koppel', 'ABEL.loop']), ('_a*', []), ('_*anka', ['__anka', '__ANKA']), ('_*a*', ['__anka', '__ANKA'])]
        for (pat, res) in tests:
            res.sort()
            a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=True, show_all=False).keys())
            self.assertEqual(a, res)

    def test_nocase_showall(self):
        if False:
            print('Hello World!')
        ns = root.__dict__
        tests = [('a*', ['abbot', 'abel', 'ABEL', 'active', 'arna']), ('?b*.?o*', ['abbot.koppel', 'abbot.loop', 'abel.koppel', 'abel.loop', 'ABEL.koppel', 'ABEL.loop']), ('_a*', ['_apan', '_APAN']), ('_*anka', ['__anka', '__ANKA']), ('_*a*', ['__anka', '__ANKA', '_apan', '_APAN'])]
        for (pat, res) in tests:
            res.sort()
            a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=True, show_all=True).keys())
            a.sort()
            self.assertEqual(a, res)

    def test_dict_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        'Dictionaries should be indexed by attributes, not by keys. This was\n        causing Github issue 129.'
        ns = {'az': {'king': 55}, 'pq': {1: 0}}
        tests = [('a*', ['az']), ('az.k*', ['az.keys']), ('pq.k*', ['pq.keys'])]
        for (pat, res) in tests:
            res.sort()
            a = sorted(wildcard.list_namespace(ns, 'all', pat, ignore_case=False, show_all=True).keys())
            self.assertEqual(a, res)

    def test_dict_dir(self):
        if False:
            for i in range(10):
                print('nop')

        class A(object):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.a = 1
                self.b = 2

            def __getattribute__(self, name):
                if False:
                    print('Hello World!')
                if name == 'a':
                    raise AttributeError
                return object.__getattribute__(self, name)
        a = A()
        adict = wildcard.dict_dir(a)
        assert 'a' not in adict
        self.assertEqual(adict['b'], 2)