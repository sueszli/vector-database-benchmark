import unittest
from django.template.smartif import IfParser

class SmartIfTests(unittest.TestCase):

    def assertCalcEqual(self, expected, tokens):
        if False:
            print('Hello World!')
        self.assertEqual(expected, IfParser(tokens).parse().eval({}))

    def test_not(self):
        if False:
            for i in range(10):
                print('nop')
        var = IfParser(['not', False]).parse()
        self.assertEqual('(not (literal False))', repr(var))
        self.assertTrue(var.eval({}))
        self.assertFalse(IfParser(['not', True]).parse().eval({}))

    def test_or(self):
        if False:
            for i in range(10):
                print('nop')
        var = IfParser([True, 'or', False]).parse()
        self.assertEqual('(or (literal True) (literal False))', repr(var))
        self.assertTrue(var.eval({}))

    def test_in(self):
        if False:
            i = 10
            return i + 15
        list_ = [1, 2, 3]
        self.assertCalcEqual(True, [1, 'in', list_])
        self.assertCalcEqual(False, [1, 'in', None])
        self.assertCalcEqual(False, [None, 'in', list_])

    def test_not_in(self):
        if False:
            return 10
        list_ = [1, 2, 3]
        self.assertCalcEqual(False, [1, 'not', 'in', list_])
        self.assertCalcEqual(True, [4, 'not', 'in', list_])
        self.assertCalcEqual(False, [1, 'not', 'in', None])
        self.assertCalcEqual(True, [None, 'not', 'in', list_])

    def test_precedence(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertCalcEqual(True, [False, 'and', False, 'or', True])
        self.assertCalcEqual(True, [True, 'or', False, 'and', False])
        self.assertCalcEqual(True, [1, 'or', 1, '==', 2])
        self.assertCalcEqual(True, [True, '==', True, 'or', True, '==', False])
        self.assertEqual('(or (and (== (literal 1) (literal 2)) (literal 3)) (literal 4))', repr(IfParser([1, '==', 2, 'and', 3, 'or', 4]).parse()))