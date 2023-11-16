"""Tests generating test combinations."""
from collections import OrderedDict
from absl.testing import parameterized
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.eager import test

class TestingCombinationsTest(test.TestCase):

    def test_combine(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual([{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2, 'b': 2}, {'a': 2, 'b': 3}], combinations.combine(a=[1, 2], b=[2, 3]))

    def test_arguments_sorted(self):
        if False:
            return 10
        self.assertEqual([OrderedDict([('aa', 1), ('ab', 2)]), OrderedDict([('aa', 1), ('ab', 3)]), OrderedDict([('aa', 2), ('ab', 2)]), OrderedDict([('aa', 2), ('ab', 3)])], combinations.combine(ab=[2, 3], aa=[1, 2]))

    def test_combine_single_parameter(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual([{'a': 1, 'b': 2}, {'a': 2, 'b': 2}], combinations.combine(a=[1, 2], b=2))

    def test_add(self):
        if False:
            print('Hello World!')
        self.assertEqual([{'a': 1}, {'a': 2}, {'b': 2}, {'b': 3}], combinations.combine(a=[1, 2]) + combinations.combine(b=[2, 3]))

    def test_times(self):
        if False:
            print('Hello World!')
        c1 = combinations.combine(mode=['graph'], loss=['callable', 'tensor'])
        c2 = combinations.combine(mode=['eager'], loss=['callable'])
        c3 = combinations.combine(distribution=['d1', 'd2'])
        c4 = combinations.times(c3, c1 + c2)
        self.assertEqual([OrderedDict([('distribution', 'd1'), ('loss', 'callable'), ('mode', 'graph')]), OrderedDict([('distribution', 'd1'), ('loss', 'tensor'), ('mode', 'graph')]), OrderedDict([('distribution', 'd1'), ('loss', 'callable'), ('mode', 'eager')]), OrderedDict([('distribution', 'd2'), ('loss', 'callable'), ('mode', 'graph')]), OrderedDict([('distribution', 'd2'), ('loss', 'tensor'), ('mode', 'graph')]), OrderedDict([('distribution', 'd2'), ('loss', 'callable'), ('mode', 'eager')])], c4)

    def test_times_variable_arguments(self):
        if False:
            i = 10
            return i + 15
        c1 = combinations.combine(mode=['graph', 'eager'])
        c2 = combinations.combine(optimizer=['adam', 'gd'])
        c3 = combinations.combine(distribution=['d1', 'd2'])
        c4 = combinations.times(c3, c1, c2)
        self.assertEqual([OrderedDict([('distribution', 'd1'), ('mode', 'graph'), ('optimizer', 'adam')]), OrderedDict([('distribution', 'd1'), ('mode', 'graph'), ('optimizer', 'gd')]), OrderedDict([('distribution', 'd1'), ('mode', 'eager'), ('optimizer', 'adam')]), OrderedDict([('distribution', 'd1'), ('mode', 'eager'), ('optimizer', 'gd')]), OrderedDict([('distribution', 'd2'), ('mode', 'graph'), ('optimizer', 'adam')]), OrderedDict([('distribution', 'd2'), ('mode', 'graph'), ('optimizer', 'gd')]), OrderedDict([('distribution', 'd2'), ('mode', 'eager'), ('optimizer', 'adam')]), OrderedDict([('distribution', 'd2'), ('mode', 'eager'), ('optimizer', 'gd')])], c4)
        self.assertEqual(combinations.combine(mode=['graph', 'eager'], optimizer=['adam', 'gd'], distribution=['d1', 'd2']), c4)

    def test_overlapping_keys(self):
        if False:
            return 10
        c1 = combinations.combine(mode=['graph'], loss=['callable', 'tensor'])
        c2 = combinations.combine(mode=['eager'], loss=['callable'])
        with self.assertRaisesRegex(ValueError, '.*Keys.+overlap.+'):
            _ = combinations.times(c1, c2)

@combinations.generate(combinations.combine(a=[1, 0], b=[2, 3], c=[1]))
class CombineTheTestSuite(parameterized.TestCase):

    def test_add_things(self, a, b, c):
        if False:
            return 10
        self.assertLessEqual(3, a + b + c)
        self.assertLessEqual(a + b + c, 5)

    def test_add_things_one_more(self, a, b, c):
        if False:
            i = 10
            return i + 15
        self.assertLessEqual(3, a + b + c)
        self.assertLessEqual(a + b + c, 5)

    def not_a_test(self, a=0, b=0, c=0):
        if False:
            for i in range(10):
                print('nop')
        del a, b, c
        self.fail()

    def _test_but_private(self, a=0, b=0, c=0):
        if False:
            for i in range(10):
                print('nop')
        del a, b, c
        self.fail()
    test_member = 0
if __name__ == '__main__':
    test.main()