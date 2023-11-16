from runner.koan import *

class AboutTrueAndFalse(Koan):

    def truth_value(self, condition):
        if False:
            i = 10
            return i + 15
        if condition:
            return 'true stuff'
        else:
            return 'false stuff'

    def test_true_is_treated_as_true(self):
        if False:
            print('Hello World!')
        self.assertEqual(__, self.truth_value(True))

    def test_false_is_treated_as_false(self):
        if False:
            print('Hello World!')
        self.assertEqual(__, self.truth_value(False))

    def test_none_is_treated_as_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(__, self.truth_value(None))

    def test_zero_is_treated_as_false(self):
        if False:
            while True:
                i = 10
        self.assertEqual(__, self.truth_value(0))

    def test_empty_collections_are_treated_as_false(self):
        if False:
            return 10
        self.assertEqual(__, self.truth_value([]))
        self.assertEqual(__, self.truth_value(()))
        self.assertEqual(__, self.truth_value({}))
        self.assertEqual(__, self.truth_value(set()))

    def test_blank_strings_are_treated_as_false(self):
        if False:
            print('Hello World!')
        self.assertEqual(__, self.truth_value(''))

    def test_everything_else_is_treated_as_true(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(__, self.truth_value(1))
        self.assertEqual(__, self.truth_value([0]))
        self.assertEqual(__, self.truth_value((0,)))
        self.assertEqual(__, self.truth_value('Python is named after Monty Python'))
        self.assertEqual(__, self.truth_value(' '))
        self.assertEqual(__, self.truth_value('0'))