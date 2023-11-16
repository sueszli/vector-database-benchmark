from runner.koan import *

class AboutAsserts(Koan):

    def test_assert_truth(self):
        if False:
            i = 10
            return i + 15
        '\n        We shall contemplate truth by testing reality, via asserts.\n        '
        self.assertTrue(False)

    def test_assert_with_message(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enlightenment may be more easily achieved with appropriate messages.\n        '
        self.assertTrue(False, 'This should be True -- Please fix this')

    def test_fill_in_values(self):
        if False:
            print('Hello World!')
        '\n        Sometimes we will ask you to fill in the values\n        '
        self.assertEqual(__, 1 + 1)

    def test_assert_equality(self):
        if False:
            print('Hello World!')
        '\n        To understand reality, we must compare our expectations against reality.\n        '
        expected_value = __
        actual_value = 1 + 1
        self.assertTrue(expected_value == actual_value)

    def test_a_better_way_of_asserting_equality(self):
        if False:
            print('Hello World!')
        '\n        Some ways of asserting equality are better than others.\n        '
        expected_value = __
        actual_value = 1 + 1
        self.assertEqual(expected_value, actual_value)

    def test_that_unittest_asserts_work_the_same_way_as_python_asserts(self):
        if False:
            print('Hello World!')
        '\n        Understand what lies within.\n        '
        assert False

    def test_that_sometimes_we_need_to_know_the_class_type(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        What is in a class name?\n        '
        self.assertEqual(__, 'navel'.__class__)