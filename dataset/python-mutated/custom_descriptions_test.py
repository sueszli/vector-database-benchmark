"""Tests for custom description module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import custom_descriptions
from fire import testutils
LINE_LENGTH = 80

class CustomDescriptionTest(testutils.BaseTestCase):

    def test_string_type_summary_enough_space(self):
        if False:
            while True:
                i = 10
        component = 'Test'
        summary = custom_descriptions.GetSummary(obj=component, available_space=80, line_length=LINE_LENGTH)
        self.assertEqual(summary, '"Test"')

    def test_string_type_summary_not_enough_space_truncated(self):
        if False:
            for i in range(10):
                print('nop')
        component = 'Test'
        summary = custom_descriptions.GetSummary(obj=component, available_space=5, line_length=LINE_LENGTH)
        self.assertEqual(summary, '"..."')

    def test_string_type_summary_not_enough_space_new_line(self):
        if False:
            while True:
                i = 10
        component = 'Test'
        summary = custom_descriptions.GetSummary(obj=component, available_space=4, line_length=LINE_LENGTH)
        self.assertEqual(summary, '"Test"')

    def test_string_type_summary_not_enough_space_long_truncated(self):
        if False:
            i = 10
            return i + 15
        component = 'Lorem ipsum dolor sit amet'
        summary = custom_descriptions.GetSummary(obj=component, available_space=10, line_length=LINE_LENGTH)
        self.assertEqual(summary, '"Lorem..."')

    def test_string_type_description_enough_space(self):
        if False:
            return 10
        component = 'Test'
        description = custom_descriptions.GetDescription(obj=component, available_space=80, line_length=LINE_LENGTH)
        self.assertEqual(description, 'The string "Test"')

    def test_string_type_description_not_enough_space_truncated(self):
        if False:
            print('Hello World!')
        component = 'Lorem ipsum dolor sit amet'
        description = custom_descriptions.GetDescription(obj=component, available_space=20, line_length=LINE_LENGTH)
        self.assertEqual(description, 'The string "Lore..."')

    def test_string_type_description_not_enough_space_new_line(self):
        if False:
            while True:
                i = 10
        component = 'Lorem ipsum dolor sit amet'
        description = custom_descriptions.GetDescription(obj=component, available_space=10, line_length=LINE_LENGTH)
        self.assertEqual(description, 'The string "Lorem ipsum dolor sit amet"')
if __name__ == '__main__':
    testutils.main()