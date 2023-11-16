from wtforms.form import Form
from superset.forms import CommaSeparatedListField, filter_not_empty_values
from tests.integration_tests.base_tests import SupersetTestCase

class TestForm(SupersetTestCase):

    def test_comma_separated_list_field(self):
        if False:
            print('Hello World!')
        field = CommaSeparatedListField().bind(Form(), 'foo')
        field.process_formdata([''])
        self.assertEqual(field.data, [''])
        field.process_formdata(['a,comma,separated,list'])
        self.assertEqual(field.data, ['a', 'comma', 'separated', 'list'])

    def test_filter_not_empty_values(self):
        if False:
            print('Hello World!')
        self.assertEqual(filter_not_empty_values(None), None)
        self.assertEqual(filter_not_empty_values([]), None)
        self.assertEqual(filter_not_empty_values(['']), None)
        self.assertEqual(filter_not_empty_values(['hi']), ['hi'])