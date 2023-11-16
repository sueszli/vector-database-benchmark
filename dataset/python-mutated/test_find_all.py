from selenium.common.exceptions import StaleElementReferenceException
from helium import find_all, Button, TextField, write
from tests.api import BrowserAT

class FindAllTest(BrowserAT):

    def get_page(self):
        if False:
            i = 10
            return i + 15
        return 'test_gui_elements.html'

    def test_find_all_duplicate_button(self):
        if False:
            while True:
                i = 10
        self.assertEqual(4, len(find_all(Button('Duplicate Button'))))

    def test_find_all_duplicate_button_to_right_of(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(2, len(find_all(Button('Duplicate Button', to_right_of='Row 1'))))

    def test_find_all_duplicate_button_below_to_right_of(self):
        if False:
            while True:
                i = 10
        self.assertEqual(1, len(find_all(Button('Duplicate Button', below='Column 1', to_right_of='Row 1'))))

    def test_find_all_nested_search_areas(self):
        if False:
            print('Hello World!')
        button = Button('Duplicate Button', below='Column 1', to_right_of='Row 1')
        self.assertEqual(1, len(find_all(Button('Duplicate Button', below=button))))

    def test_find_all_non_existent_button(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual([], find_all(Button('Non-existent Button')))

    def test_find_all_yields_api_elements(self):
        if False:
            return 10
        self.assertIsInstance(find_all(TextField('Example Text Field'))[0], TextField)

    def test_interact_with_found_elements(self):
        if False:
            return 10
        all_tfs = find_all(TextField())
        example_tf = None
        for text_field in all_tfs:
            try:
                id_ = text_field.web_element.get_attribute('id')
            except StaleElementReferenceException:
                pass
            else:
                if id_ == 'exampleTextFieldId':
                    example_tf = text_field
        self.assertIsNotNone(example_tf)
        write('test_interact_with_found_elements', into=example_tf)
        self.assertEqual('test_interact_with_found_elements', TextField('Example Text Field').value)