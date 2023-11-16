from helium import S
from tests.api import BrowserAT

class STest(BrowserAT):

    def get_page(self):
        if False:
            return 10
        return 'test_gui_elements.html'

    def test_find_by_id(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFindsEltWithId(S('#checkBoxId'), 'checkBoxId')

    def test_find_by_name(self):
        if False:
            return 10
        self.assertFindsEltWithId(S('@checkBoxName'), 'checkBoxId')

    def test_find_by_class(self):
        if False:
            while True:
                i = 10
        self.assertFindsEltWithId(S('.checkBoxClass'), 'checkBoxId')

    def test_find_by_xpath(self):
        if False:
            print('Hello World!')
        self.assertFindsEltWithId(S("//input[@type='checkbox' and @id='checkBoxId']"), 'checkBoxId')

    def test_find_by_css_selector(self):
        if False:
            print('Hello World!')
        self.assertFindsEltWithId(S('input.checkBoxClass'), 'checkBoxId')