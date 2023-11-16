"""Classic Page Object Model with BaseCase inheritance."""
from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class DataPage:

    def go_to_data_url(self, sb):
        if False:
            print('Hello World!')
        sb.open('data:text/html,<p>Hello!</p><input />')

    def add_input_text(self, sb, text):
        if False:
            while True:
                i = 10
        sb.type('input', text)

class ObjTests(BaseCase):

    def test_data_url_page(self):
        if False:
            return 10
        DataPage().go_to_data_url(self)
        self.assert_text('Hello!', 'p')
        DataPage().add_input_text(self, 'Goodbye!')