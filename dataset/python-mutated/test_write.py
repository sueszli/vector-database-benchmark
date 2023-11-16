from helium import write, TextField
from tests.api import BrowserAT

class WriteTest(BrowserAT):

    def get_page(self):
        if False:
            print('Hello World!')
        return 'test_write.html'

    def test_write(self):
        if False:
            while True:
                i = 10
        write('Hello World!')
        self.assertEqual('Hello World!', TextField('Autofocus text field').value)

    def test_write_into(self):
        if False:
            while True:
                i = 10
        write('Hi there!', into='Normal text field')
        self.assertEqual('Hi there!', TextField('Normal text field').value)

    def test_write_into_text_field_to_right_of(self):
        if False:
            for i in range(10):
                print('nop')
        write('Hi there!', into=TextField(to_right_of='Normal text field'))
        self.assertEqual('Hi there!', TextField('Normal text field').value)