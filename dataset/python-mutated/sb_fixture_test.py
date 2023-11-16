"""Classic Page Object Model with the "sb" fixture."""

class DataPage:

    def go_to_data_url(self, sb):
        if False:
            print('Hello World!')
        sb.open('data:text/html,<p>Hello!</p><input />')

    def add_input_text(self, sb, text):
        if False:
            for i in range(10):
                print('nop')
        sb.type('input', text)

class ObjTests:

    def test_data_url_page(self, sb):
        if False:
            while True:
                i = 10
        DataPage().go_to_data_url(sb)
        sb.assert_text('Hello!', 'p')
        DataPage().add_input_text(sb, 'Goodbye!')