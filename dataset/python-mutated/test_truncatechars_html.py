from django.template.defaultfilters import truncatechars_html
from django.test import SimpleTestCase

class FunctionTests(SimpleTestCase):

    def test_truncate_zero(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(truncatechars_html('<p>one <a href="#">two - three <br>four</a> five</p>', 0), '…')

    def test_truncate(self):
        if False:
            return 10
        self.assertEqual(truncatechars_html('<p>one <a href="#">two - three <br>four</a> five</p>', 4), '<p>one…</p>')

    def test_truncate2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(truncatechars_html('<p>one <a href="#">two - three <br>four</a> five</p>', 9), '<p>one <a href="#">two …</a></p>')

    def test_truncate3(self):
        if False:
            print('Hello World!')
        self.assertEqual(truncatechars_html('<p>one <a href="#">two - three <br>four</a> five</p>', 100), '<p>one <a href="#">two - three <br>four</a> five</p>')

    def test_truncate_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(truncatechars_html('<b>Ångström</b> was here', 3), '<b>Ån…</b>')

    def test_truncate_something(self):
        if False:
            while True:
                i = 10
        self.assertEqual(truncatechars_html('a<b>b</b>c', 3), 'a<b>b</b>c')

    def test_invalid_arg(self):
        if False:
            for i in range(10):
                print('nop')
        html = '<p>one <a href="#">two - three <br>four</a> five</p>'
        self.assertEqual(truncatechars_html(html, 'a'), html)