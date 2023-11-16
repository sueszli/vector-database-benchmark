from django.template.defaultfilters import truncatewords_html
from django.test import SimpleTestCase

class FunctionTests(SimpleTestCase):

    def test_truncate_zero(self):
        if False:
            while True:
                i = 10
        self.assertEqual(truncatewords_html('<p>one <a href="#">two - three <br>four</a> five</p>', 0), '')

    def test_truncate(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(truncatewords_html('<p>one <a href="#">two - three <br>four</a> five</p>', 2), '<p>one <a href="#">two …</a></p>')

    def test_truncate2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(truncatewords_html('<p>one <a href="#">two - three <br>four</a> five</p>', 4), '<p>one <a href="#">two - three …</a></p>')

    def test_truncate3(self):
        if False:
            while True:
                i = 10
        self.assertEqual(truncatewords_html('<p>one <a href="#">two - three <br>four</a> five</p>', 5), '<p>one <a href="#">two - three <br>four …</a></p>')

    def test_truncate4(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(truncatewords_html('<p>one <a href="#">two - three <br>four</a> five</p>', 100), '<p>one <a href="#">two - three <br>four</a> five</p>')

    def test_truncate_unicode(self):
        if False:
            return 10
        self.assertEqual(truncatewords_html('Ångström was here', 1), 'Ångström …')

    def test_truncate_complex(self):
        if False:
            print('Hello World!')
        self.assertEqual(truncatewords_html('<i>Buenos d&iacute;as! &#x00bf;C&oacute;mo est&aacute;?</i>', 3), '<i>Buenos d&iacute;as! &#x00bf;C&oacute;mo …</i>')

    def test_invalid_arg(self):
        if False:
            while True:
                i = 10
        self.assertEqual(truncatewords_html('<p>string</p>', 'a'), '<p>string</p>')