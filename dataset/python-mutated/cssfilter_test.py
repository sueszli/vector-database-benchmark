import unittest
from r2.lib.cssfilter import validate_css

class TestCSSFilter(unittest.TestCase):

    def assertInvalid(self, css):
        if False:
            i = 10
            return i + 15
        (serialized, errors) = validate_css(css, {})
        self.assertNotEqual(errors, [])

    def test_offsite_url(self):
        if False:
            return 10
        testcase = u"*{background-image:url('http://foobar/')}"
        self.assertInvalid(testcase)

    def test_nested_url(self):
        if False:
            for i in range(10):
                print('nop')
        testcase = u"*{background-image:calc(url('http://foobar/'))}"
        self.assertInvalid(testcase)

    def test_url_prelude(self):
        if False:
            print('Hello World!')
        testcase = u"*[foo=url('http://foobar/')]{color:red;}"
        self.assertInvalid(testcase)

    def test_invalid_property(self):
        if False:
            print('Hello World!')
        testcase = u'*{foo: red;}'
        self.assertInvalid(testcase)

    def test_import(self):
        if False:
            return 10
        testcase = u"@import 'foobar'; *{}"
        self.assertInvalid(testcase)

    def test_import_rule(self):
        if False:
            print('Hello World!')
        testcase = u"*{ @import 'foobar'; }"
        self.assertInvalid(testcase)

    def test_invalid_function(self):
        if False:
            for i in range(10):
                print('nop')
        testcase = u'*{color:expression(alert(1));}'
        self.assertInvalid(testcase)

    def test_invalid_function_prelude(self):
        if False:
            print('Hello World!')
        testcase = u'*[foo=expression(alert(1))]{color:red;}'
        self.assertInvalid(testcase)

    def test_semicolon_function(self):
        if False:
            while True:
                i = 10
        testcase = u'*{color: calc(;color:red;);}'
        self.assertInvalid(testcase)

    def test_semicolon_block(self):
        if False:
            i = 10
            return i + 15
        testcase = u'*{color: [;color:red;];}'
        self.assertInvalid(testcase)

    def test_escape_prelude(self):
        if False:
            print('Hello World!')
        testcase = u'*[foo=bar{}*{color:blue}]{color:red;}'
        self.assertInvalid(testcase)

    def test_escape_url(self):
        if False:
            print('Hello World!')
        testcase = u"*{background-image: url('foo bar');}"
        self.assertInvalid(testcase)

    def test_control_chars(self):
        if False:
            print('Hello World!')
        testcase = u"*{font-family:'foobar\x03;color:red;';}"
        self.assertInvalid(testcase)

    def test_embedded_nulls(self):
        if False:
            print('Hello World!')
        testcase = u"*{font-family:'foo\x00bar'}"
        self.assertInvalid(testcase)

    def test_escaped_url(self):
        if False:
            for i in range(10):
                print('nop')
        testcase = u"*{background-image:\\u\\r\\l('http://foobar/')}"
        self.assertInvalid(testcase)

    def test_escape_function_obfuscation(self):
        if False:
            return 10
        testcase = u'*{color: expression\\28 alert\\28 1 \\29 \\29 }'
        self.assertInvalid(testcase)

    def test_attr_url(self):
        if False:
            i = 10
            return i + 15
        testcase = u'*{background-image:attr(foobar url);}'
        self.assertInvalid(testcase)