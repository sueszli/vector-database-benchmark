from django.test import TestCase
from wagtail.test.utils import WagtailTestUtils
from wagtail.whitelist import Whitelister, allow_without_attributes, attribute_rule, check_url

class TestCheckUrl(TestCase):

    def test_allowed_url_schemes(self):
        if False:
            for i in range(10):
                print('nop')
        for url_scheme in ['', 'http', 'https', 'ftp', 'mailto', 'tel']:
            url = url_scheme + '://www.example.com'
            self.assertTrue(bool(check_url(url)))

    def test_disallowed_url_scheme(self):
        if False:
            while True:
                i = 10
        self.assertFalse(bool(check_url('invalid://url')))

    def test_crafty_disallowed_url_scheme(self):
        if False:
            print('Hello World!')
        "\n        Some URL parsers do not parse 'jav\tascript:' as a valid scheme.\n        Browsers, however, do. The checker needs to catch these crafty schemes\n        "
        self.assertFalse(bool(check_url("jav\tascript:alert('XSS')")))

class TestAttributeRule(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.soup = self.get_soup('<b foo="bar">baz</b>', 'html5lib')

    def test_no_rule_for_attr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that attribute_rule() drops attributes for\n        which no rule has been defined.\n        '
        tag = self.soup.b
        fn = attribute_rule({'snowman': 'barbecue'})
        fn(tag)
        self.assertEqual(str(tag), '<b>baz</b>')

    def test_rule_true_for_attr(self):
        if False:
            while True:
                i = 10
        '\n        Test that attribute_rule() does not change attributes\n        when the corresponding rule returns True\n        '
        tag = self.soup.b
        fn = attribute_rule({'foo': True})
        fn(tag)
        self.assertEqual(str(tag), '<b foo="bar">baz</b>')

    def test_rule_false_for_attr(self):
        if False:
            while True:
                i = 10
        '\n        Test that attribute_rule() drops attributes\n        when the corresponding rule returns False\n        '
        tag = self.soup.b
        fn = attribute_rule({'foo': False})
        fn(tag)
        self.assertEqual(str(tag), '<b>baz</b>')

    def test_callable_called_on_attr(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that when the rule returns a callable,\n        attribute_rule() replaces the attribute with\n        the result of calling the callable on the attribute.\n        '
        tag = self.soup.b
        fn = attribute_rule({'foo': len})
        fn(tag)
        self.assertEqual(str(tag), '<b foo="3">baz</b>')

    def test_callable_returns_None(self):
        if False:
            while True:
                i = 10
        '\n        Test that when the rule returns a callable,\n        attribute_rule() replaces the attribute with\n        the result of calling the callable on the attribute.\n        '
        tag = self.soup.b
        fn = attribute_rule({'foo': lambda x: None})
        fn(tag)
        self.assertEqual(str(tag), '<b>baz</b>')

    def test_allow_without_attributes(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that attribute_rule() with will drop all\n        attributes.\n        '
        soup = self.get_soup('<b foo="bar" baz="quux" snowman="barbecue"></b>', 'html5lib')
        tag = soup.b
        allow_without_attributes(tag)
        self.assertEqual(str(tag), '<b></b>')

class TestWhitelister(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.whitelister = Whitelister()

    def test_clean_unknown_node(self):
        if False:
            while True:
                i = 10
        '\n        Unknown node should remove a node from the parent document\n        '
        soup = self.get_soup('<foo><bar>baz</bar>quux</foo>', 'html5lib')
        tag = soup.foo
        self.whitelister.clean_unknown_node('', soup.bar)
        self.assertEqual(str(tag), '<foo>quux</foo>')

    def test_clean_tag_node_cleans_nested_recognised_node(self):
        if False:
            return 10
        '\n        <b> tags are allowed without attributes. This remains true\n        when tags are nested.\n        '
        soup = self.get_soup('<b><b class="delete me">foo</b></b>', 'html5lib')
        tag = soup.b
        self.whitelister.clean_tag_node(tag, tag)
        self.assertEqual(str(tag), '<b><b>foo</b></b>')

    def test_clean_tag_node_disallows_nested_unrecognised_node(self):
        if False:
            print('Hello World!')
        '\n        <foo> tags should be removed, even when nested.\n        '
        soup = self.get_soup('<b><foo>bar</foo></b>', 'html5lib')
        tag = soup.b
        self.whitelister.clean_tag_node(tag, tag)
        self.assertEqual(str(tag), '<b>bar</b>')

    def test_clean_string_node_does_nothing(self):
        if False:
            return 10
        soup = self.get_soup('<b>bar</b>', 'html5lib')
        string = soup.b.string
        self.whitelister.clean_string_node(string, string)
        self.assertEqual(str(string), 'bar')

    def test_clean_node_does_not_change_navigable_strings(self):
        if False:
            i = 10
            return i + 15
        soup = self.get_soup('<b>bar</b>', 'html5lib')
        string = soup.b.string
        self.whitelister.clean_node(string, string)
        self.assertEqual(str(string), 'bar')

    def test_clean(self):
        if False:
            while True:
                i = 10
        '\n        Whitelister.clean should remove disallowed tags and attributes from\n        a string\n        '
        string = '<b foo="bar">snowman <barbecue>Yorkshire</barbecue></b>'
        cleaned_string = self.whitelister.clean(string)
        self.assertEqual(cleaned_string, '<b>snowman Yorkshire</b>')

    def test_clean_comments(self):
        if False:
            i = 10
            return i + 15
        string = '<b>snowman Yorkshire<!--[if gte mso 10]>MS word junk<![endif]--></b>'
        cleaned_string = self.whitelister.clean(string)
        self.assertEqual(cleaned_string, '<b>snowman Yorkshire</b>')

    def test_quoting(self):
        if False:
            return 10
        string = '<img alt="Arthur &quot;two sheds&quot; Jackson" sheds="2">'
        cleaned_string = self.whitelister.clean(string)
        self.assertEqual(cleaned_string, '<img alt="Arthur &quot;two sheds&quot; Jackson"/>')