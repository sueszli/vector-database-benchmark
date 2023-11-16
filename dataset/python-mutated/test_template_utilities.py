from __future__ import annotations
import jinja2
import unittest
from ansible.template import AnsibleUndefined, _escape_backslashes, _count_newlines_from_end

class TestBackslashEscape(unittest.TestCase):
    test_data = (dict(template=u"{{ 'test2 %s' | format('\\1') }}", intermediate=u"{{ 'test2 %s' | format('\\\\1') }}", expectation=u'test2 \\1', args=dict()), dict(template=u"Test 2\\3: {{ '\\1 %s' | format('\\2') }}", intermediate=u"Test 2\\3: {{ '\\\\1 %s' | format('\\\\2') }}", expectation=u'Test 2\\3: \\1 \\2', args=dict()), dict(template=u"Test 2\\3: {{ 'test2 %s' | format('\\1') }}; \\done", intermediate=u"Test 2\\3: {{ 'test2 %s' | format('\\\\1') }}; \\done", expectation=u'Test 2\\3: test2 \\1; \\done', args=dict()), dict(template=u"{{ 'test2 %s' | format(var1) }}", intermediate=u"{{ 'test2 %s' | format(var1) }}", expectation=u'test2 \\1', args=dict(var1=u'\\1')), dict(template=u"Test 2\\3: {{ var1 | format('\\2') }}", intermediate=u"Test 2\\3: {{ var1 | format('\\\\2') }}", expectation=u'Test 2\\3: \\1 \\2', args=dict(var1=u'\\1 %s')))

    def setUp(self):
        if False:
            while True:
                i = 10
        self.env = jinja2.Environment()

    def test_backslash_escaping(self):
        if False:
            print('Hello World!')
        for test in self.test_data:
            intermediate = _escape_backslashes(test['template'], self.env)
            self.assertEqual(intermediate, test['intermediate'])
            template = jinja2.Template(intermediate)
            args = test['args']
            self.assertEqual(template.render(**args), test['expectation'])

class TestCountNewlines(unittest.TestCase):

    def test_zero_length_string(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(_count_newlines_from_end(u''), 0)

    def test_short_string(self):
        if False:
            return 10
        self.assertEqual(_count_newlines_from_end(u'The quick\n'), 1)

    def test_one_newline(self):
        if False:
            while True:
                i = 10
        self.assertEqual(_count_newlines_from_end(u'The quick brown fox jumped over the lazy dog' * 1000 + u'\n'), 1)

    def test_multiple_newlines(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(_count_newlines_from_end(u'The quick brown fox jumped over the lazy dog' * 1000 + u'\n\n\n'), 3)

    def test_zero_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(_count_newlines_from_end(u'The quick brown fox jumped over the lazy dog' * 1000), 0)

    def test_all_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(_count_newlines_from_end(u'\n' * 10), 10)

    def test_mostly_newlines(self):
        if False:
            return 10
        self.assertEqual(_count_newlines_from_end(u'The quick brown fox jumped over the lazy dog' + u'\n' * 1000), 1000)

class TestAnsibleUndefined(unittest.TestCase):

    def test_getattr(self):
        if False:
            print('Hello World!')
        val = AnsibleUndefined()
        self.assertIs(getattr(val, 'foo'), val)
        self.assertRaises(AttributeError, getattr, val, '__UNSAFE__')