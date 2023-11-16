import sys
import unittest
from typing import Optional
try:
    from tools.lib.template_parser import TemplateParserError, is_django_block_tag, tokenize, validate
except ImportError:
    print('ERROR!!! You need to run this via tools/test-tools.')
    sys.exit(1)

class ParserTest(unittest.TestCase):

    def _assert_validate_error(self, error: str, fn: Optional[str]=None, text: Optional[str]=None, template_format: Optional[str]=None) -> None:
        if False:
            return 10
        with self.assertRaisesRegex(TemplateParserError, error):
            validate(fn=fn, text=text, template_format=template_format)

    def test_is_django_block_tag(self) -> None:
        if False:
            return 10
        self.assertTrue(is_django_block_tag('block'))
        self.assertFalse(is_django_block_tag('not a django tag'))

    def test_validate_vanilla_html(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that validate() does not raise errors for\n        well-formed HTML.\n        '
        my_html = '\n            <table>\n                <tr>\n                <td>foo</td>\n                </tr>\n            </table>'
        validate(text=my_html)

    def test_validate_handlebars(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        my_html = '\n            {{#with stream}}\n                <p>{{stream}}</p>\n            {{/with}}\n            '
        validate(text=my_html, template_format='handlebars')

    def test_validate_handlebars_partial_block(self) -> None:
        if False:
            print('Hello World!')
        my_html = '\n            {{#> generic_thing }}\n                <p>hello!</p>\n            {{/generic_thing}}\n            '
        validate(text=my_html, template_format='handlebars')

    def test_validate_bad_handlebars_partial_block(self) -> None:
        if False:
            return 10
        my_html = '\n            {{#> generic_thing }}\n                <p>hello!</p>\n            {{# generic_thing}}\n            '
        self._assert_validate_error('Missing end tag for the token at row 4 13!', text=my_html, template_format='handlebars')

    def test_validate_comment(self) -> None:
        if False:
            return 10
        my_html = '\n            <!---\n                <h1>foo</h1>\n            -->'
        validate(text=my_html)

    def test_validate_django(self) -> None:
        if False:
            while True:
                i = 10
        my_html = '\n            {% include "some_other.html" %}\n            {% if foo %}\n                <p>bar</p>\n            {% endif %}\n            '
        validate(text=my_html, template_format='django')
        my_html = '\n            {% block "content" %}\n                {% with className="class" %}\n                {% include \'foobar\' %}\n                {% endwith %}\n            {% endblock %}\n            '
        validate(text=my_html)

    def test_validate_no_start_tag(self) -> None:
        if False:
            print('Hello World!')
        my_html = '\n            foo</p>\n        '
        self._assert_validate_error('No start tag', text=my_html)

    def test_validate_mismatched_tag(self) -> None:
        if False:
            return 10
        my_html = '\n            <b>foo</i>\n        '
        self._assert_validate_error('Mismatched tags: \\(b != i\\)', text=my_html)

    def test_validate_bad_indentation(self) -> None:
        if False:
            return 10
        my_html = '\n            <p>\n                foo\n                </p>\n        '
        self._assert_validate_error('Indentation for start/end tags does not match.', text=my_html)

    def test_validate_state_depth(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        my_html = '\n            <b>\n        '
        self._assert_validate_error('Missing end tag', text=my_html)

    def test_validate_incomplete_handlebars_tag_1(self) -> None:
        if False:
            return 10
        my_html = '\n            {{# foo\n        '
        self._assert_validate_error('Tag missing "}}" at line 2 col 13:"{{# foo\n        "', text=my_html, template_format='handlebars')

    def test_validate_incomplete_handlebars_tag_2(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        my_html = '\n            {{# foo }\n        '
        self._assert_validate_error('Tag missing "}}" at line 2 col 13:"{{# foo }\n"', text=my_html, template_format='handlebars')

    def test_validate_incomplete_django_tag_1(self) -> None:
        if False:
            while True:
                i = 10
        my_html = '\n            {% foo\n        '
        self._assert_validate_error('Tag missing "%}" at line 2 col 13:"{% foo\n        "', text=my_html, template_format='django')

    def test_validate_incomplete_django_tag_2(self) -> None:
        if False:
            print('Hello World!')
        my_html = '\n            {% foo %\n        '
        self._assert_validate_error('Tag missing "%}" at line 2 col 13:"{% foo %\n"', text=my_html, template_format='django')

    def test_validate_incomplete_html_tag_1(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        my_html = '\n            <b\n        '
        self._assert_validate_error('Tag missing ">" at line 2 col 13:"<b\n        "', text=my_html)

    def test_validate_incomplete_html_tag_2(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        my_html = '\n            <a href="\n        '
        my_html1 = '\n            <a href=""\n        '
        self._assert_validate_error('Tag missing ">" at line 2 col 13:"<a href=""\n        "', text=my_html1)
        self._assert_validate_error('Unbalanced quotes at line 2 col 13:"<a href="\n        "', text=my_html)

    def test_validate_empty_html_tag(self) -> None:
        if False:
            print('Hello World!')
        my_html = '\n            < >\n        '
        self._assert_validate_error('Tag name missing', text=my_html)

    def test_code_blocks(self) -> None:
        if False:
            return 10
        my_html = '\n            <code>\n                x = 5\n                y = x + 1\n            </code>'
        validate(text=my_html)
        my_html = '<code>process_widgets()</code>'
        validate(text=my_html)
        my_html = '\n            <code>x =\n            5</code>\n            '
        self._assert_validate_error('Code tag is split across two lines.', text=my_html)

    def test_anchor_blocks(self) -> None:
        if False:
            return 10
        my_html = '\n            <a href="/some/url">\n            Click here\n            for more info.\n            </a>'
        validate(text=my_html)
        my_html = '<a href="/some/url">click here</a>'
        validate(text=my_html)
        my_html = '\n            <a class="twitter-timeline" href="https://twitter.com/ZulipStatus"\n                data-widget-id="443457763394334720"\n                data-screen-name="ZulipStatus"\n                >@ZulipStatus on Twitter</a>.\n            '
        validate(text=my_html)

    def test_validate_jinja2_whitespace_markers_1(self) -> None:
        if False:
            while True:
                i = 10
        my_html = '\n        {% if foo -%}\n        this is foo\n        {% endif %}\n        '
        validate(text=my_html, template_format='django')

    def test_validate_jinja2_whitespace_markers_2(self) -> None:
        if False:
            while True:
                i = 10
        my_html = '\n        {% if foo %}\n        this is foo\n        {%- endif %}\n        '
        validate(text=my_html, template_format='django')

    def test_validate_jinja2_whitespace_markers_3(self) -> None:
        if False:
            i = 10
            return i + 15
        my_html = '\n        {% if foo %}\n        this is foo\n        {% endif -%}\n        '
        validate(text=my_html, template_format='django')

    def test_validate_jinja2_whitespace_markers_4(self) -> None:
        if False:
            return 10
        my_html = '\n        {%- if foo %}\n        this is foo\n        {% endif %}\n        '
        validate(text=my_html, template_format='django')

    def test_validate_mismatch_jinja2_whitespace_markers_1(self) -> None:
        if False:
            return 10
        my_html = '\n        {% if foo %}\n        this is foo\n        {%- if bar %}\n        '
        self._assert_validate_error('Missing end tag', text=my_html, template_format='django')

    def test_validate_jinja2_whitespace_type2_markers(self) -> None:
        if False:
            return 10
        my_html = '\n        {%- if foo -%}\n        this is foo\n        {% endif %}\n        '
        validate(text=my_html, template_format='django')

    def test_tokenize(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        tag = '<!DOCTYPE html>'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_doctype')
        tag = '<a>bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_start')
        self.assertEqual(token.tag, 'a')
        tag = '<br />bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_singleton')
        self.assertEqual(token.tag, 'br')
        tag = '<input>bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_start')
        self.assertEqual(token.tag, 'input')
        tag = '<input />bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_singleton')
        self.assertEqual(token.tag, 'input')
        tag = '</a>bla'
        token = tokenize(tag)[0]
        self.assertEqual(token.kind, 'html_end')
        self.assertEqual(token.tag, 'a')
        tag = '{{#with foo}}bla'
        token = tokenize(tag, template_format='handlebars')[0]
        self.assertEqual(token.kind, 'handlebars_start')
        self.assertEqual(token.tag, 'with')
        tag = '{{/with}}bla'
        token = tokenize(tag, template_format='handlebars')[0]
        self.assertEqual(token.kind, 'handlebars_end')
        self.assertEqual(token.tag, 'with')
        tag = '{{#> compose_banner }}bla'
        token = tokenize(tag, template_format='handlebars')[0]
        self.assertEqual(token.kind, 'handlebars_partial_block')
        self.assertEqual(token.tag, 'compose_banner')
        tag = '{% if foo %}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'django_start')
        self.assertEqual(token.tag, 'if')
        tag = '{% endif %}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'django_end')
        self.assertEqual(token.tag, 'if')
        tag = '{% if foo -%}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'jinja2_whitespace_stripped_start')
        self.assertEqual(token.tag, 'if')
        tag = '{%- endif %}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'jinja2_whitespace_stripped_end')
        self.assertEqual(token.tag, 'if')
        tag = '{%- if foo -%}bla'
        token = tokenize(tag, template_format='django')[0]
        self.assertEqual(token.kind, 'jinja2_whitespace_stripped_type2_start')
        self.assertEqual(token.tag, 'if')