import pytest
from django.test import TestCase
from rest_framework.compat import apply_markdown
from rest_framework.utils.formatting import dedent
from rest_framework.views import APIView
DESCRIPTION = 'an example docstring\n====================\n\n* list\n* list\n\nanother header\n--------------\n\n    code block\n\nindented\n\n# hash style header #\n\n```json\n[{\n    "alpha": 1,\n    "beta": "this is a string"\n}]\n```'
MARKDOWN_DOCSTRING = '<h2 id="an-example-docstring">an example docstring</h2>\n<ul>\n<li>list</li>\n<li>list</li>\n</ul>\n<h3 id="another-header">another header</h3>\n<pre><code>code block\n</code></pre>\n<p>indented</p>\n<h2 id="hash-style-header">hash style header</h2>\n<div class="highlight"><pre><span></span><span class="p">[{</span><span class="w"></span><br /><span class="w">    </span><span class="nt">&quot;alpha&quot;</span><span class="p">:</span><span class="w"> </span><span class="mi">1</span><span class="p">,</span><span class="w"></span><br /><span class="w">    </span><span class="nt">&quot;beta&quot;</span><span class="p">:</span><span class="w"> </span><span class="s2">&quot;this is a string&quot;</span><span class="w"></span><br /><span class="p">}]</span><span class="w"></span><br /></pre></div>\n<p><br /></p>'

class TestViewNamesAndDescriptions(TestCase):

    def test_view_name_uses_class_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure view names are based on the class name.\n        '

        class MockView(APIView):
            pass
        assert MockView().get_view_name() == 'Mock'

    def test_view_name_uses_name_attribute(self):
        if False:
            return 10

        class MockView(APIView):
            name = 'Foo'
        assert MockView().get_view_name() == 'Foo'

    def test_view_name_uses_suffix_attribute(self):
        if False:
            return 10

        class MockView(APIView):
            suffix = 'List'
        assert MockView().get_view_name() == 'Mock List'

    def test_view_name_preferences_name_over_suffix(self):
        if False:
            for i in range(10):
                print('nop')

        class MockView(APIView):
            name = 'Foo'
            suffix = 'List'
        assert MockView().get_view_name() == 'Foo'

    def test_view_description_uses_docstring(self):
        if False:
            i = 10
            return i + 15
        'Ensure view descriptions are based on the docstring.'

        class MockView(APIView):
            """an example docstring
            ====================

            * list
            * list

            another header
            --------------

                code block

            indented

            # hash style header #

            ```json
            [{
                "alpha": 1,
                "beta": "this is a string"
            }]
            ```"""
        assert MockView().get_view_description() == DESCRIPTION

    def test_view_description_uses_description_attribute(self):
        if False:
            i = 10
            return i + 15

        class MockView(APIView):
            description = 'Foo'
        assert MockView().get_view_description() == 'Foo'

    def test_view_description_allows_empty_description(self):
        if False:
            print('Hello World!')

        class MockView(APIView):
            """Description."""
            description = ''
        assert MockView().get_view_description() == ''

    def test_view_description_can_be_empty(self):
        if False:
            while True:
                i = 10
        "\n        Ensure that if a view has no docstring,\n        then it's description is the empty string.\n        "

        class MockView(APIView):
            pass
        assert MockView().get_view_description() == ''

    def test_view_description_can_be_promise(self):
        if False:
            i = 10
            return i + 15
        '\n        Ensure a view may have a docstring that is actually a lazily evaluated\n        class that can be converted to a string.\n\n        See: https://github.com/encode/django-rest-framework/issues/1708\n        '

        class MockLazyStr:

            def __init__(self, string):
                if False:
                    i = 10
                    return i + 15
                self.s = string

            def __str__(self):
                if False:
                    i = 10
                    return i + 15
                return self.s

        class MockView(APIView):
            __doc__ = MockLazyStr('a gettext string')
        assert MockView().get_view_description() == 'a gettext string'

    @pytest.mark.skipif(not apply_markdown, reason='Markdown is not installed')
    def test_markdown(self):
        if False:
            print('Hello World!')
        '\n        Ensure markdown to HTML works as expected.\n        '
        assert apply_markdown(DESCRIPTION) == MARKDOWN_DOCSTRING

def test_dedent_tabs():
    if False:
        while True:
            i = 10
    result = 'first string\n\nsecond string'
    assert dedent('    first string\n\n    second string') == result
    assert dedent('first string\n\n    second string') == result
    assert dedent('\tfirst string\n\n\tsecond string') == result
    assert dedent('first string\n\n\tsecond string') == result