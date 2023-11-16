"""
Additional unit tests for :mod:`behave.model`.
"""
from __future__ import absolute_import, print_function
from behave.model import ScenarioOutline, ScenarioOutlineBuilder, Table, Tag, Row
from behave.model_describe import ModelDescriptor
from behave.textutil import text
from behave.parser import parse_step, parse_tags
import six
import pytest

def make_row(*data, **kwargs):
    if False:
        print('Hello World!')
    line = kwargs.pop('line', None)
    data2 = dict(data, **kwargs)
    headings = list(data2.keys())
    cells = [text(value) for value in data2.values()]
    return Row(headings, cells, line=line)

def step_to_text(step, indentation='    '):
    if False:
        return 10
    step_text = u'%s %s' % (step.keyword, step.name)
    more_text = None
    if step.text:
        more_text = ModelDescriptor.describe_docstring(step.text, indentation)
    elif step.table:
        more_text = ModelDescriptor.describe_table(step.table, indentation)
    if more_text:
        step_text = u'%s\n%s' % (step_text, more_text)
    return step_text.rstrip()

class TestScenarioOutlineBuilder(object):
    """Unit tests for the templating mechanism that is provided by the
    :class:`behave.model:ScenarioOutlineBuilder`.
    """

    @staticmethod
    def assert_make_step_for_row(step_text, expected_text, params=None):
        if False:
            i = 10
            return i + 15
        if params is None:
            params = {}
        step = parse_step(step_text)
        row = make_row(**params)
        output = ScenarioOutlineBuilder.make_step_for_row(step, row)
        assert step_to_text(output) == expected_text

    @staticmethod
    def assert_make_row_tags(tag_text, expected_tags, params=None):
        if False:
            print('Hello World!')
        if params is None:
            params = {}
        tags = parse_tags(tag_text)
        row = make_row(**params)
        actual_tags = ScenarioOutlineBuilder.make_row_tags(tags, row)
        assert actual_tags == expected_tags

    def test_make_step_for_row__without_placeholders_remains_unchanged(self):
        if False:
            while True:
                i = 10
        step_text = u'Given a step without placeholders'
        expected_text = text(step_text)
        params = dict(firstname='Alice', lastname='Beauville')
        self.assert_make_step_for_row(step_text, expected_text, params)

    def test_make_step_for_row__with_placeholders_in_step(self):
        if False:
            print('Hello World!')
        step_text = u'Given a person with "<firstname> <lastname>"'
        expected_text = u'Given a person with "Alice Beauville"'
        params = dict(firstname='Alice', lastname='Beauville')
        self.assert_make_step_for_row(step_text, expected_text, params)

    def test_make_step_for_row__with_placeholders_in_text(self):
        if False:
            for i in range(10):
                print('nop')
        step_text = u'Given a simple multi-line text:\n    """\n    <param_1>\n    Hello Alice\n    <param_2> <param_3>\n    __FINI__\n    """ \n'.strip()
        expected_text = u'Given a simple multi-line text\n    """\n    Param_1\n    Hello Alice\n    Hello Bob\n    __FINI__\n    """ \n'.strip()
        params = dict(param_1='Param_1', param_2='Hello', param_3='Bob')
        self.assert_make_step_for_row(step_text, expected_text, params)

    def test_make_step_for_row__without_placeholders_in_table(self):
        if False:
            while True:
                i = 10
        step_text = u'Given a simple data table\n    | Column_1 | Column_2 |\n    | Lorem ipsum | Ipsum lorem |\n'.strip()
        expected_text = u'Given a simple data table\n    | Column_1    | Column_2    |\n    | Lorem ipsum | Ipsum lorem |\n'.strip()
        self.assert_make_step_for_row(step_text, expected_text, params=None)

    def test_make_step_for_row__with_placeholders_in_table_headings(self):
        if False:
            i = 10
            return i + 15
        step_text = u'Given a simple data table:\n    | <param_1> | Column_2 | <param_2>_<param_3> |\n    | Lorem ipsum | 1234   | Ipsum lorem |\n'.strip()
        expected_text = u'Given a simple data table\n    | Column_1    | Column_2 | Hello_Column_3 |\n    | Lorem ipsum | 1234     | Ipsum lorem    |\n'.strip()
        params = dict(param_1='Column_1', param_2='Hello', param_3='Column_3')
        self.assert_make_step_for_row(step_text, expected_text, params)

    def test_make_step_for_row__with_placeholders_in_table_cells(self):
        if False:
            while True:
                i = 10
        step_text = u'Given a simple data table:\n    | Column_1 | Column_2 |\n    | Lorem ipsum | <param_1> |\n    | <param_2> <param_3> | Ipsum lorem |\n'.strip()
        expected_text = u'Given a simple data table\n    | Column_1    | Column_2    |\n    | Lorem ipsum | Cell_1      |\n    | Hello Alice | Ipsum lorem |\n'.strip()
        params = dict(param_1='Cell_1', param_2='Hello', param_3='Alice')
        self.assert_make_step_for_row(step_text, expected_text, params)

    @pytest.mark.parametrize('tag_template,expected', [(u'@use.with_category1=<param_1>', u'use.with_category1=PARAM_1'), (u'@not.with_category2=<param_2>', u'not.with_category2=PARAM_2')])
    def test_make_row_tags__with_active_tag_syntax(self, tag_template, expected):
        if False:
            while True:
                i = 10
        params = dict(param_1='PARAM_1', param_2='PARAM_2', param_3='UNUSED')
        expected_tags = [expected]
        self.assert_make_row_tags(tag_template, expected_tags, params)

    @pytest.mark.parametrize('tag_template,expected', [(u'@tag_1.func(param1=<param_1>,param2=<param_2>)', u'tag_1.func(param1=PARAM_1,param2=PARAM_2)')])
    def test_make_row_tags__with_function_like_syntax(self, tag_template, expected):
        if False:
            return 10
        tag_template = u'@tag_1.func(param1=<param_1>,param2=<param_2>)'
        params = dict(param_1='PARAM_1', param_2='PARAM_2', param_3='UNUSED')
        expected_tags = [u'tag_1.func(param1=PARAM_1,param2=PARAM_2)']
        self.assert_make_row_tags(tag_template, expected_tags, params)

    @pytest.mark.parametrize('tag_template,expected', [(u'@tag.category1:param1=<param_1>', u'tag.category1:param1=PARAM_1'), (u'@tag.category2:param1=<param_1>,param2=<param_2>', u'tag.category2:param1=PARAM_1,param2=PARAM_2'), (u'@tag.category3:param1=<param_1>;param2=<param_2>', u'tag.category3:param1=PARAM_1;param2=PARAM_2')])
    def test_make_row_tags__with_params_syntax(self, tag_template, expected):
        if False:
            return 10
        params = dict(param_1='PARAM_1', param_2='PARAM_2', param_3='UNUSED')
        expected_tags = [expected]
        self.assert_make_row_tags(tag_template, expected_tags, params)

class TestTag(object):
    """
    Translation rules are:
      * alnum chars => same, kept
      * space chars => "_"
      * other chars => deleted

    Preserve following characters (in addition to alnums, like: A-z, 0-9):
      * dots        => "." (support: dotted-names, active-tag name schema)
      * minus       => "-" (support: dashed-names)
      * underscore  => "_"
      * equal       => "=" (support: active-tag name schema)
      * colon       => ":" (support: active-tag name schema or similar)
      * semicolon   => ";" (support: complex tag notation)
      * comma       => "," (support: complex tag notation)
      * opening-parens  => "(" (support: complex tag notation)
      * closing-parens  => ")" (support: complex tag notation)
    """
    SAME_AS_TAG = '$SAME_AS_TAG'

    def check_make_name(self, tag, expected):
        if False:
            i = 10
            return i + 15
        if expected is self.SAME_AS_TAG:
            expected = tag
        actual_name = Tag.make_name(tag, allowed_chars=Tag.allowed_chars)
        assert actual_name == expected

    @pytest.mark.parametrize('tag,expected', [(u'foo.bar', SAME_AS_TAG)])
    def test_make_name__with_dotted_names(self, tag, expected):
        if False:
            print('Hello World!')
        self.check_make_name(tag, expected)

    @pytest.mark.parametrize('tag,expected', [(u'foo-bar', SAME_AS_TAG)])
    def test_make_name__with_dashed_names(self, tag, expected):
        if False:
            return 10
        self.check_make_name(tag, expected)

    @pytest.mark.parametrize('tag,expected', [(u'foo bar', 'foo_bar'), (u'foo\tbar', 'foo_bar'), (u'foo\nbar', 'foo_bar')])
    def test_make_name__spaces_replaced_with_underscore(self, tag, expected):
        if False:
            i = 10
            return i + 15
        self.check_make_name(tag, expected)

    @pytest.mark.parametrize('tag,expected', [(u'foo_bar', SAME_AS_TAG), (u'foo=bar', SAME_AS_TAG), (u'foo:bar', SAME_AS_TAG), (u'foo;bar', SAME_AS_TAG), (u'foo,bar', SAME_AS_TAG), (u'foo(bar=1)', SAME_AS_TAG)])
    def test_make_name__alloweds_char_remain_unmodified(self, tag, expected):
        if False:
            print('Hello World!')
        self.check_make_name(tag, expected)

    @pytest.mark.parametrize('tag,expected', [(u'foo<bar>', u'foobar'), (u'foo$bar', u'foobar')])
    def test_make_name__other_chars_are_removed(self, tag, expected):
        if False:
            print('Hello World!')
        self.check_make_name(tag, expected)