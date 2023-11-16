from __future__ import absolute_import
import unittest2
from st2common.util import jinja as jinja_utils

class JinjaUtilsRenderTestCase(unittest2.TestCase):

    def test_render_values(self):
        if False:
            while True:
                i = 10
        actual = jinja_utils.render_values(mapping={'k1': '{{a}}', 'k2': '{{b}}'}, context={'a': 'v1', 'b': 'v2'})
        expected = {'k2': 'v2', 'k1': 'v1'}
        self.assertEqual(actual, expected)

    def test_render_values_skip_missing(self):
        if False:
            for i in range(10):
                print('nop')
        actual = jinja_utils.render_values(mapping={'k1': '{{a}}', 'k2': '{{b}}', 'k3': '{{c}}'}, context={'a': 'v1', 'b': 'v2'}, allow_undefined=True)
        expected = {'k2': 'v2', 'k1': 'v1', 'k3': ''}
        self.assertEqual(actual, expected)

    def test_render_values_ascii_and_unicode_values(self):
        if False:
            i = 10
            return i + 15
        mapping = {'k_ascii': '{{a}}', 'k_unicode': '{{b}}', 'k_ascii_unicode': '{{c}}'}
        context = {'a': 'some ascii value', 'b': '٩(̾●̮̮̃̾•̃̾)۶ ٩(̾●̮̮̃̾•̃̾)۶ ćšž', 'c': 'some ascii some ٩(̾●̮̮̃̾•̃̾)۶ ٩(̾●̮̮̃̾•̃̾)۶ '}
        expected = {'k_ascii': 'some ascii value', 'k_unicode': '٩(̾●̮̮̃̾•̃̾)۶ ٩(̾●̮̮̃̾•̃̾)۶ ćšž', 'k_ascii_unicode': 'some ascii some ٩(̾●̮̮̃̾•̃̾)۶ ٩(̾●̮̮̃̾•̃̾)۶ '}
        actual = jinja_utils.render_values(mapping=mapping, context=context, allow_undefined=True)
        self.assertEqual(actual, expected)

    def test_convert_str_to_raw(self):
        if False:
            return 10
        jinja_expr = '{{foobar}}'
        expected_raw_block = '{% raw %}{{foobar}}{% endraw %}'
        self.assertEqual(expected_raw_block, jinja_utils.convert_jinja_to_raw_block(jinja_expr))
        jinja_block_expr = '{% for item in items %}foobar{% end for %}'
        expected_raw_block = '{% raw %}{% for item in items %}foobar{% end for %}{% endraw %}'
        self.assertEqual(expected_raw_block, jinja_utils.convert_jinja_to_raw_block(jinja_block_expr))

    def test_convert_list_to_raw(self):
        if False:
            print('Hello World!')
        jinja_expr = ['foobar', '{{foo}}', '{{bar}}', '{% for item in items %}foobar{% end for %}', {'foobar': '{{foobar}}'}]
        expected_raw_block = ['foobar', '{% raw %}{{foo}}{% endraw %}', '{% raw %}{{bar}}{% endraw %}', '{% raw %}{% for item in items %}foobar{% end for %}{% endraw %}', {'foobar': '{% raw %}{{foobar}}{% endraw %}'}]
        self.assertListEqual(expected_raw_block, jinja_utils.convert_jinja_to_raw_block(jinja_expr))

    def test_convert_dict_to_raw(self):
        if False:
            i = 10
            return i + 15
        jinja_expr = {'var1': 'foobar', 'var2': ['{{foo}}', '{{bar}}'], 'var3': {'foobar': '{{foobar}}'}, 'var4': {'foobar': '{% for item in items %}foobar{% end for %}'}}
        expected_raw_block = {'var1': 'foobar', 'var2': ['{% raw %}{{foo}}{% endraw %}', '{% raw %}{{bar}}{% endraw %}'], 'var3': {'foobar': '{% raw %}{{foobar}}{% endraw %}'}, 'var4': {'foobar': '{% raw %}{% for item in items %}foobar{% end for %}{% endraw %}'}}
        self.assertDictEqual(expected_raw_block, jinja_utils.convert_jinja_to_raw_block(jinja_expr))