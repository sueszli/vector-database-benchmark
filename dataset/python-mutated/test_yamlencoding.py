"""
Tests for salt.utils.yamlencoding
"""
import salt.utils.yaml
import salt.utils.yamlencoding
from tests.support.unit import TestCase

class YamlEncodingTestCase(TestCase):

    def test_yaml_dquote(self):
        if False:
            while True:
                i = 10
        for teststr in ('"\\ []{}"',):
            self.assertEqual(teststr, salt.utils.yaml.safe_load(salt.utils.yamlencoding.yaml_dquote(teststr)))

    def test_yaml_dquote_doesNotAddNewLines(self):
        if False:
            return 10
        teststr = '"' * 100
        self.assertNotIn('\n', salt.utils.yamlencoding.yaml_dquote(teststr))

    def test_yaml_squote(self):
        if False:
            for i in range(10):
                print('nop')
        ret = salt.utils.yamlencoding.yaml_squote('"')
        self.assertEqual(ret, '\'"\'')

    def test_yaml_squote_doesNotAddNewLines(self):
        if False:
            for i in range(10):
                print('nop')
        teststr = "'" * 100
        self.assertNotIn('\n', salt.utils.yamlencoding.yaml_squote(teststr))

    def test_yaml_encode(self):
        if False:
            i = 10
            return i + 15
        for testobj in (None, True, False, '[7, 5]', '"monkey"', 5, 7.5, '2014-06-02 15:30:29.7'):
            self.assertEqual(testobj, salt.utils.yaml.safe_load(salt.utils.yamlencoding.yaml_encode(testobj)))
        for testobj in ({}, [], set()):
            self.assertRaises(TypeError, salt.utils.yamlencoding.yaml_encode, testobj)