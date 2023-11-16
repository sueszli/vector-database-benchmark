"""
    Unit tests for salt.utils.yamlloader.SaltYamlSafeLoader
"""
import textwrap
from yaml.constructor import ConstructorError
import salt.utils.files
from salt.utils.yamlloader import SaltYamlSafeLoader, yaml
from tests.support.mock import mock_open, patch
from tests.support.unit import TestCase

class YamlLoaderTestCase(TestCase):
    """
    TestCase for salt.utils.yamlloader module
    """

    @staticmethod
    def render_yaml(data):
        if False:
            return 10
        '\n        Takes a YAML string, puts it into a mock file, passes that to the YAML\n        SaltYamlSafeLoader and then returns the rendered/parsed YAML data\n        '
        with patch('salt.utils.files.fopen', mock_open(read_data=data)) as mocked_file:
            with salt.utils.files.fopen(mocked_file) as mocked_stream:
                return SaltYamlSafeLoader(mocked_stream).get_data()

    def test_yaml_basics(self):
        if False:
            print('Hello World!')
        '\n        Test parsing an ordinary path\n        '
        self.assertEqual(self.render_yaml(textwrap.dedent('                p1:\n                  - alpha\n                  - beta')), {'p1': ['alpha', 'beta']})

    def test_yaml_merge(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test YAML anchors\n        '
        self.assertEqual(self.render_yaml(textwrap.dedent('                p1: &p1\n                  v1: alpha\n                p2:\n                  <<: *p1\n                  v2: beta')), {'p1': {'v1': 'alpha'}, 'p2': {'v1': 'alpha', 'v2': 'beta'}})
        self.assertEqual(self.render_yaml(textwrap.dedent('                p1: &p1\n                  v1: alpha\n                p2:\n                  <<: *p1\n                  v1: new_alpha')), {'p1': {'v1': 'alpha'}, 'p2': {'v1': 'new_alpha'}})
        self.assertEqual(self.render_yaml(textwrap.dedent('                p1: &p1\n                  v1: &v1\n                    - t1\n                    - t2\n                p2:\n                  v2: *v1')), {'p2': {'v2': ['t1', 't2']}, 'p1': {'v1': ['t1', 't2']}})

    def test_yaml_duplicates(self):
        if False:
            return 10
        '\n        Test that duplicates still throw an error\n        '
        with self.assertRaises(ConstructorError):
            self.render_yaml(textwrap.dedent('                p1: alpha\n                p1: beta'))
        with self.assertRaises(ConstructorError):
            self.render_yaml(textwrap.dedent('                p1: &p1\n                  v1: alpha\n                p2:\n                  <<: *p1\n                  v2: beta\n                  v2: betabeta'))

    def test_yaml_with_plain_scalars(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that plain (i.e. unqoted) string and non-string scalars are\n        properly handled\n        '
        self.assertEqual(self.render_yaml(textwrap.dedent('                foo:\n                  b: {foo: bar, one: 1, list: [1, two, 3]}')), {'foo': {'b': {'foo': 'bar', 'one': 1, 'list': [1, 'two', 3]}}})

    def test_not_yaml_monkey_patching(self):
        if False:
            while True:
                i = 10
        if hasattr(yaml, 'CSafeLoader'):
            assert yaml.SafeLoader != yaml.CSafeLoader