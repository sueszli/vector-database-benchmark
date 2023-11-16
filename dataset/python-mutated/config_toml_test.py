from luigi.configuration import LuigiTomlParser, get_config, add_config_path
from helpers import LuigiTestCase

class TomlConfigParserTest(LuigiTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        add_config_path('test/testconfig/luigi.toml')
        add_config_path('test/testconfig/luigi_local.toml')

    def setUp(self):
        if False:
            while True:
                i = 10
        LuigiTomlParser._instance = None
        super(TomlConfigParserTest, self).setUp()

    def test_get_config(self):
        if False:
            print('Hello World!')
        config = get_config('toml')
        self.assertIsInstance(config, LuigiTomlParser)

    def test_file_reading(self):
        if False:
            print('Hello World!')
        config = get_config('toml')
        self.assertIn('hdfs', config.data)

    def test_get(self):
        if False:
            while True:
                i = 10
        config = get_config('toml')
        self.assertEqual(config.get('hdfs', 'client'), 'hadoopcli')
        self.assertEqual(config.get('hdfs', 'client', 'test'), 'hadoopcli')
        self.assertEqual(config.get('hdfs', 'test', 'check'), 'check')
        with self.assertRaises(KeyError):
            config.get('hdfs', 'test')
        self.assertEqual(config.get('hdfs', 'namenode_host'), 'localhost')
        self.assertEqual(config.get('hdfs', 'namenode_port'), 50030)

    def test_set(self):
        if False:
            return 10
        config = get_config('toml')
        self.assertEqual(config.get('hdfs', 'client'), 'hadoopcli')
        config.set('hdfs', 'client', 'test')
        self.assertEqual(config.get('hdfs', 'client'), 'test')
        config.set('hdfs', 'check', 'test me')
        self.assertEqual(config.get('hdfs', 'check'), 'test me')

    def test_has_option(self):
        if False:
            print('Hello World!')
        config = get_config('toml')
        self.assertTrue(config.has_option('hdfs', 'client'))
        self.assertFalse(config.has_option('hdfs', 'nope'))
        self.assertFalse(config.has_option('nope', 'client'))

class HelpersTest(LuigiTestCase):

    def test_add_without_install(self):
        if False:
            while True:
                i = 10
        enabled = LuigiTomlParser.enabled
        LuigiTomlParser.enabled = False
        with self.assertRaises(ImportError):
            add_config_path('test/testconfig/luigi.toml')
        LuigiTomlParser.enabled = enabled

    def test_get_without_install(self):
        if False:
            while True:
                i = 10
        enabled = LuigiTomlParser.enabled
        LuigiTomlParser.enabled = False
        with self.assertRaises(ImportError):
            get_config('toml')
        LuigiTomlParser.enabled = enabled