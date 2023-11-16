from luigi.setup_logging import DaemonLogging, InterfaceLogging
from luigi.configuration import LuigiTomlParser, LuigiConfigParser, get_config
from helpers import unittest

class TestDaemonLogging(unittest.TestCase):
    cls = DaemonLogging

    def setUp(self):
        if False:
            return 10
        self.cls._configured = False

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.cls._configured = False
        self.cls.config = get_config()

    def test_cli(self):
        if False:
            for i in range(10):
                print('nop')
        opts = type('opts', (), {})
        opts.background = True
        result = self.cls._cli(opts)
        self.assertTrue(result)
        opts.background = False
        opts.logdir = './tests/'
        result = self.cls._cli(opts)
        self.assertTrue(result)
        opts.background = False
        opts.logdir = False
        result = self.cls._cli(opts)
        self.assertFalse(result)

    def test_section(self):
        if False:
            return 10
        self.cls.config = {'logging': {'version': 1, 'disable_existing_loggers': False, 'formatters': {'mockformatter': {'format': '{levelname}: {message}', 'style': '{', 'datefmt': '%Y-%m-%d %H:%M:%S'}}, 'handlers': {'mockhandler': {'class': 'logging.StreamHandler', 'level': 'INFO', 'formatter': 'mockformatter'}}, 'loggers': {'mocklogger': {'handlers': ('mockhandler',), 'level': 'INFO', 'disabled': False, 'propagate': False}}}}
        result = self.cls._section(None)
        self.assertTrue(result)
        self.cls.config = LuigiTomlParser()
        self.cls.config.read(['./test/testconfig/luigi_logging.toml'])
        result = self.cls._section(None)
        self.assertTrue(result)
        self.cls.config = {}
        result = self.cls._section(None)
        self.assertFalse(result)

    def test_section_cfg(self):
        if False:
            for i in range(10):
                print('nop')
        self.cls.config = LuigiConfigParser.instance()
        result = self.cls._section(None)
        self.assertFalse(result)

    def test_cfg(self):
        if False:
            print('Hello World!')
        self.cls.config = LuigiTomlParser()
        self.cls.config.data = {}
        result = self.cls._conf(None)
        self.assertFalse(result)
        self.cls.config.data = {'core': {'logging_conf_file': './blah'}}
        with self.assertRaises(OSError):
            self.cls._conf(None)
        self.cls.config.data = {'core': {'logging_conf_file': './test/testconfig/logging.cfg'}}
        result = self.cls._conf(None)
        self.assertTrue(result)

    def test_default(self):
        if False:
            while True:
                i = 10
        result = self.cls._default(None)
        self.assertTrue(result)

class TestInterfaceLogging(TestDaemonLogging):
    cls = InterfaceLogging

    def test_cli(self):
        if False:
            while True:
                i = 10
        opts = type('opts', (), {})
        result = self.cls._cli(opts)
        self.assertFalse(result)

    def test_cfg(self):
        if False:
            return 10
        self.cls.config = LuigiTomlParser()
        self.cls.config.data = {}
        opts = type('opts', (), {})
        opts.logging_conf_file = ''
        result = self.cls._conf(opts)
        self.assertFalse(result)
        opts.logging_conf_file = './blah'
        with self.assertRaises(OSError):
            self.cls._conf(opts)
        opts.logging_conf_file = './test/testconfig/logging.cfg'
        result = self.cls._conf(opts)
        self.assertTrue(result)

    def test_default(self):
        if False:
            i = 10
            return i + 15
        opts = type('opts', (), {})
        opts.log_level = 'INFO'
        result = self.cls._default(opts)
        self.assertTrue(result)

class PatchedLogging(InterfaceLogging):

    @classmethod
    def _cli(cls, *args):
        if False:
            for i in range(10):
                print('nop')
        cls.calls.append('_cli')
        return '_cli' not in cls.patched

    @classmethod
    def _conf(cls, *args):
        if False:
            while True:
                i = 10
        cls.calls.append('_conf')
        return '_conf' not in cls.patched

    @classmethod
    def _section(cls, *args):
        if False:
            for i in range(10):
                print('nop')
        cls.calls.append('_section')
        return '_section' not in cls.patched

    @classmethod
    def _default(cls, *args):
        if False:
            print('Hello World!')
        cls.calls.append('_default')
        return '_default' not in cls.patched

class TestSetup(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.opts = type('opts', (), {})
        self.cls = PatchedLogging
        self.cls.calls = []
        self.cls.config = LuigiTomlParser()
        self.cls._configured = False
        self.cls.patched = ('_cli', '_conf', '_section', '_default')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.cls.config = get_config()

    def test_configured(self):
        if False:
            return 10
        self.cls._configured = True
        result = self.cls.setup(self.opts)
        self.assertEqual(self.cls.calls, [])
        self.assertFalse(result)

    def test_disabled(self):
        if False:
            print('Hello World!')
        self.cls.config.data = {'core': {'no_configure_logging': True}}
        result = self.cls.setup(self.opts)
        self.assertEqual(self.cls.calls, [])
        self.assertFalse(result)

    def test_order(self):
        if False:
            print('Hello World!')
        self.cls.setup(self.opts)
        self.assertEqual(self.cls.calls, ['_cli', '_conf', '_section', '_default'])

    def test_cli(self):
        if False:
            i = 10
            return i + 15
        self.cls.patched = ()
        result = self.cls.setup(self.opts)
        self.assertTrue(result)
        self.assertEqual(self.cls.calls, ['_cli'])

    def test_conf(self):
        if False:
            for i in range(10):
                print('nop')
        self.cls.patched = ('_cli',)
        result = self.cls.setup(self.opts)
        self.assertTrue(result)
        self.assertEqual(self.cls.calls, ['_cli', '_conf'])

    def test_section(self):
        if False:
            print('Hello World!')
        self.cls.patched = ('_cli', '_conf')
        result = self.cls.setup(self.opts)
        self.assertTrue(result)
        self.assertEqual(self.cls.calls, ['_cli', '_conf', '_section'])

    def test_default(self):
        if False:
            while True:
                i = 10
        self.cls.patched = ('_cli', '_conf', '_section')
        result = self.cls.setup(self.opts)
        self.assertTrue(result)
        self.assertEqual(self.cls.calls, ['_cli', '_conf', '_section', '_default'])