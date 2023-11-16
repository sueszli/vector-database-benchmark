from __future__ import annotations
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from ansible.module_utils.common._utils import get_all_subclasses
from ansible.modules import hostname
from units.modules.utils import ModuleTestCase, set_module_args

class TestHostname(ModuleTestCase):

    @patch('os.path.isfile')
    def test_stategy_get_never_writes_in_check_mode(self, isfile):
        if False:
            while True:
                i = 10
        isfile.return_value = True
        set_module_args({'name': 'fooname', '_ansible_check_mode': True})
        subclasses = get_all_subclasses(hostname.BaseStrategy)
        module = MagicMock()
        for cls in subclasses:
            instance = cls(module)
            instance.module.run_command = MagicMock()
            instance.module.run_command.return_value = (0, '', '')
            m = mock_open()
            builtins = 'builtins'
            with patch('%s.open' % builtins, m):
                instance.get_permanent_hostname()
                instance.get_current_hostname()
                self.assertFalse(m.return_value.write.called, msg='%s called write, should not have' % str(cls))

    def test_all_named_strategies_exist(self):
        if False:
            for i in range(10):
                print('nop')
        'Loop through the STRATS and see if anything is missing.'
        for (_name, prefix) in hostname.STRATS.items():
            classname = '%sStrategy' % prefix
            cls = getattr(hostname, classname, None)
            assert cls is not None
            self.assertTrue(issubclass(cls, hostname.BaseStrategy))

class TestRedhatStrategy(ModuleTestCase):

    def setUp(self):
        if False:
            return 10
        super(TestRedhatStrategy, self).setUp()
        self.testdir = tempfile.mkdtemp(prefix='ansible-test-hostname-')
        self.network_file = os.path.join(self.testdir, 'network')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestRedhatStrategy, self).tearDown()
        shutil.rmtree(self.testdir, ignore_errors=True)

    @property
    def instance(self):
        if False:
            i = 10
            return i + 15
        self.module = MagicMock()
        instance = hostname.RedHatStrategy(self.module)
        instance.NETWORK_FILE = self.network_file
        return instance

    def test_get_permanent_hostname_missing(self):
        if False:
            return 10
        self.assertIsNone(self.instance.get_permanent_hostname())
        self.assertTrue(self.module.fail_json.called)
        self.module.fail_json.assert_called_with('Unable to locate HOSTNAME entry in %s' % self.network_file)

    def test_get_permanent_hostname_line_missing(self):
        if False:
            i = 10
            return i + 15
        with open(self.network_file, 'w') as f:
            f.write('# some other content\n')
        self.assertIsNone(self.instance.get_permanent_hostname())
        self.module.fail_json.assert_called_with('Unable to locate HOSTNAME entry in %s' % self.network_file)

    def test_get_permanent_hostname_existing(self):
        if False:
            for i in range(10):
                print('nop')
        with open(self.network_file, 'w') as f:
            f.write('some other content\nHOSTNAME=foobar\nmore content\n')
        self.assertEqual(self.instance.get_permanent_hostname(), 'foobar')

    def test_get_permanent_hostname_existing_whitespace(self):
        if False:
            i = 10
            return i + 15
        with open(self.network_file, 'w') as f:
            f.write('some other content\n     HOSTNAME=foobar   \nmore content\n')
        self.assertEqual(self.instance.get_permanent_hostname(), 'foobar')

    def test_set_permanent_hostname_missing(self):
        if False:
            print('Hello World!')
        self.instance.set_permanent_hostname('foobar')
        with open(self.network_file) as f:
            self.assertEqual(f.read(), 'HOSTNAME=foobar\n')

    def test_set_permanent_hostname_line_missing(self):
        if False:
            i = 10
            return i + 15
        with open(self.network_file, 'w') as f:
            f.write('# some other content\n')
        self.instance.set_permanent_hostname('foobar')
        with open(self.network_file) as f:
            self.assertEqual(f.read(), '# some other content\nHOSTNAME=foobar\n')

    def test_set_permanent_hostname_existing(self):
        if False:
            while True:
                i = 10
        with open(self.network_file, 'w') as f:
            f.write('some other content\nHOSTNAME=spam\nmore content\n')
        self.instance.set_permanent_hostname('foobar')
        with open(self.network_file) as f:
            self.assertEqual(f.read(), 'some other content\nHOSTNAME=foobar\nmore content\n')

    def test_set_permanent_hostname_existing_whitespace(self):
        if False:
            while True:
                i = 10
        with open(self.network_file, 'w') as f:
            f.write('some other content\n     HOSTNAME=spam   \nmore content\n')
        self.instance.set_permanent_hostname('foobar')
        with open(self.network_file) as f:
            self.assertEqual(f.read(), 'some other content\nHOSTNAME=foobar\nmore content\n')