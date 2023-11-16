import pytest
import salt.utils.platform
from tests.support.case import ModuleCase

@pytest.mark.windows_whitelisted
class StdTest(ModuleCase):
    """
    Test standard client calls
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.TIMEOUT = 600 if salt.utils.platform.is_windows() else 10

    @pytest.mark.slow_test
    def test_cli(self):
        if False:
            print('Hello World!')
        '\n        Test cli function\n        '
        cmd_iter = self.client.cmd_cli('minion', 'test.arg', ['foo', 'bar', 'baz'], kwarg={'qux': 'quux'})
        for ret in cmd_iter:
            data = ret['minion']['ret']
            self.assertEqual(data['args'], ['foo', 'bar', 'baz'])
            self.assertEqual(data['kwargs']['qux'], 'quux')

    @pytest.mark.slow_test
    def test_iter(self):
        if False:
            while True:
                i = 10
        '\n        test cmd_iter\n        '
        cmd_iter = self.client.cmd_iter('minion', 'test.arg', ['foo', 'bar', 'baz'], kwarg={'qux': 'quux'})
        for ret in cmd_iter:
            data = ret['minion']['ret']
            self.assertEqual(data['args'], ['foo', 'bar', 'baz'])
            self.assertEqual(data['kwargs']['qux'], 'quux')

    @pytest.mark.slow_test
    def test_iter_no_block(self):
        if False:
            print('Hello World!')
        '\n        test cmd_iter_no_block\n        '
        cmd_iter = self.client.cmd_iter_no_block('minion', 'test.arg', ['foo', 'bar', 'baz'], kwarg={'qux': 'quux'})
        for ret in cmd_iter:
            if ret is None:
                continue
            data = ret['minion']['ret']
            self.assertEqual(data['args'], ['foo', 'bar', 'baz'])
            self.assertEqual(data['kwargs']['qux'], 'quux')

    @pytest.mark.slow_test
    def test_full_returns(self):
        if False:
            print('Hello World!')
        '\n        test cmd_iter\n        '
        ret = self.client.cmd_full_return('minion', 'test.arg', ['foo', 'bar', 'baz'], timeout=self.TIMEOUT, kwarg={'qux': 'quux'})
        data = ret['minion']['ret']
        self.assertEqual(data['args'], ['foo', 'bar', 'baz'])
        self.assertEqual(data['kwargs']['qux'], 'quux')

    @pytest.mark.slow_test
    def test_kwarg_type(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that kwargs end up on the client as the same type\n        '
        terrible_yaml_string = 'foo: ""\n# \''
        ret = self.client.cmd_full_return('minion', 'test.arg_type', ['a', 1], kwarg={'outer': {'a': terrible_yaml_string}, 'inner': 'value'}, timeout=self.TIMEOUT)
        data = ret['minion']['ret']
        self.assertIn(str.__name__, data['args'][0])
        self.assertIn('int', data['args'][1])
        self.assertIn('dict', data['kwargs']['outer'])
        self.assertIn(str.__name__, data['kwargs']['inner'])

    @pytest.mark.slow_test
    def test_full_return_kwarg(self):
        if False:
            while True:
                i = 10
        ret = self.client.cmd('minion', 'test.ping', full_return=True, timeout=self.TIMEOUT)
        for (mid, data) in ret.items():
            self.assertIn('retcode', data)

    @pytest.mark.slow_test
    def test_cmd_arg_kwarg_parsing(self):
        if False:
            while True:
                i = 10
        ret = self.client.cmd('minion', 'test.arg_clean', arg=['foo', 'bar=off', 'baz={qux: 123}'], kwarg={'quux': 'Quux'}, timeout=self.TIMEOUT)
        self.assertEqual(ret['minion'], {'args': ['foo'], 'kwargs': {'bar': False, 'baz': {'qux': 123}, 'quux': 'Quux'}})