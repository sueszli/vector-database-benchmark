import os
import pytest
import salt.utils.files
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
            i = 10
            return i + 15
        '\n        Test cli function\n        '
        cmd_iter = self.client.cmd_cli('minion', 'test.ping', timeout=20)
        for ret in cmd_iter:
            self.assertTrue(ret['minion'])
        cmd_iter = self.client.cmd_cli('minion', 'test.sleep', [6], timeout=20)
        num_ret = 0
        for ret in cmd_iter:
            num_ret += 1
            self.assertTrue(ret['minion'])
        assert num_ret > 0
        key_file = os.path.join(self.master_opts['pki_dir'], 'minions', 'footest')
        with salt.utils.files.fopen(key_file, 'a'):
            pass
        try:
            cmd_iter = self.client.cmd_cli('footest', 'test.ping', timeout=20)
            num_ret = 0
            for ret in cmd_iter:
                num_ret += 1
                self.assertTrue(ret['minion'])
            assert num_ret == 0
        finally:
            os.unlink(key_file)

    @pytest.mark.slow_test
    def test_iter(self):
        if False:
            return 10
        '\n        test cmd_iter\n        '
        cmd_iter = self.client.cmd_iter('minion', 'test.ping')
        for ret in cmd_iter:
            self.assertTrue(ret['minion'])

    @pytest.mark.slow_test
    def test_iter_no_block(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test cmd_iter_no_block\n        '
        cmd_iter = self.client.cmd_iter_no_block('minion', 'test.ping')
        for ret in cmd_iter:
            if ret is None:
                continue
            self.assertTrue(ret['minion'])

    @pytest.mark.slow_test
    def test_batch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        test cmd_batch\n        '
        cmd_batch = self.client.cmd_batch('minion', 'test.ping')
        for ret in cmd_batch:
            self.assertTrue(ret['minion'])

    @pytest.mark.slow_test
    def test_batch_raw(self):
        if False:
            print('Hello World!')
        '\n        test cmd_batch with raw option\n        '
        cmd_batch = self.client.cmd_batch('minion', 'test.ping', raw=True)
        for ret in cmd_batch:
            self.assertTrue(ret['data']['success'])

    @pytest.mark.slow_test
    def test_full_returns(self):
        if False:
            print('Hello World!')
        '\n        test cmd_iter\n        '
        ret = self.client.cmd_full_return('minion', 'test.ping', timeout=20)
        self.assertIn('minion', ret)
        self.assertEqual({'ret': True, 'success': True}, ret['minion'])

    @pytest.mark.slow_test
    def test_disconnected_return(self):
        if False:
            return 10
        '\n        Test return/messaging on a disconnected minion\n        '
        test_ret = 'Minion did not return. [No response]'
        test_out = 'no_return'
        key_file = os.path.join(self.master_opts['pki_dir'], 'minions', 'disconnected')
        with salt.utils.files.fopen(key_file, 'a'):
            pass
        try:
            cmd_iter = self.client.cmd_cli('disconnected', 'test.ping', show_timeout=True)
            num_ret = 0
            for ret in cmd_iter:
                num_ret += 1
                assert ret['disconnected']['ret'].startswith(test_ret), ret['disconnected']['ret']
                assert ret['disconnected']['out'] == test_out, ret['disconnected']['out']
            self.assertEqual(num_ret, 1)
        finally:
            os.unlink(key_file)

    @pytest.mark.slow_test
    def test_missing_minion_list(self):
        if False:
            i = 10
            return i + 15
        '\n        test cmd with missing minion in nodegroup\n        '
        ret = self.client.cmd('minion,ghostminion', 'test.ping', tgt_type='list')
        assert 'minion' in ret
        assert 'ghostminion' in ret
        assert ret['minion'] is True
        assert ret['ghostminion'].startswith('Minion did not return. [No response]'), ret['ghostminion']

    @pytest.mark.slow_test
    def test_missing_minion_nodegroup(self):
        if False:
            i = 10
            return i + 15
        '\n        test cmd with missing minion in nodegroup\n        '
        ret = self.client.cmd('missing_minion', 'test.ping', tgt_type='nodegroup')
        assert 'minion' in ret
        assert 'ghostminion' in ret
        assert ret['minion'] is True
        assert ret['ghostminion'].startswith('Minion did not return. [No response]'), ret['ghostminion']