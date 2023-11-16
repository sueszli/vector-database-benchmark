"""
Tests for runner_returns
"""
import errno
import os
import socket
import tempfile
import pytest
import salt.payload
import salt.utils.args
import salt.utils.files
import salt.utils.jid
import salt.utils.yaml
from tests.support.case import ShellCase
from tests.support.runtests import RUNTIME_VARS

@pytest.mark.windows_whitelisted
class RunnerReturnsTest(ShellCase):
    """
    Test the "runner_returns" feature
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create the temp file and master.d directory\n        '
        self.job_dir = os.path.join(self.master_opts['cachedir'], 'jobs')
        self.hash_type = self.master_opts['hash_type']
        self.master_d_dir = os.path.join(self.config_dir, 'master.d')
        try:
            os.makedirs(self.master_d_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        self.conf = tempfile.NamedTemporaryFile(mode='w', suffix='.conf', dir=self.master_d_dir, delete=False)

    def tearDown(self):
        if False:
            print('Hello World!')
        '\n        Close the tempfile.NamedTemporaryFile object, cleaning it up\n        '
        try:
            self.conf.close()
        except OSError:
            pass
        salt.utils.files.rm_rf(self.master_d_dir)
        self.run_run_plus('test.arg')

    @staticmethod
    def clean_return(data):
        if False:
            print('Hello World!')
        '\n        Remove kwargs and timestamp (things that are variable) so we have a\n        stable value to assert\n        '
        data['fun_args'][1] = salt.utils.args.clean_kwargs(**data['fun_args'][1])
        data['return']['kwargs'] = salt.utils.args.clean_kwargs(**data['return']['kwargs'])
        data.pop('_stamp')

    def write_conf(self, data):
        if False:
            while True:
                i = 10
        '\n        Dump the config dict to the conf file\n        '
        self.conf.write(salt.utils.yaml.safe_dump(data, default_flow_style=False))
        self.conf.flush()
        self.conf.close()

    @pytest.mark.slow_test
    def test_runner_returns_disabled(self):
        if False:
            i = 10
            return i + 15
        '\n        Test with runner_returns disabled\n        '
        self.write_conf({'runner_returns': False})
        ret = self.run_run_plus('test.arg', 'foo', bar='hello world!')
        jid = ret.get('jid')
        if jid is None:
            raise Exception('jid missing from run_run_plus output')
        serialized_return = os.path.join(salt.utils.jid.jid_dir(jid, self.job_dir, self.hash_type), 'master', 'return.p')
        self.assertFalse(os.path.isfile(serialized_return))

    @pytest.mark.slow_test
    def test_runner_returns_enabled(self):
        if False:
            print('Hello World!')
        '\n        Test with runner_returns enabled\n        '
        self.write_conf({'runner_returns': True})
        ret = self.run_run_plus('test.arg', 'foo', bar='hello world!')
        jid = ret.get('jid')
        if jid is None:
            raise Exception('jid missing from run_run_plus output')
        serialized_return = os.path.join(salt.utils.jid.jid_dir(jid, self.job_dir, self.hash_type), 'master', 'return.p')
        with salt.utils.files.fopen(serialized_return, 'rb') as fp_:
            deserialized = salt.payload.loads(fp_.read(), encoding='utf-8')
        self.clean_return(deserialized['return'])
        if 'SUDO_USER' in os.environ:
            user = 'sudo_{}'.format(os.environ['SUDO_USER'])
        else:
            user = RUNTIME_VARS.RUNNING_TESTS_USER
        if salt.utils.platform.is_windows():
            user = 'sudo_{}\\{}'.format(socket.gethostname(), user)
        self.assertEqual(deserialized, {'return': {'fun': 'runner.test.arg', 'fun_args': ['foo', {'bar': 'hello world!'}], 'jid': jid, 'return': {'args': ['foo'], 'kwargs': {'bar': 'hello world!'}}, 'success': True, 'user': user}})