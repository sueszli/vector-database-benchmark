from __future__ import absolute_import
import os
import sys
import unittest
import mock
from oslo_config import cfg
from st2common.constants.pack import SYSTEM_PACK_NAMES
from st2common.util.sandboxing import get_sandbox_path
from st2common.util.sandboxing import get_sandbox_python_path
from st2common.util.sandboxing import get_sandbox_python_path_for_python_action
from st2common.util.sandboxing import get_sandbox_python_binary_path
from st2common.util.sandboxing import clear_virtualenv_prefix
from st2common.util.sandboxing import get_virtualenv_prefix
from st2common.util.sandboxing import set_virtualenv_prefix
import st2tests.config as tests_config
__all__ = ['SandboxingUtilsTestCase']

class SandboxingUtilsTestCase(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        if False:
            return 10
        super(SandboxingUtilsTestCase, self).setUp()
        set_virtualenv_prefix(self.old_virtualenv_prefix)

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        tests_config.parse_args()
        cls.old_virtualenv_prefix = get_virtualenv_prefix()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        set_virtualenv_prefix(cls.old_virtualenv_prefix)

    def assertEndsWith(self, string, ending_substr, msg=None):
        if False:
            while True:
                i = 10
        msg = msg or "'{string}'' does not end with '{ending_substr}'"
        try:
            assert string.endswith(ending_substr) is True
        except AssertionError as e:
            print(dir(e))
            print(e.args)
            e.args = (msg.format(string=string, ending_substr=ending_substr),)
            raise e

    def test_get_sandbox_python_binary_path(self):
        if False:
            return 10
        result = get_sandbox_python_binary_path(pack='mapack')
        expected = os.path.join(cfg.CONF.system.base_path, 'virtualenvs/mapack/bin/python')
        self.assertEqual(result, expected)
        result = get_sandbox_python_binary_path(pack=SYSTEM_PACK_NAMES[0])
        self.assertEqual(result, sys.executable)

    def test_get_sandbox_path(self):
        if False:
            return 10
        virtualenv_path = '/home/venv/test'
        with mock.patch.dict(os.environ, {'PATH': '/home/path1:/home/path2:/home/path3:'}):
            result = get_sandbox_path(virtualenv_path=virtualenv_path)
        self.assertEqual(result, f'{virtualenv_path}/bin/:/home/path1:/home/path2:/home/path3')

    @mock.patch('st2common.util.sandboxing.get_python_lib')
    def test_get_sandbox_python_path(self, mock_get_python_lib):
        if False:
            print('Hello World!')
        python_path = get_sandbox_python_path(inherit_from_parent=False, inherit_parent_virtualenv=False)
        self.assertEqual(python_path, ':')
        with mock.patch.dict(os.environ, {'PYTHONPATH': ':/data/test1:/data/test2'}):
            python_path = get_sandbox_python_path(inherit_from_parent=True, inherit_parent_virtualenv=False)
        self.assertEqual(python_path, ':/data/test1:/data/test2')
        clear_virtualenv_prefix()
        with mock.patch.dict(os.environ, {'PYTHONPATH': ':/data/test1:/data/test2'}):
            python_path = get_sandbox_python_path(inherit_from_parent=True, inherit_parent_virtualenv=False)
        self.assertEqual(python_path, ':/data/test1:/data/test2')
        sys.real_prefix = '/usr'
        mock_get_python_lib.return_value = f'{sys.prefix}/virtualenvtest'
        with mock.patch.dict(os.environ, {'PYTHONPATH': ':/data/test1:/data/test2'}):
            python_path = get_sandbox_python_path(inherit_from_parent=True, inherit_parent_virtualenv=True)
        self.assertEqual(python_path, f':/data/test1:/data/test2:{sys.prefix}/virtualenvtest')

    @mock.patch('os.path.isdir', mock.Mock(return_value=True))
    @mock.patch('os.listdir', mock.Mock(return_value=['python3.6']))
    @mock.patch('st2common.util.sandboxing.get_python_lib')
    def test_get_sandbox_python_path_for_python_action_no_inheritance(self, mock_get_python_lib):
        if False:
            for i in range(10):
                print('nop')
        python_path = get_sandbox_python_path_for_python_action(pack='dummy_pack', inherit_from_parent=False, inherit_parent_virtualenv=False)
        actual_path = python_path.strip(':').split(':')
        self.assertEqual(len(actual_path), 3)
        self.assertEndsWith(actual_path[0], 'virtualenvs/dummy_pack/lib/python3.6')
        self.assertEndsWith(actual_path[1], 'virtualenvs/dummy_pack/lib/python3.6/site-packages')
        self.assertEndsWith(actual_path[2], 'packs/dummy_pack/actions/lib')

    @mock.patch('os.path.isdir', mock.Mock(return_value=True))
    @mock.patch('os.listdir', mock.Mock(return_value=['python3.6']))
    @mock.patch('st2common.util.sandboxing.get_python_lib')
    def test_get_sandbox_python_path_for_python_action_inherit_from_parent_process_only(self, mock_get_python_lib):
        if False:
            i = 10
            return i + 15
        with mock.patch.dict(os.environ, {'PYTHONPATH': ':/data/test1:/data/test2'}):
            python_path = get_sandbox_python_path(inherit_from_parent=True, inherit_parent_virtualenv=False)
            self.assertEqual(python_path, ':/data/test1:/data/test2')
            python_path = get_sandbox_python_path_for_python_action(pack='dummy_pack', inherit_from_parent=True, inherit_parent_virtualenv=False)
        actual_path = python_path.strip(':').split(':')
        self.assertEqual(len(actual_path), 6)
        self.assertEndsWith(actual_path[0], 'virtualenvs/dummy_pack/lib/python3.6')
        self.assertEndsWith(actual_path[1], 'virtualenvs/dummy_pack/lib/python3.6/site-packages')
        self.assertEndsWith(actual_path[2], 'packs/dummy_pack/actions/lib')
        self.assertEqual(actual_path[3], '')
        self.assertEqual(actual_path[4], '/data/test1')
        self.assertEqual(actual_path[5], '/data/test2')

    @mock.patch('os.path.isdir', mock.Mock(return_value=True))
    @mock.patch('os.listdir', mock.Mock(return_value=['python3.6']))
    @mock.patch('st2common.util.sandboxing.get_python_lib')
    def test_get_sandbox_python_path_for_python_action_inherit_from_parent_process_and_venv(self, mock_get_python_lib):
        if False:
            print('Hello World!')
        clear_virtualenv_prefix()
        with mock.patch.dict(os.environ, {'PYTHONPATH': ':/data/test1:/data/test2'}):
            python_path = get_sandbox_python_path(inherit_from_parent=True, inherit_parent_virtualenv=False)
            self.assertEqual(python_path, ':/data/test1:/data/test2')
            python_path = get_sandbox_python_path_for_python_action(pack='dummy_pack', inherit_from_parent=True, inherit_parent_virtualenv=True)
        actual_path = python_path.strip(':').split(':')
        self.assertEqual(len(actual_path), 6)
        self.assertEndsWith(actual_path[0], 'virtualenvs/dummy_pack/lib/python3.6')
        self.assertEndsWith(actual_path[1], 'virtualenvs/dummy_pack/lib/python3.6/site-packages')
        self.assertEndsWith(actual_path[2], 'packs/dummy_pack/actions/lib')
        self.assertEqual(actual_path[3], '')
        self.assertEqual(actual_path[4], '/data/test1')
        self.assertEqual(actual_path[5], '/data/test2')
        sys.real_prefix = '/usr'
        mock_get_python_lib.return_value = f'{sys.prefix}/virtualenvtest'
        with mock.patch.dict(os.environ, {'PYTHONPATH': ':/data/test1:/data/test2'}):
            python_path = get_sandbox_python_path_for_python_action(pack='dummy_pack', inherit_from_parent=True, inherit_parent_virtualenv=True)
        actual_path = python_path.strip(':').split(':')
        self.assertEqual(len(actual_path), 7)
        self.assertEndsWith(actual_path[0], 'virtualenvs/dummy_pack/lib/python3.6')
        self.assertEndsWith(actual_path[1], 'virtualenvs/dummy_pack/lib/python3.6/site-packages')
        self.assertEndsWith(actual_path[2], 'packs/dummy_pack/actions/lib')
        self.assertEqual(actual_path[3], '')
        self.assertEqual(actual_path[4], '/data/test1')
        self.assertEqual(actual_path[5], '/data/test2')
        self.assertEqual(actual_path[6], f'{sys.prefix}/virtualenvtest')