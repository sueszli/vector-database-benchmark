from __future__ import absolute_import
import st2tests.config as tests_config
tests_config.parse_args()
from unittest2 import TestCase
from st2common.models.system.action import RemoteScriptAction

class RemoteScriptActionTestCase(TestCase):

    def test_parameter_formatting(self):
        if False:
            while True:
                i = 10
        named_args = {'--foo1': 'bar1', '--foo2': 'bar2', '--foo3': True, '--foo4': False}
        action = RemoteScriptAction(name='foo', action_exec_id='dummy', script_local_path_abs='test.py', script_local_libs_path_abs='/', remote_dir='/tmp', named_args=named_args, positional_args=None)
        self.assertEqual(action.command, '/tmp/test.py --foo1=bar1 --foo2=bar2 --foo3')