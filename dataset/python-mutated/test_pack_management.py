from __future__ import absolute_import
import os
import sys
import unittest2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACK_ACTIONS_DIR = os.path.join(BASE_DIR, '../../../contrib/packs/actions')
PACK_ACTIONS_DIR = os.path.abspath(PACK_ACTIONS_DIR)
sys.path.insert(0, PACK_ACTIONS_DIR)
from st2common.util.monkey_patch import use_select_poll_workaround
use_select_poll_workaround()
from st2common.util.pack_management import eval_repo_url
__all__ = ['InstallPackTestCase']

class InstallPackTestCase(unittest2.TestCase):

    def test_eval_repo(self):
        if False:
            return 10
        result = eval_repo_url('stackstorm/st2contrib')
        self.assertEqual(result, 'https://github.com/stackstorm/st2contrib')
        result = eval_repo_url('git@github.com:StackStorm/st2contrib.git')
        self.assertEqual(result, 'git@github.com:StackStorm/st2contrib.git')
        result = eval_repo_url('gitlab@gitlab.com:StackStorm/st2contrib.git')
        self.assertEqual(result, 'gitlab@gitlab.com:StackStorm/st2contrib.git')
        repo_url = 'https://github.com/StackStorm/st2contrib.git'
        result = eval_repo_url(repo_url)
        self.assertEqual(result, repo_url)
        repo_url = 'https://git-wip-us.apache.org/repos/asf/libcloud.git'
        result = eval_repo_url(repo_url)
        self.assertEqual(result, repo_url)