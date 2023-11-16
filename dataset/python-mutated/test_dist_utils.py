import os
import sys
import mock
import unittest2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_PATH = os.path.join(BASE_DIR, '../../../scripts/')
sys.path.insert(0, SCRIPTS_PATH)
from dist_utils import fetch_requirements
from dist_utils import apply_vagrant_workaround
from dist_utils import get_version_string
__all__ = ['DistUtilsTestCase']
REQUIREMENTS_PATH_1 = os.path.join(BASE_DIR, '../fixtures/requirements-used-for-tests.txt')
REQUIREMENTS_PATH_2 = os.path.join(BASE_DIR, '../../../requirements.txt')
VERSION_FILE_PATH = os.path.join(BASE_DIR, '../fixtures/version_file.py')

class DistUtilsTestCase(unittest2.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(DistUtilsTestCase, self).setUp()
        if 'pip' in sys.modules:
            del sys.modules['pip']

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super(DistUtilsTestCase, self).tearDown()

    def test_get_version_string(self):
        if False:
            i = 10
            return i + 15
        version = get_version_string(VERSION_FILE_PATH)
        self.assertEqual(version, '1.2.3')

    def test_apply_vagrant_workaround(self):
        if False:
            while True:
                i = 10
        with mock.patch('os.link') as _:
            os.environ['USER'] = 'stanley'
            apply_vagrant_workaround()
            self.assertTrue(os.link)
        with mock.patch('os.link') as _:
            os.environ['USER'] = 'vagrant'
            apply_vagrant_workaround()
            self.assertFalse(getattr(os, 'link', None))

    def test_fetch_requirements(self):
        if False:
            return 10
        expected_reqs = ['RandomWords', 'amqp==2.5.1', 'argcomplete', 'bcrypt==3.1.6', 'flex==6.14.0', 'logshipper', 'orquesta', 'st2-auth-backend-flat-file', 'logshipper-editable', 'python_runner', 'SomePackageHq', 'SomePackageSvn', 'gitpython==2.1.11', 'ose-timer==0.7.5', 'oslo.config<1.13,>=1.12.1', 'requests[security]<2.22.0,>=2.21.0', 'retrying==1.3.3', 'zake==0.2.2']
        expected_links = ['git+https://github.com/Kami/logshipper.git@stackstorm_patched#egg=logshipper', 'git+https://github.com/StackStorm/orquesta.git@224c1a589a6007eb0598a62ee99d674e7836d369#egg=orquesta', 'git+https://github.com/StackStorm/st2-auth-backend-flat-file.git@master#egg=st2-auth-backend-flat-file', 'git+https://github.com/Kami/logshipper.git@stackstorm_patched#egg=logshipper-editable', 'git+https://github.com/StackStorm/st2.git#egg=python_runner&subdirectory=contrib/runners/python_runner', 'hg+https://hg.repo/some_pkg.git#egg=SomePackageHq', 'svn+svn://svn.repo/some_pkg/trunk/@ma-branch#egg=SomePackageSvn']
        (reqs, links) = fetch_requirements(REQUIREMENTS_PATH_1)
        self.assertEqual(reqs, expected_reqs)
        self.assertEqual(links, expected_links)
        (reqs, links) = fetch_requirements(REQUIREMENTS_PATH_2)
        self.assertGreater(len(reqs), 0)
        self.assertGreater(len(links), 0)