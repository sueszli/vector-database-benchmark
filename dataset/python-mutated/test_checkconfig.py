import os
import re
import sys
import textwrap
from io import StringIO
from unittest import mock
from twisted.trial import unittest
from buildbot.scripts import base
from buildbot.scripts import checkconfig
from buildbot.test.util import dirs

class TestConfigLoader(dirs.DirsMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.configdir = self.mktemp()
        return self.setUpDirs(self.configdir)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tearDownDirs()

    def do_test_load(self, config='', other_files=None, stdout_re=None, stderr_re=None):
        if False:
            for i in range(10):
                print('nop')
        if other_files is None:
            other_files = {}
        configFile = os.path.join(self.configdir, 'master.cfg')
        with open(configFile, 'w', encoding='utf-8') as f:
            f.write(config)
        for (filename, contents) in other_files.items():
            if isinstance(filename, type(())):
                fn = os.path.join(self.configdir, *filename)
                dn = os.path.dirname(fn)
                if not os.path.isdir(dn):
                    os.makedirs(dn)
            else:
                fn = os.path.join(self.configdir, filename)
            with open(fn, 'w', encoding='utf-8') as f:
                f.write(contents)
        (old_stdout, old_stderr) = (sys.stdout, sys.stderr)
        stdout = sys.stdout = StringIO()
        stderr = sys.stderr = StringIO()
        try:
            checkconfig._loadConfig(basedir=self.configdir, configFile='master.cfg', quiet=False)
        finally:
            (sys.stdout, sys.stderr) = (old_stdout, old_stderr)
        if stdout_re:
            stdout = stdout.getvalue()
            self.assertTrue(stdout_re.search(stdout), stdout)
        if stderr_re:
            stderr = stderr.getvalue()
            self.assertTrue(stderr_re.search(stderr), stderr)

    def test_success(self):
        if False:
            return 10
        len_sys_path = len(sys.path)
        config = textwrap.dedent("                c = BuildmasterConfig = {}\n                c['multiMaster'] = True\n                c['schedulers'] = []\n                from buildbot.config import BuilderConfig\n                from buildbot.process.factory import BuildFactory\n                c['builders'] = [\n                    BuilderConfig('testbuilder', factory=BuildFactory(),\n                                  workername='worker'),\n                ]\n                from buildbot.worker import Worker\n                c['workers'] = [\n                    Worker('worker', 'pass'),\n                ]\n                c['protocols'] = {'pb': {'port': 9989}}\n                ")
        self.do_test_load(config=config, stdout_re=re.compile('Config file is good!'))
        self.assertEqual(len(sys.path), len_sys_path)

    def test_failure_ImportError(self):
        if False:
            print('Hello World!')
        config = textwrap.dedent('                import test_scripts_checkconfig_does_not_exist\n                ')
        self.do_test_load(config=config, stderr_re=re.compile("No module named '?test_scripts_checkconfig_does_not_exist'?"))
        self.flushLoggedErrors()

    def test_failure_no_workers(self):
        if False:
            return 10
        config = textwrap.dedent('                BuildmasterConfig={}\n                ')
        self.do_test_load(config=config, stderr_re=re.compile('no workers'))
        self.flushLoggedErrors()

    def test_success_imports(self):
        if False:
            print('Hello World!')
        config = textwrap.dedent("                from othermodule import port\n                c = BuildmasterConfig = {}\n                c['schedulers'] = []\n                c['builders'] = []\n                c['workers'] = []\n                c['protocols'] = {'pb': {'port': port}}\n                ")
        other_files = {'othermodule.py': 'port = 9989'}
        self.do_test_load(config=config, other_files=other_files)

    def test_success_import_package(self):
        if False:
            return 10
        config = textwrap.dedent("                from otherpackage.othermodule import port\n                c = BuildmasterConfig = {}\n                c['schedulers'] = []\n                c['builders'] = []\n                c['workers'] = []\n                c['protocols'] = {'pb': {'port': 9989}}\n                ")
        other_files = {('otherpackage', '__init__.py'): '', ('otherpackage', 'othermodule.py'): 'port = 9989'}
        self.do_test_load(config=config, other_files=other_files)

class TestCheckconfig(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.loadConfig = mock.Mock(spec=checkconfig._loadConfig, return_value=3)
        self.patch(checkconfig, 'checkconfig', checkconfig.checkconfig._orig)
        self.patch(checkconfig, '_loadConfig', self.loadConfig)

    def test_checkconfig_default(self):
        if False:
            print('Hello World!')
        self.assertEqual(checkconfig.checkconfig({}), 3)
        self.loadConfig.assert_called_with(basedir=os.getcwd(), configFile='master.cfg', quiet=None)

    def test_checkconfig_given_dir(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(checkconfig.checkconfig({'configFile': '.'}), 3)
        self.loadConfig.assert_called_with(basedir='.', configFile='master.cfg', quiet=None)

    def test_checkconfig_given_file(self):
        if False:
            return 10
        config = {'configFile': 'master.cfg'}
        self.assertEqual(checkconfig.checkconfig(config), 3)
        self.loadConfig.assert_called_with(basedir=os.getcwd(), configFile='master.cfg', quiet=None)

    def test_checkconfig_quiet(self):
        if False:
            i = 10
            return i + 15
        config = {'configFile': 'master.cfg', 'quiet': True}
        self.assertEqual(checkconfig.checkconfig(config), 3)
        self.loadConfig.assert_called_with(basedir=os.getcwd(), configFile='master.cfg', quiet=True)

    def test_checkconfig_syntaxError_quiet(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When C{base.getConfigFileFromTac} raises L{SyntaxError},\n        C{checkconfig.checkconfig} return an error.\n        '
        mockGetConfig = mock.Mock(spec=base.getConfigFileFromTac, side_effect=SyntaxError)
        self.patch(checkconfig, 'getConfigFileFromTac', mockGetConfig)
        config = {'configFile': '.', 'quiet': True}
        self.assertEqual(checkconfig.checkconfig(config), 1)