import os
import shutil
import sys
from subprocess import call
from subprocess import check_call
from textwrap import dedent
from twisted.trial import unittest

class BuildbotWWWPkg(unittest.TestCase):
    pkgName = 'buildbot_www'
    pkgPaths = ['www', 'base']
    epName = 'base'
    loadTestScript = dedent('\n        from importlib.metadata import entry_points\n        import re\n        apps = {}\n        for ep in entry_points().get(\'buildbot.www\', []):\n            apps[ep.name] = ep.load()\n\n        print(apps["%(epName)s"])\n        assert("scripts.js" in apps["%(epName)s"].resource.listNames())\n        assert(re.match(r\'\\d+\\.\\d+\\.\\d+\', apps["%(epName)s"].version) is not None)\n        assert(apps["%(epName)s"].description is not None)\n        ')

    @property
    def path(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', *self.pkgPaths))

    def rmtree(self, d):
        if False:
            i = 10
            return i + 15
        if os.path.isdir(d):
            shutil.rmtree(d)

    def setUp(self):
        if False:
            while True:
                i = 10
        call('pip uninstall -y ' + self.pkgName, shell=True)
        self.rmtree(os.path.join(self.path, 'build'))
        self.rmtree(os.path.join(self.path, 'dist'))
        self.rmtree(os.path.join(self.path, 'static'))

    def run_setup(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        check_call([sys.executable, 'setup.py', cmd], cwd=self.path)

    def check_correct_installation(self):
        if False:
            return 10
        check_call([sys.executable, '-c', self.loadTestScript % dict(epName=self.epName)])

    def test_install(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_setup('install')
        self.check_correct_installation()

    def test_wheel(self):
        if False:
            return 10
        self.run_setup('bdist_wheel')
        check_call('pip install dist/*.whl', shell=True, cwd=self.path)
        self.check_correct_installation()

    def test_develop(self):
        if False:
            return 10
        self.run_setup('develop')
        self.check_correct_installation()

    def test_develop_via_pip(self):
        if False:
            i = 10
            return i + 15
        check_call('pip install -e .', shell=True, cwd=self.path)
        self.check_correct_installation()

    def test_sdist(self):
        if False:
            while True:
                i = 10
        self.run_setup('sdist')
        check_call('pip install dist/*.tar.gz', shell=True, cwd=self.path)
        self.check_correct_installation()

class BuildbotConsolePkg(BuildbotWWWPkg):
    pkgName = 'buildbot-console-view'
    pkgPaths = ['www', 'console_view']
    epName = 'console_view'

class BuildbotWaterfallPkg(BuildbotWWWPkg):
    pkgName = 'buildbot-waterfall-view'
    pkgPaths = ['www', 'waterfall_view']
    epName = 'waterfall_view'

class BuildbotCodeparameterPkg(BuildbotWWWPkg):
    pkgName = 'buildbot-codeparameter'
    pkgPaths = ['www', 'codeparameter']
    epName = 'codeparameter'