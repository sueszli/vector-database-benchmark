import os
import sys
from io import StringIO
from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.config import master as config_master
from buildbot.db import connector
from buildbot.db import masters
from buildbot.db import model
from buildbot.scripts import base
from buildbot.scripts import upgrade_master
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import dirs
from buildbot.test.util import misc
from buildbot.test.util import www

def mkconfig(**kwargs):
    if False:
        i = 10
        return i + 15
    config = {'quiet': False, 'replace': False, 'basedir': 'test'}
    config.update(kwargs)
    return config

class TestUpgradeMaster(dirs.DirsMixin, misc.StdoutAssertionsMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch(upgrade_master, 'upgradeMaster', upgrade_master.upgradeMaster._orig)
        self.setUpDirs('test')
        self.setUpStdoutAssertions()

    def patchFunctions(self, basedirOk=True, configOk=True):
        if False:
            i = 10
            return i + 15
        self.calls = []

        def checkBasedir(config):
            if False:
                i = 10
                return i + 15
            self.calls.append('checkBasedir')
            return basedirOk
        self.patch(base, 'checkBasedir', checkBasedir)

        def loadConfig(config, configFileName='master.cfg'):
            if False:
                return 10
            self.calls.append('loadConfig')
            return config_master.MasterConfig() if configOk else False
        self.patch(base, 'loadConfig', loadConfig)

        def upgradeFiles(config):
            if False:
                return 10
            self.calls.append('upgradeFiles')
        self.patch(upgrade_master, 'upgradeFiles', upgradeFiles)

        def upgradeDatabase(config, master_cfg):
            if False:
                print('Hello World!')
            self.assertIsInstance(master_cfg, config_master.MasterConfig)
            self.calls.append('upgradeDatabase')
        self.patch(upgrade_master, 'upgradeDatabase', upgradeDatabase)

    @defer.inlineCallbacks
    def test_upgradeMaster_success(self):
        if False:
            i = 10
            return i + 15
        self.patchFunctions()
        rv = (yield upgrade_master.upgradeMaster(mkconfig()))
        self.assertEqual(rv, 0)
        self.assertInStdout('upgrade complete')

    @defer.inlineCallbacks
    def test_upgradeMaster_quiet(self):
        if False:
            for i in range(10):
                print('nop')
        self.patchFunctions()
        rv = (yield upgrade_master.upgradeMaster(mkconfig(quiet=True)))
        self.assertEqual(rv, 0)
        self.assertWasQuiet()

    @defer.inlineCallbacks
    def test_upgradeMaster_bad_basedir(self):
        if False:
            for i in range(10):
                print('nop')
        self.patchFunctions(basedirOk=False)
        rv = (yield upgrade_master.upgradeMaster(mkconfig()))
        self.assertEqual(rv, 1)

    @defer.inlineCallbacks
    def test_upgradeMaster_bad_config(self):
        if False:
            return 10
        self.patchFunctions(configOk=False)
        rv = (yield upgrade_master.upgradeMaster(mkconfig()))
        self.assertEqual(rv, 1)

class TestUpgradeMasterFunctions(www.WwwTestMixin, dirs.DirsMixin, misc.StdoutAssertionsMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        self.setUpDirs('test')
        self.basedir = os.path.abspath(os.path.join('test', 'basedir'))
        self.setUpStdoutAssertions()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tearDownDirs()

    def writeFile(self, path, contents):
        if False:
            print('Hello World!')
        with open(path, 'wt', encoding='utf-8') as f:
            f.write(contents)

    def readFile(self, path):
        if False:
            print('Hello World!')
        with open(path, 'rt', encoding='utf-8') as f:
            return f.read()

    def test_installFile(self):
        if False:
            print('Hello World!')
        self.writeFile('test/srcfile', 'source data')
        upgrade_master.installFile(mkconfig(), 'test/destfile', 'test/srcfile')
        self.assertEqual(self.readFile('test/destfile'), 'source data')
        self.assertInStdout('creating test/destfile')

    def test_installFile_existing_differing(self):
        if False:
            for i in range(10):
                print('nop')
        self.writeFile('test/srcfile', 'source data')
        self.writeFile('test/destfile', 'dest data')
        upgrade_master.installFile(mkconfig(), 'test/destfile', 'test/srcfile')
        self.assertEqual(self.readFile('test/destfile'), 'dest data')
        self.assertEqual(self.readFile('test/destfile.new'), 'source data')
        self.assertInStdout('writing new contents to')

    def test_installFile_existing_differing_overwrite(self):
        if False:
            i = 10
            return i + 15
        self.writeFile('test/srcfile', 'source data')
        self.writeFile('test/destfile', 'dest data')
        upgrade_master.installFile(mkconfig(), 'test/destfile', 'test/srcfile', overwrite=True)
        self.assertEqual(self.readFile('test/destfile'), 'source data')
        self.assertFalse(os.path.exists('test/destfile.new'))
        self.assertInStdout('overwriting')

    def test_installFile_existing_same(self):
        if False:
            while True:
                i = 10
        self.writeFile('test/srcfile', 'source data')
        self.writeFile('test/destfile', 'source data')
        upgrade_master.installFile(mkconfig(), 'test/destfile', 'test/srcfile')
        self.assertEqual(self.readFile('test/destfile'), 'source data')
        self.assertFalse(os.path.exists('test/destfile.new'))
        self.assertWasQuiet()

    def test_installFile_quiet(self):
        if False:
            print('Hello World!')
        self.writeFile('test/srcfile', 'source data')
        upgrade_master.installFile(mkconfig(quiet=True), 'test/destfile', 'test/srcfile')
        self.assertWasQuiet()

    def test_upgradeFiles(self):
        if False:
            return 10
        upgrade_master.upgradeFiles(mkconfig())
        for f in ['test/master.cfg.sample']:
            self.assertTrue(os.path.exists(f), f'{f} not found')
        self.assertInStdout('upgrading basedir')

    def test_upgradeFiles_notice_about_unused_public_html(self):
        if False:
            return 10
        os.mkdir('test/public_html')
        self.writeFile('test/public_html/index.html', 'INDEX')
        upgrade_master.upgradeFiles(mkconfig())
        self.assertInStdout('public_html is not used')

    @defer.inlineCallbacks
    def test_upgradeDatabase(self):
        if False:
            print('Hello World!')
        setup = mock.Mock(side_effect=lambda **kwargs: defer.succeed(None))
        self.patch(connector.DBConnector, 'setup', setup)
        upgrade = mock.Mock(side_effect=lambda **kwargs: defer.succeed(None))
        self.patch(model.Model, 'upgrade', upgrade)
        setAllMastersActiveLongTimeAgo = mock.Mock(side_effect=lambda **kwargs: defer.succeed(None))
        self.patch(masters.MastersConnectorComponent, 'setAllMastersActiveLongTimeAgo', setAllMastersActiveLongTimeAgo)
        yield upgrade_master.upgradeDatabase(mkconfig(basedir='test', quiet=True), config_master.MasterConfig())
        setup.asset_called_with(check_version=False, verbose=False)
        upgrade.assert_called_with()
        self.assertWasQuiet()

    @defer.inlineCallbacks
    def test_upgradeDatabaseFail(self):
        if False:
            i = 10
            return i + 15
        setup = mock.Mock(side_effect=lambda **kwargs: defer.succeed(None))
        self.patch(connector.DBConnector, 'setup', setup)
        self.patch(sys, 'stderr', StringIO())
        upgrade = mock.Mock(side_effect=lambda **kwargs: defer.fail(Exception('o noz')))
        self.patch(model.Model, 'upgrade', upgrade)
        ret = (yield upgrade_master._upgradeMaster(mkconfig(basedir='test', quiet=True), config_master.MasterConfig()))
        self.assertEqual(ret, 1)
        self.assertIn('problem while upgrading!:\nTraceback (most recent call last):\n', sys.stderr.getvalue())
        self.assertIn('o noz', sys.stderr.getvalue())