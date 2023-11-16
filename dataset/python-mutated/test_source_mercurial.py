from twisted.internet import error
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from buildbot import config
from buildbot.process import remotetransfer
from buildbot.process.results import FAILURE
from buildbot.process.results import RETRY
from buildbot.process.results import SUCCESS
from buildbot.steps.source import mercurial
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectDownloadFile
from buildbot.test.steps import ExpectRemoteRef
from buildbot.test.steps import ExpectRmdir
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import ExpectStat
from buildbot.test.util import sourcesteps

class TestMercurial(sourcesteps.SourceStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        return self.setUpSourceStep()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tearDownSourceStep()

    def patch_workerVersionIsOlderThan(self, result):
        if False:
            print('Hello World!')
        self.patch(mercurial.Mercurial, 'workerVersionIsOlderThan', lambda x, y, z: result)

    def test_no_repourl(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(config.ConfigErrors):
            mercurial.Mercurial(mode='full')

    def test_incorrect_mode(self):
        if False:
            print('Hello World!')
        with self.assertRaises(config.ConfigErrors):
            mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='invalid')

    def test_incorrect_method(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(config.ConfigErrors):
            mercurial.Mercurial(repourl='http://hg.mozilla.org', method='invalid')

    def test_incorrect_branchType(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(config.ConfigErrors):
            mercurial.Mercurial(repourl='http://hg.mozilla.org', branchType='invalid')

    def test_mode_full_clean(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clean_win32path(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo'))
        self.build.path_module = namedModule('ntpath')
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir\\.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir\\.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clean_timeout(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', timeout=1, mode='full', method='clean', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', timeout=1, command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge']).exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', timeout=1, command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clean_patch(self):
        if False:
            print('Hello World!')
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo'), patch=(1, 'patch'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(0), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'import', '--no-commit', '-p', '1', '-'], initial_stdin='patch').exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clean_patch_worker_2_16(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo'), patch=(1, 'patch'), worker_version={'*': '2.16'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(0), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), slavedest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), slavedest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'import', '--no-commit', '-p', '1', '-'], initial_stdin='patch').exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clean_patch_fail(self):
        if False:
            return 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo'), patch=(1, 'patch'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(0), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'import', '--no-commit', '-p', '1', '-'], initial_stdin='patch').exit(1))
        self.expect_outcome(result=FAILURE, state_string='update (failure)')
        return self.run_step()

    def test_mode_full_clean_no_existing_repo(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clobber(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clobber', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_fresh(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='fresh', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge', '--all']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_fresh_no_existing_repo(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='fresh', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_fresh_retry(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='fresh', branchType='inrepo', retry=(0, 2)))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_no_existing_repo_dirname(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='dirname'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_retry(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='dirname', retry=(0, 1)))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_branch_change_dirname(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org/', mode='incremental', branchType='dirname', defaultBranch='devel'), {'branch': 'stable'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org/stable']).exit(0), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org/stable', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_no_existing_repo_inrepo(self):
        if False:
            return 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_existing_repo(self):
        if False:
            return 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_existing_repo_added_files(self):
        if False:
            print('Hello World!')
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).stdout('foo\nbar/baz\n').exit(1), ExpectRmdir(dir=['wkdir/foo', 'wkdir/bar/baz'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_existing_repo_added_files_old_rmdir(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='inrepo'))
        self.patch_workerVersionIsOlderThan(True)
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).stdout('foo\nbar/baz\n').exit(1), ExpectRmdir(dir='wkdir/foo', log_environ=True).exit(0), ExpectRmdir(dir='wkdir/bar/baz', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_given_revision(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='inrepo'), {'revision': 'abcdef01'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'abcdef01']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'abcdef01']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_branch_change(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='inrepo'), {'branch': 'stable'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'stable']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'clone', '--noupdate', 'http://hg.mozilla.org', '.']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'stable']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_branch_change_no_clobberOnBranchChange(self):
        if False:
            while True:
                i = 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='incremental', branchType='inrepo', clobberOnBranchChange=False), {'branch': 'stable'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'stable']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch']).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()']).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'stable']).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n']).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clean_env(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo', env={'abc': '123'}))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version'], env={'abc': '123'}).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.hg', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge'], env={'abc': '123'}).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default'], env={'abc': '123'}).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch'], env={'abc': '123'}).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()'], env={'abc': '123'}).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default'], env={'abc': '123'}).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n'], env={'abc': '123'}).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_full_clean_log_environ(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo', logEnviron=False))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version'], log_environ=False).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=False).exit(1), ExpectStat(file='wkdir/.hg', log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--config', 'extensions.purge=', 'purge'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'pull', 'http://hg.mozilla.org', '--rev', 'default'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'identify', '--branch'], log_environ=False).stdout('default').exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'locate', 'set:added()'], log_environ=False).exit(1), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'update', '--clean', '--rev', 'default'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['hg', '--verbose', 'parents', '--template', '{node}\\n'], log_environ=False).stdout('\n').stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_command_fails(self):
        if False:
            return 10
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='fresh', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).exit(1))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_worker_connection_lost(self):
        if False:
            print('Hello World!')
        self.setup_step(mercurial.Mercurial(repourl='http://hg.mozilla.org', mode='full', method='clean', branchType='inrepo'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['hg', '--verbose', '--version']).error(error.ConnectionLost()))
        self.expect_outcome(result=RETRY, state_string='update (retry)')
        return self.run_step()