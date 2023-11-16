from twisted.internet import error
from twisted.trial import unittest
from buildbot.process import remotetransfer
from buildbot.process.results import FAILURE
from buildbot.process.results import RETRY
from buildbot.process.results import SUCCESS
from buildbot.steps.source import mtn
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectCpdir
from buildbot.test.steps import ExpectDownloadFile
from buildbot.test.steps import ExpectRemoteRef
from buildbot.test.steps import ExpectRmdir
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import ExpectStat
from buildbot.test.util import config
from buildbot.test.util import sourcesteps

class TestMonotone(sourcesteps.SourceStepMixin, config.ConfigErrorsMixin, TestReactorMixin, unittest.TestCase):
    REVID = '95215e2a9a9f8b6f5c9664e3807cd34617ea928c'
    MTN_VER = 'monotone 1.0 (base revision: UNKNOWN_REV)'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        return self.setUpSourceStep()

    def tearDown(self):
        if False:
            return 10
        return self.tearDownSourceStep()

    def test_mode_full_clean(self):
        if False:
            return 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'unknown']).stdout('file1\nfile2').exit(0), ExpectRmdir(dir=['wkdir/file1', 'wkdir/file2'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_clean_patch(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'), patch=(1, 'patch'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'unknown']).stdout('file1\nfile2').exit(0), ExpectRmdir(dir=['wkdir/file1', 'wkdir/file2'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['patch', '-p1', '--remove-empty-files', '--force', '--forward', '-i', '.buildbot-diff']).exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_clean_patch_worker_2_16(self):
        if False:
            while True:
                i = 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'), patch=(1, 'patch'), worker_version={'*': '2.16'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'unknown']).stdout('file1\nfile2').exit(0), ExpectRmdir(dir=['wkdir/file1', 'wkdir/file2'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), slavedest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), slavedest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['patch', '-p1', '--remove-empty-files', '--force', '--forward', '-i', '.buildbot-diff']).exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_clean_patch_fail(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'), patch=(1, 'patch'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'unknown']).stdout('file1\nfile2').exit(0), ExpectRmdir(dir=['wkdir/file1', 'wkdir/file2'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['patch', '-p1', '--remove-empty-files', '--force', '--forward', '-i', '.buildbot-diff']).exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(1))
        self.expect_outcome(result=FAILURE, state_string='update (failure)')
        return self.run_step()

    def test_mode_full_clean_no_existing_db(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(1), ExpectShell(workdir='.', command=['mtn', 'db', 'init', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'unknown']).stdout('file1\nfile2').exit(0), ExpectRmdir(dir=['wkdir/file1', 'wkdir/file2'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_clean_no_existing_checkout(self):
        if False:
            return 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'checkout', 'wkdir', '--db', 'db.mtn', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_clean_from_scratch(self):
        if False:
            while True:
                i = 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(1), ExpectShell(workdir='.', command=['mtn', 'db', 'init', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'checkout', 'wkdir', '--db', 'db.mtn', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_clobber(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clobber', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'checkout', 'wkdir', '--db', 'db.mtn', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_clobber_no_existing_db(self):
        if False:
            while True:
                i = 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clobber', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(1), ExpectShell(workdir='.', command=['mtn', 'db', 'init', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'checkout', 'wkdir', '--db', 'db.mtn', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_incremental_no_existing_db(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(1), ExpectShell(workdir='.', command=['mtn', 'db', 'init', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_incremental_no_existing_checkout(self):
        if False:
            return 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'checkout', 'wkdir', '--db', 'db.mtn', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_incremental_from_scratch(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(1), ExpectShell(workdir='.', command=['mtn', 'db', 'init', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'checkout', 'wkdir', '--db', 'db.mtn', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_incremental(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_incremental_retry(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master', retry=(0, 1)))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(1), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_fresh(self):
        if False:
            while True:
                i = 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='fresh', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'unknown']).stdout('file1\nfile2').exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'ignored']).stdout('file3\nfile4').exit(0), ExpectRmdir(dir=['wkdir/file1', 'wkdir/file2', 'wkdir/file3', 'wkdir/file4'], log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_incremental_given_revision(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'), {'revision': 'abcdef01'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'abcdef01', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout('abcdef019a9f8b6f5c9664e3807cd34617ea928c').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'abcdef019a9f8b6f5c9664e3807cd34617ea928c', 'Monotone')
        return self.run_step()

    def test_mode_full_copy(self):
        if False:
            return 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='copy', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/_MTN', log_environ=True).exit(0), ExpectShell(workdir='source', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectCpdir(fromdir='source', todir='build', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='build', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_mode_full_no_method(self):
        if False:
            while True:
                i = 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/_MTN', log_environ=True).exit(0), ExpectShell(workdir='source', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectCpdir(fromdir='source', todir='build', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='build', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_incorrect_method(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesConfigError('Invalid method for mode == full'):
            mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='wrongmethod', branch='master')

    def test_incremental_invalid_method(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesConfigError('Incremental mode does not require method'):
            mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', method='fresh', branch='master')

    def test_repourl(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError('must provide repourl'):
            mtn.Monotone(mode='full', branch='master')

    def test_branch(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesConfigError('must provide branch'):
            mtn.Monotone(repourl='mtn://localhost/monotone', mode='full')

    def test_mode_incremental_patched(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('').exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'ls', 'unknown']).stdout('file1\nfile2').exit(0), ExpectRmdir(dir=['wkdir/file1', 'wkdir/file2'], log_environ=True).exit(0), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_worker_connection_lost(self):
        if False:
            while True:
                i = 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='full', method='clean', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).error(error.ConnectionLost()))
        self.expect_outcome(result=RETRY, state_string='update (retry)')
        return self.run_step()

    def test_database_migration(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('migration needed').exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'migrate', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_database_invalid(self):
        if False:
            while True:
                i = 10
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('not a monotone database').exit(0))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_database_too_new(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('too new, cannot use').exit(0), ExpectRmdir(dir='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'init', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()

    def test_database_empty(self):
        if False:
            print('Hello World!')
        self.setup_step(mtn.Monotone(repourl='mtn://localhost/monotone', mode='incremental', branch='master'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['mtn', '--version']).stdout(self.MTN_VER).exit(0), ExpectStat(file='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'info', '--db', 'db.mtn']).stdout('database has no tables').exit(0), ExpectRmdir(dir='db.mtn', log_environ=True).exit(0), ExpectShell(workdir='.', command=['mtn', 'db', 'init', '--db', 'db.mtn']).exit(0), ExpectShell(workdir='.', command=['mtn', 'pull', 'mtn://localhost/monotone?master', '--db', 'db.mtn', '--ticker=dot']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_MTN', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'update', '--revision', 'h:master', '--branch', 'master']).exit(0), ExpectShell(workdir='wkdir', command=['mtn', 'automate', 'select', 'w:']).stdout(self.REVID).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', self.REVID, 'Monotone')
        return self.run_step()