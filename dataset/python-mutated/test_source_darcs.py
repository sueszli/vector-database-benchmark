from twisted.internet import error
from twisted.trial import unittest
from buildbot import config
from buildbot.process import remotetransfer
from buildbot.process.results import RETRY
from buildbot.process.results import SUCCESS
from buildbot.steps.source import darcs
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectCpdir
from buildbot.test.steps import ExpectDownloadFile
from buildbot.test.steps import ExpectRemoteRef
from buildbot.test.steps import ExpectRmdir
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import ExpectStat
from buildbot.test.util import sourcesteps

class TestDarcs(sourcesteps.SourceStepMixin, TestReactorMixin, unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        return self.setUpSourceStep()

    def tearDown(self):
        if False:
            print('Hello World!')
        return self.tearDownSourceStep()

    def test_no_empty_step_config(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(config.ConfigErrors):
            darcs.Darcs()

    def test_incorrect_method(self):
        if False:
            return 10
        with self.assertRaises(config.ConfigErrors):
            darcs.Darcs(repourl='http://localhost/darcs', mode='full', method='fresh')

    def test_incremental_invalid_method(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(config.ConfigErrors):
            darcs.Darcs(repourl='http://localhost/darcs', mode='incremental', method='fresh')

    def test_no_repo_url(self):
        if False:
            print('Hello World!')
        with self.assertRaises(config.ConfigErrors):
            darcs.Darcs(mode='full', method='fresh')

    def test_mode_full_clobber(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='full', method='clobber'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['darcs', 'get', '--verbose', '--lazy', '--repo-name', 'wkdir', 'http://localhost/darcs']).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_full_copy(self):
        if False:
            return 10
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='full', method='copy'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/_darcs', log_environ=True).exit(0), ExpectShell(workdir='source', command=['darcs', 'pull', '--all', '--verbose']).exit(0), ExpectCpdir(fromdir='source', todir='build', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='build', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_full_no_method(self):
        if False:
            while True:
                i = 10
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='full'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/_darcs', log_environ=True).exit(0), ExpectShell(workdir='source', command=['darcs', 'pull', '--all', '--verbose']).exit(0), ExpectCpdir(fromdir='source', todir='build', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='build', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_incremental(self):
        if False:
            print('Hello World!')
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='incremental'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_darcs', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'pull', '--all', '--verbose']).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_incremental_patched(self):
        if False:
            while True:
                i = 10
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='incremental'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(0), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/_darcs', log_environ=True).exit(0), ExpectShell(workdir='source', command=['darcs', 'pull', '--all', '--verbose']).exit(0), ExpectCpdir(fromdir='source', todir='build', log_environ=True, timeout=1200).exit(0), ExpectStat(file='build/_darcs', log_environ=True).exit(0), ExpectShell(workdir='build', command=['darcs', 'pull', '--all', '--verbose']).exit(0), ExpectShell(workdir='build', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_incremental_patch(self):
        if False:
            while True:
                i = 10
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='incremental'), patch=(1, 'patch'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_darcs', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'pull', '--all', '--verbose']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['patch', '-p1', '--remove-empty-files', '--force', '--forward', '-i', '.buildbot-diff']).exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_full_clobber_retry(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='full', method='clobber', retry=(0, 2)))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['darcs', 'get', '--verbose', '--lazy', '--repo-name', 'wkdir', 'http://localhost/darcs']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['darcs', 'get', '--verbose', '--lazy', '--repo-name', 'wkdir', 'http://localhost/darcs']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='.', command=['darcs', 'get', '--verbose', '--lazy', '--repo-name', 'wkdir', 'http://localhost/darcs']).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_full_clobber_revision(self):
        if False:
            while True:
                i = 10
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='full', method='clobber'), {'revision': 'abcdef01'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.darcs-context', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='.', command=['darcs', 'get', '--verbose', '--lazy', '--repo-name', 'wkdir', '--context', '.darcs-context', 'http://localhost/darcs']).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_full_clobber_revision_worker_2_16(self):
        if False:
            print('Hello World!')
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='full', method='clobber'), {'revision': 'abcdef01'}, worker_version={'*': '2.16'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), slavedest='.darcs-context', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='.', command=['darcs', 'get', '--verbose', '--lazy', '--repo-name', 'wkdir', '--context', '.darcs-context', 'http://localhost/darcs']).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_mode_incremental_no_existing_repo(self):
        if False:
            print('Hello World!')
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='incremental'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/_darcs', log_environ=True).exit(1), ExpectShell(workdir='.', command=['darcs', 'get', '--verbose', '--lazy', '--repo-name', 'wkdir', 'http://localhost/darcs']).exit(0), ExpectShell(workdir='wkdir', command=['darcs', 'changes', '--max-count=1']).stdout('Tue Aug 20 09:18:41 IST 2013 abc@gmail.com').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'Tue Aug 20 09:18:41 IST 2013 abc@gmail.com', 'Darcs')
        return self.run_step()

    def test_worker_connection_lost(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(darcs.Darcs(repourl='http://localhost/darcs', mode='full', method='clobber'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['darcs', '--version']).error(error.ConnectionLost()))
        self.expect_outcome(result=RETRY, state_string='update (retry)')
        return self.run_step()