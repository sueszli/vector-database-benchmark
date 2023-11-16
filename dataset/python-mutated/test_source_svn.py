from parameterized import parameterized
from twisted.internet import defer
from twisted.internet import error
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from buildbot import config
from buildbot.interfaces import WorkerSetupError
from buildbot.process import buildstep
from buildbot.process import remotetransfer
from buildbot.process.results import FAILURE
from buildbot.process.results import RETRY
from buildbot.process.results import SUCCESS
from buildbot.steps.source import svn
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectCpdir
from buildbot.test.steps import ExpectDownloadFile
from buildbot.test.steps import ExpectRemoteRef
from buildbot.test.steps import ExpectRmdir
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import ExpectStat
from buildbot.test.util import sourcesteps
from buildbot.test.util.properties import ConstantRenderable

class TestSVN(sourcesteps.SourceStepMixin, TestReactorMixin, unittest.TestCase):
    svn_st_xml = '<?xml version="1.0"?>\n        <status>\n            <target path=".">\n                <entry path="svn_external_path">\n                    <wc-status props="none" item="external">\n                    </wc-status>\n                </entry>\n                <entry path="svn_external_path/unversioned_file1">\n                    <wc-status props="none" item="unversioned">\n                    </wc-status>\n                </entry>\n                <entry path="svn_external_path/unversioned_file2_uniçode">\n                    <wc-status props="none" item="unversioned">\n                    </wc-status>\n                </entry>\n            </target>\n        </status>\n        '
    svn_st_xml_corrupt = '<?xml version="1.0"?>\n            <target path=".">\n                <entry path="svn_external_path">\n                    <wc-status props="none" item="external">\n                    </wc-status>\n                </entry>\n                <entry path="svn_external_path/unversioned_file">\n                    <wc-status props="none" item="unversioned">\n                    </wc-status>\n                </entry>\n            </target>\n        </status>\n        '
    svn_st_xml_empty = '<?xml version="1.0"?>\n                          <status>\n                          <target path=".">\n                          </target>\n                          </status>'
    svn_info_stdout_xml = '<?xml version="1.0"?>\n                            <info>\n                            <entry\n                               kind="dir"\n                               path="."\n                               revision="100">\n                            <url>http://svn.red-bean.com/repos/test</url>\n                            <repository>\n                            <root>http://svn.red-bean.com/repos/test</root>\n                            <uuid>5e7d134a-54fb-0310-bd04-b611643e5c25</uuid>\n                            </repository>\n                            <wc-info>\n                            <schedule>normal</schedule>\n                            <depth>infinity</depth>\n                            </wc-info>\n                            <commit\n                               revision="90">\n                            <author>sally</author>\n                            <date>2003-01-15T23:35:12.847647Z</date>\n                            </commit>\n                            </entry>\n                            </info>'
    svn_info_stdout_xml_nonintegerrevision = '<?xml version="1.0"?>\n                            <info>\n                            <entry\n                               kind="dir"\n                               path="."\n                               revision="a10">\n                            <url>http://svn.red-bean.com/repos/test</url>\n                            <repository>\n                            <root>http://svn.red-bean.com/repos/test</root>\n                            <uuid>5e7d134a-54fb-0310-bd04-b611643e5c25</uuid>\n                            </repository>\n                            <wc-info>\n                            <schedule>normal</schedule>\n                            <depth>infinity</depth>\n                            </wc-info>\n                            <commit\n                               revision="a10">\n                            <author>sally</author>\n                            <date>2003-01-15T23:35:12.847647Z</date>\n                            </commit>\n                            </entry>\n                            </info>'

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
            i = 10
            return i + 15
        self.patch(svn.SVN, 'workerVersionIsOlderThan', lambda x, y, z: result)

    def test_no_repourl(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(config.ConfigErrors):
            svn.SVN()

    def test_incorrect_mode(self):
        if False:
            return 10
        with self.assertRaises(config.ConfigErrors):
            svn.SVN(repourl='http://svn.local/app/trunk', mode='invalid')

    def test_incorrect_method(self):
        if False:
            return 10
        with self.assertRaises(config.ConfigErrors):
            svn.SVN(repourl='http://svn.local/app/trunk', method='invalid')

    def test_svn_not_installed(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(1))
        self.expect_exception(WorkerSetupError)
        return self.run_step()

    def test_corrupt_xml(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_st_xml_corrupt).exit(0))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    @defer.inlineCallbacks
    def test_revision_noninteger(self):
        if False:
            return 10
        svnTestStep = svn.SVN(repourl='http://svn.local/app/trunk')
        self.setup_step(svnTestStep)
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml_nonintegerrevision).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'a10', 'SVN')
        yield self.run_step()
        revision = self.step.getProperty('got_revision')
        with self.assertRaises(ValueError):
            int(revision)

    def test_revision_missing(self):
        if False:
            while True:
                i = 10
        "Fail if 'revision' tag isn't there"
        svn_info_stdout = self.svn_info_stdout_xml.replace('entry', 'Blah')
        svnTestStep = svn.SVN(repourl='http://svn.local/app/trunk')
        self.setup_step(svnTestStep)
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(svn_info_stdout).exit(0))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_mode_incremental(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='pass', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_timeout(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', timeout=1, password='pass', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', timeout=1, command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).exit(0), ExpectShell(workdir='wkdir', timeout=1, command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_repourl_renderable(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(svn.SVN(repourl=ConstantRenderable('http://svn.local/trunk'), mode='incremental'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?>' + '<url>http://svn.local/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_repourl_canonical(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/trunk/test app', mode='incremental'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/trunk/test%20app</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_repourl_not_updatable(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl=ConstantRenderable('http://svn.local/trunk/app'), mode='incremental'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/trunk/app', '.', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_retry(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(svn.SVN(repourl=ConstantRenderable('http://svn.local/trunk/app'), mode='incremental', retry=(0, 1)))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/trunk/app', '.', '--non-interactive', '--no-auth-cache']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/trunk/app', '.', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_repourl_not_updatable_svninfo_mismatch(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(svn.SVN(repourl=ConstantRenderable('http://svn.local/trunk/app'), mode='incremental'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/branch/foo/app</url>').exit(0), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/trunk/app', '.', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_given_revision(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--revision', '100', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_win32path(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='pass', extra_args=['--random']))
        self.build.path_module = namedModule('ntpath')
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir\\.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir\\.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_mode_incremental_preferLastChangedRev(self):
        if False:
            while True:
                i = 10
        "Give the last-changed rev if 'preferLastChangedRev' is set"
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', preferLastChangedRev=True, password='pass', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '90', 'SVN')
        return self.run_step()

    def test_mode_incremental_preferLastChangedRev_butMissing(self):
        if False:
            return 10
        "If 'preferLastChangedRev' is set, but missing, fall back\n        to the regular revision value."
        svn_info_stdout = self.svn_info_stdout_xml.replace('commit', 'Blah')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', preferLastChangedRev=True, password='pass', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(svn_info_stdout).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_clobber(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clobber'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/app/trunk', '.', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_clobber_given_revision(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clobber'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/app/trunk', '.', '--revision', '100', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_fresh(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='fresh', depth='infinite'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--depth', 'infinite']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--no-ignore', '--non-interactive', '--no-auth-cache', '--depth', 'infinite']).stdout(self.svn_st_xml_empty).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--depth', 'infinite']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).stdout('\n').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_fresh_retry(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='fresh', retry=(0, 2)))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/app/trunk', '.', '--non-interactive', '--no-auth-cache']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/app/trunk', '.', '--non-interactive', '--no-auth-cache']).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/app/trunk', '.', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).stdout('\n').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_fresh_given_revision(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='fresh', depth='infinite'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--depth', 'infinite']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--no-ignore', '--non-interactive', '--no-auth-cache', '--depth', 'infinite']).stdout(self.svn_st_xml_empty).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--revision', '100', '--non-interactive', '--no-auth-cache', '--depth', 'infinite']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).stdout('\n').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_fresh_keep_on_purge(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', keep_on_purge=['svn_external_path/unversioned_file1']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--no-ignore', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml).exit(0), ExpectRmdir(dir=['wkdir/svn_external_path/unversioned_file2_uniçode'], log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_clean(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clean'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml_empty).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_clean_given_revision(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clean'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml_empty).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--revision', '100', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_not_updatable(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clean'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/app/trunk', '.', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_not_updatable_given_revision(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clean'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'checkout', 'http://svn.local/app/trunk', '.', '--revision', '100', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_clean_old_rmdir(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clean'))
        self.patch_workerVersionIsOlderThan(True)
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml).exit(0), ExpectRmdir(dir='wkdir/svn_external_path/unversioned_file1', log_environ=True, timeout=1200).exit(0), ExpectRmdir(dir='wkdir/svn_external_path/unversioned_file2_uniçode', log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_clean_new_rmdir(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clean'))
        self.patch_workerVersionIsOlderThan(False)
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml).exit(0), ExpectRmdir(dir=['wkdir/svn_external_path/unversioned_file1', 'wkdir/svn_external_path/unversioned_file2_uniçode'], log_environ=True, timeout=1200).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_copy(self):
        if False:
            while True:
                i = 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='copy', codebase='app'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/app/.svn', log_environ=True).exit(0), ExpectShell(workdir='source/app', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source/app', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectCpdir(fromdir='source/app', todir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', {'app': '100'}, 'SVN')
        return self.run_step()

    def test_mode_full_copy_given_revision(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='copy'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', command=['svn', 'update', '--revision', '100', '--non-interactive', '--no-auth-cache']).exit(0), ExpectCpdir(fromdir='source', todir='wkdir', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_export(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='export'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='', command=['svn', 'export', 'source', 'wkdir']).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_export_patch(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='export'), patch=(1, 'patch'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml).exit(0), ExpectRmdir(dir=['wkdir/svn_external_path/unversioned_file1', 'wkdir/svn_external_path/unversioned_file2_uniçode'], log_environ=True, timeout=1200).exit(0), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='', command=['svn', 'export', 'source', 'wkdir']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), workerdest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['patch', '-p1', '--remove-empty-files', '--force', '--forward', '-i', '.buildbot-diff']).exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_export_patch_worker_2_16(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='export'), patch=(1, 'patch'), worker_version={'*': '2.16'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml).exit(0), ExpectRmdir(dir=['wkdir/svn_external_path/unversioned_file1', 'wkdir/svn_external_path/unversioned_file2_uniçode'], log_environ=True, timeout=1200).exit(0), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='', command=['svn', 'export', 'source', 'wkdir']).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), slavedest='.buildbot-diff', workdir='wkdir', mode=None).exit(0), ExpectDownloadFile(blocksize=32768, maxsize=None, reader=ExpectRemoteRef(remotetransfer.StringFileReader), slavedest='.buildbot-patched', workdir='wkdir', mode=None).exit(0), ExpectShell(workdir='wkdir', command=['patch', '-p1', '--remove-empty-files', '--force', '--forward', '-i', '.buildbot-diff']).exit(0), ExpectRmdir(dir='wkdir/.buildbot-diff', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_export_timeout(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', timeout=1, mode='full', method='export'))
        self.expect_commands(ExpectShell(workdir='wkdir', timeout=1, command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', timeout=1, command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', timeout=1, command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='', timeout=1, command=['svn', 'export', 'source', 'wkdir']).exit(0), ExpectShell(workdir='source', timeout=1, command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_export_given_revision(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='export'), {'revision': '100'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', command=['svn', 'update', '--revision', '100', '--non-interactive', '--no-auth-cache']).exit(0), ExpectShell(workdir='', command=['svn', 'export', '--revision', '100', 'source', 'wkdir']).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_full_export_auth(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='export', username='svn_username', password='svn_password'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'svn_username', '--password', ('obfuscated', 'svn_password', 'XXXXXX')]).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'svn_username', '--password', ('obfuscated', 'svn_password', 'XXXXXX')]).exit(0), ExpectShell(workdir='', command=['svn', 'export', '--username', 'svn_username', '--password', ('obfuscated', 'svn_password', 'XXXXXX'), 'source', 'wkdir']).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_with_env(self):
        if False:
            while True:
                i = 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='pass', extra_args=['--random'], env={'abc': '123'}))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version'], env={'abc': '123'}).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random'], env={'abc': '123'}).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random'], env={'abc': '123'}).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml'], env={'abc': '123'}).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_mode_incremental_log_environ(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='pass', extra_args=['--random'], logEnviron=False))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version'], log_environ=False).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=False).exit(1), ExpectStat(file='wkdir/.svn', log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random'], log_environ=False).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random'], log_environ=False).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml'], log_environ=False).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', '100', 'SVN')
        return self.run_step()

    def test_command_fails(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='pass', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).exit(1))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_bogus_svnversion(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='pass', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><entry kind="dir" path="/a/b/c" revision="1"><url>http://svn.local/app/trunk</url></entry>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', 'pass', 'XXXXXX'), '--random']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout('1x0y0').exit(0))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_rmdir_fails_clobber(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='clobber'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(1))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_rmdir_fails_copy(self):
        if False:
            return 10
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='copy'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(1))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_cpdir_fails_copy(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', method='copy'))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectRmdir(dir='wkdir', log_environ=True, timeout=1200).exit(0), ExpectStat(file='source/.svn', log_environ=True).exit(0), ExpectShell(workdir='source', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='source', command=['svn', 'update', '--non-interactive', '--no-auth-cache']).exit(0), ExpectCpdir(fromdir='source', todir='wkdir', log_environ=True).exit(1))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_rmdir_fails_purge(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='full', keep_on_purge=['svn_external_path/unversioned_file1']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'status', '--xml', '--no-ignore', '--non-interactive', '--no-auth-cache']).stdout(self.svn_st_xml).exit(0), ExpectRmdir(dir=['wkdir/svn_external_path/unversioned_file2_uniçode'], log_environ=True, timeout=1200).exit(1))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

    def test_worker_connection_lost(self):
        if False:
            print('Hello World!')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='pass', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).error(error.ConnectionLost()))
        self.expect_outcome(result=RETRY, state_string='update (retry)')
        return self.run_step()

    def test_empty_password(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', password='', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', '', 'XXXXXX'), '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--password', ('obfuscated', '', 'XXXXXX'), '--random']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_omit_password(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(svn.SVN(repourl='http://svn.local/app/trunk', mode='incremental', username='user', extra_args=['--random']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['svn', '--version']).exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectStat(file='wkdir/.svn', log_environ=True).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml', '--non-interactive', '--no-auth-cache', '--username', 'user', '--random']).stdout('<?xml version="1.0"?><url>http://svn.local/app/trunk</url>').exit(0), ExpectShell(workdir='wkdir', command=['svn', 'update', '--non-interactive', '--no-auth-cache', '--username', 'user', '--random']).exit(0), ExpectShell(workdir='wkdir', command=['svn', 'info', '--xml']).stdout(self.svn_info_stdout_xml).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

class TestGetUnversionedFiles(unittest.TestCase):

    def test_getUnversionedFiles_does_not_list_externals(self):
        if False:
            for i in range(10):
                print('nop')
        svn_st_xml = '<?xml version="1.0"?>\n        <status>\n            <target path=".">\n                <entry path="svn_external_path">\n                    <wc-status props="none" item="external">\n                    </wc-status>\n                </entry>\n                <entry path="svn_external_path/unversioned_file">\n                    <wc-status props="none" item="unversioned">\n                    </wc-status>\n                </entry>\n            </target>\n        </status>\n        '
        unversioned_files = list(svn.SVN.getUnversionedFiles(svn_st_xml, []))
        self.assertEqual(['svn_external_path/unversioned_file'], unversioned_files)

    def test_getUnversionedFiles_does_not_list_missing(self):
        if False:
            return 10
        svn_st_xml = '<?xml version="1.0"?>\n        <status>\n            <target path=".">\n                <entry path="missing_file">\n                    <wc-status props="none" item="missing"></wc-status>\n                </entry>\n            </target>\n        </status>\n        '
        unversioned_files = list(svn.SVN.getUnversionedFiles(svn_st_xml, []))
        self.assertEqual([], unversioned_files)

    def test_getUnversionedFiles_corrupted_xml(self):
        if False:
            for i in range(10):
                print('nop')
        svn_st_xml_corrupt = '<?xml version="1.0"?>\n            <target path=".">\n                <entry path="svn_external_path">\n                    <wc-status props="none" item="external">\n                    </wc-status>\n                </entry>\n                <entry path="svn_external_path/unversioned_file">\n                    <wc-status props="none" item="unversioned">\n                    </wc-status>\n                </entry>\n            </target>\n        </status>\n        '
        with self.assertRaises(buildstep.BuildStepFailed):
            list(svn.SVN.getUnversionedFiles(svn_st_xml_corrupt, []))

    def test_getUnversionedFiles_no_path(self):
        if False:
            while True:
                i = 10
        svn_st_xml = '<?xml version="1.0"?>\n        <status>\n            <target path=".">\n                <entry path="svn_external_path">\n                    <wc-status props="none" item="external">\n                    </wc-status>\n                </entry>\n                <entry>\n                    <wc-status props="none" item="unversioned">\n                    </wc-status>\n                </entry>\n            </target>\n        </status>\n        '
        unversioned_files = list(svn.SVN.getUnversionedFiles(svn_st_xml, []))
        self.assertEqual([], unversioned_files)

    def test_getUnversionedFiles_no_item(self):
        if False:
            print('Hello World!')
        svn_st_xml = '<?xml version="1.0"?>\n        <status>\n            <target path=".">\n                <entry path="svn_external_path">\n                    <wc-status props="none" item="external">\n                    </wc-status>\n                </entry>\n                <entry path="svn_external_path/unversioned_file">\n                    <wc-status props="none">\n                    </wc-status>\n                </entry>\n            </target>\n        </status>\n        '
        unversioned_files = list(svn.SVN.getUnversionedFiles(svn_st_xml, []))
        self.assertEqual(['svn_external_path/unversioned_file'], unversioned_files)

    def test_getUnversionedFiles_unicode(self):
        if False:
            return 10
        svn_st_xml = '<?xml version="1.0"?>\n        <status>\n            <target path=".">\n                <entry\n                   path="Path/To/Content/Developers/François">\n                    <wc-status\n                       item="unversioned"\n                       props="none">\n                    </wc-status>\n                </entry>\n            </target>\n        </status>\n        '
        unversioned_files = list(svn.SVN.getUnversionedFiles(svn_st_xml, []))
        self.assertEqual(['Path/To/Content/Developers/François'], unversioned_files)

class TestSvnUriCanonicalize(unittest.TestCase):

    @parameterized.expand([('empty', '', ''), ('canonical', 'http://foo.com/bar', 'http://foo.com/bar'), ('lc_scheme', 'hTtP://foo.com/bar', 'http://foo.com/bar'), ('trailing_dot', 'http://foo.com./bar', 'http://foo.com/bar'), ('lc_hostname', 'http://foO.COm/bar', 'http://foo.com/bar'), ('lc_hostname_with_user', 'http://Jimmy@fOO.Com/bar', 'http://Jimmy@foo.com/bar'), ('lc_hostname_with_user_pass', 'http://Jimmy:Sekrit@fOO.Com/bar', 'http://Jimmy:Sekrit@foo.com/bar'), ('trailing_slash', 'http://foo.com/bar/', 'http://foo.com/bar'), ('trailing_slash_scheme', 'http://', 'http://'), ('trailing_slash_hostname', 'http://foo.com/', 'http://foo.com'), ('trailing_double_slash', 'http://foo.com/x//', 'http://foo.com/x'), ('double_slash', 'http://foo.com/x//y', 'http://foo.com/x/y'), ('slash', '/', '/'), ('dot', 'http://foo.com/x/./y', 'http://foo.com/x/y'), ('dot_dot', 'http://foo.com/x/../y', 'http://foo.com/y'), ('double_dot_dot', 'http://foo.com/x/y/../../z', 'http://foo.com/z'), ('dot_dot_root', 'http://foo.com/../x/y', 'http://foo.com/x/y'), ('quote_spaces', 'svn+ssh://user@host:123/My Stuff/file.doc', 'svn+ssh://user@host:123/My%20Stuff/file.doc'), ('remove_port_80', 'http://foo.com:80/bar', 'http://foo.com/bar'), ('dont_remove_port_80', 'https://foo.com:80/bar', 'https://foo.com:80/bar'), ('remove_port_443', 'https://foo.com:443/bar', 'https://foo.com/bar'), ('dont_remove_port_443', 'svn://foo.com:443/bar', 'svn://foo.com:443/bar'), ('remove_port_3690', 'svn://foo.com:3690/bar', 'svn://foo.com/bar'), ('dont_remove_port_3690', 'http://foo.com:3690/bar', 'http://foo.com:3690/bar'), ('dont_remove_port_other', 'https://foo.com:2093/bar', 'https://foo.com:2093/bar'), ('quote_funny_chars', 'http://foo.com/\x10æ%', 'http://foo.com/%10%E6%25'), ('overquoted', 'http://foo.com/%68%65%6c%6c%6f%20%77%6f%72%6c%64', 'http://foo.com/hello%20world')])
    def test_svn_uri(self, name, input, exp):
        if False:
            print('Hello World!')
        self.assertEqual(svn.SVN.svnUriCanonicalize(input), exp)