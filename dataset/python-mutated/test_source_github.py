from buildbot.process.results import SUCCESS
from buildbot.steps.source import github
from buildbot.test.steps import ExpectListdir
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import ExpectStat
from buildbot.test.unit.steps import test_source_git

class TestGitHub(test_source_git.TestGit):
    stepClass = github.GitHub

    def test_with_merge_branch(self):
        if False:
            return 10
        self.setup_step(self.stepClass(repourl='http://github.com/buildbot/buildbot.git', mode='full', method='clean'), {'branch': 'refs/pull/1234/merge', 'revision': '12345678'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['git', '--version']).stdout('git version 1.7.5').exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectListdir(dir='wkdir').files(['.git']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'clean', '-f', '-f', '-d']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'fetch', '-f', '-t', 'http://github.com/buildbot/buildbot.git', 'refs/pull/1234/merge', '--progress']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'checkout', '-f', 'FETCH_HEAD']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'checkout', '-B', 'refs/pull/1234/merge']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'rev-parse', 'HEAD']).stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'f6ad368298bd941e934a41f3babc827b2aa95a1d', 'GitHub')
        return self.run_step()

    def test_with_head_branch(self):
        if False:
            print('Hello World!')
        self.setup_step(self.stepClass(repourl='http://github.com/buildbot/buildbot.git', mode='full', method='clean'), {'branch': 'refs/pull/1234/head', 'revision': '12345678'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['git', '--version']).stdout('git version 1.7.5').exit(0), ExpectStat(file='wkdir/.buildbot-patched', log_environ=True).exit(1), ExpectListdir(dir='wkdir').files(['.git']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'clean', '-f', '-f', '-d']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'cat-file', '-e', '12345678']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'checkout', '-f', '12345678']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'checkout', '-B', 'refs/pull/1234/head']).exit(0), ExpectShell(workdir='wkdir', command=['git', 'rev-parse', 'HEAD']).stdout('f6ad368298bd941e934a41f3babc827b2aa95a1d').exit(0))
        self.expect_outcome(result=SUCCESS)
        self.expect_property('got_revision', 'f6ad368298bd941e934a41f3babc827b2aa95a1d', 'GitHub')
        return self.run_step()