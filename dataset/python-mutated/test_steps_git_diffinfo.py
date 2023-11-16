from twisted.trial import unittest
from buildbot.process import results
from buildbot.steps import gitdiffinfo
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import TestBuildStepMixin
try:
    import unidiff
except ImportError:
    unidiff = None

class TestDiffInfo(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):
    if not unidiff:
        skip = 'unidiff is required for GitDiffInfo tests'

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            while True:
                i = 10
        return self.tear_down_test_build_step()

    def test_merge_base_failure(self):
        if False:
            print('Hello World!')
        self.setup_step(gitdiffinfo.GitDiffInfo())
        self.expect_commands(ExpectShell(workdir='wkdir', command=['git', 'merge-base', 'HEAD', 'master']).log('stdio-merge-base', stderr='fatal: Not a valid object name').exit(128))
        self.expect_log_file_stderr('stdio-merge-base', 'fatal: Not a valid object name\n')
        self.expect_outcome(result=results.FAILURE, state_string='GitDiffInfo (failure)')
        return self.run_step()

    def test_diff_failure(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(gitdiffinfo.GitDiffInfo())
        self.expect_commands(ExpectShell(workdir='wkdir', command=['git', 'merge-base', 'HEAD', 'master']).log('stdio-merge-base', stdout='1234123412341234').exit(0), ExpectShell(workdir='wkdir', command=['git', 'diff', '--no-prefix', '-U0', '1234123412341234', 'HEAD']).log('stdio-diff', stderr='fatal: ambiguous argument').exit(1))
        self.expect_log_file('stdio-merge-base', '1234123412341234')
        self.expect_log_file_stderr('stdio-diff', 'fatal: ambiguous argument\n')
        self.expect_outcome(result=results.FAILURE, state_string='GitDiffInfo (failure)')
        return self.run_step()

    def test_empty_diff(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(gitdiffinfo.GitDiffInfo())
        self.expect_commands(ExpectShell(workdir='wkdir', command=['git', 'merge-base', 'HEAD', 'master']).log('stdio-merge-base', stdout='1234123412341234').exit(0), ExpectShell(workdir='wkdir', command=['git', 'diff', '--no-prefix', '-U0', '1234123412341234', 'HEAD']).log('stdio-diff', stdout='').exit(0))
        self.expect_log_file('stdio-merge-base', '1234123412341234')
        self.expect_log_file_stderr('stdio-diff', '')
        self.expect_outcome(result=results.SUCCESS, state_string='GitDiffInfo')
        self.expect_build_data('diffinfo-master', b'[]', 'GitDiffInfo')
        return self.run_step()

    def test_complex_diff(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(gitdiffinfo.GitDiffInfo())
        self.expect_commands(ExpectShell(workdir='wkdir', command=['git', 'merge-base', 'HEAD', 'master']).log('stdio-merge-base', stdout='1234123412341234').exit(0), ExpectShell(workdir='wkdir', command=['git', 'diff', '--no-prefix', '-U0', '1234123412341234', 'HEAD']).log('stdio-diff', stdout='diff --git file1 file1\ndeleted file mode 100644\nindex 42f90fd..0000000\n--- file1\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-line11\n-line12\n-line13\ndiff --git file2 file2\nindex c337bf1..1cb02b9 100644\n--- file2\n+++ file2\n@@ -4,0 +5,3 @@ line24\n+line24n\n+line24n2\n+line24n3\n@@ -15,0 +19,3 @@ line215\n+line215n\n+line215n2\n+line215n3\ndiff --git file3 file3\nnew file mode 100644\nindex 0000000..632e269\n--- /dev/null\n+++ file3\n@@ -0,0 +1,3 @@\n+line31\n+line32\n+line33\n').exit(0))
        self.expect_log_file('stdio-merge-base', '1234123412341234')
        self.expect_outcome(result=results.SUCCESS, state_string='GitDiffInfo')
        diff_info = b'[{"source_file": "file1", "target_file": "/dev/null", ' + b'"is_binary": false, "is_rename": false, ' + b'"hunks": [{"ss": 1, "sl": 3, "ts": 0, "tl": 0}]}, ' + b'{"source_file": "file2", "target_file": "file2", ' + b'"is_binary": false, "is_rename": false, ' + b'"hunks": [{"ss": 4, "sl": 0, "ts": 5, "tl": 3}, ' + b'{"ss": 15, "sl": 0, "ts": 19, "tl": 3}]}, ' + b'{"source_file": "/dev/null", "target_file": "file3", ' + b'"is_binary": false, "is_rename": false, ' + b'"hunks": [{"ss": 0, "sl": 0, "ts": 1, "tl": 3}]}]'
        self.expect_build_data('diffinfo-master', diff_info, 'GitDiffInfo')
        return self.run_step()