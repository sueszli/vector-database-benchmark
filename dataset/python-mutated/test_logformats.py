"""Black-box tests for default log_formats/log_formatters
"""
import os
from bzrlib import config, tests, workingtree

class TestLogFormats(tests.TestCaseWithTransport):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestLogFormats, self).setUp()
        conf_path = config.config_filename()
        if os.path.isfile(conf_path):
            self.fail('%s exists' % conf_path)
        config.ensure_config_dir_exists()
        f = open(conf_path, 'wb')
        try:
            f.write('[DEFAULT]\nemail=Joe Foo <joe@foo.com>\nlog_format=line\n')
        finally:
            f.close()

    def _make_simple_branch(self, relpath='.'):
        if False:
            print('Hello World!')
        wt = self.make_branch_and_tree(relpath)
        wt.commit('first revision')
        wt.commit('second revision')
        return wt

    def test_log_default_format(self):
        if False:
            print('Hello World!')
        self._make_simple_branch()
        log = self.run_bzr('log')[0]
        self.assertEqual(2, len(log.splitlines()))

    def test_log_format_arg(self):
        if False:
            print('Hello World!')
        self._make_simple_branch()
        log = self.run_bzr(['log', '--log-format', 'short'])[0]

    def test_missing_default_format(self):
        if False:
            return 10
        wt = self._make_simple_branch('a')
        self.run_bzr(['branch', 'a', 'b'])
        wt.commit('third revision')
        wt.commit('fourth revision')
        missing = self.run_bzr('missing', retcode=1, working_dir='b')[0]
        self.assertEqual(4, len(missing.splitlines()))

    def test_missing_format_arg(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self._make_simple_branch('a')
        self.run_bzr(['branch', 'a', 'b'])
        wt.commit('third revision')
        wt.commit('fourth revision')
        missing = self.run_bzr(['missing', '--log-format', 'short'], retcode=1, working_dir='b')[0]
        self.assertEqual(8, len(missing.splitlines()))

    def test_logformat_gnu_changelog(self):
        if False:
            print('Hello World!')
        wt = self.make_branch_and_tree('.')
        wt.commit('first revision', timestamp=1236045060, timezone=0)
        (log, err) = self.run_bzr(['log', '--log-format', 'gnu-changelog', '--timezone=utc'])
        self.assertEqual('', err)
        expected = '2009-03-03  Joe Foo  <joe@foo.com>\n\n\tfirst revision\n\n'
        self.assertEqualDiff(expected, log)

    def test_logformat_line_wide(self):
        if False:
            i = 10
            return i + 15
        'Author field should get larger for column widths over 80'
        wt = self.make_branch_and_tree('.')
        wt.commit('revision with a long author', committer='Person with long name SENTINEL')
        (log, err) = self.run_bzr('log --line')
        self.assertNotContainsString(log, 'SENTINEL')
        self.overrideEnv('BZR_COLUMNS', '116')
        (log, err) = self.run_bzr('log --line')
        self.assertContainsString(log, 'SENT...')
        self.overrideEnv('BZR_COLUMNS', '0')
        (log, err) = self.run_bzr('log --line')
        self.assertContainsString(log, 'SENTINEL')