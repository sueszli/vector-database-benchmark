"""Black-box tests for running bzr outside of a working tree."""
import os
from bzrlib import osutils, tests, transport, urlutils

class TestOutsideWT(tests.ChrootedTestCase):
    """Test that bzr gives proper errors outside of a working tree."""

    def test_cwd_log(self):
        if False:
            i = 10
            return i + 15
        tmp_dir = osutils.realpath(osutils.mkdtemp())
        self.permit_url('file:///')
        self.addCleanup(osutils.rmtree, tmp_dir)
        (out, err) = self.run_bzr('log', retcode=3, working_dir=tmp_dir)
        self.assertEqual(u'bzr: ERROR: Not a branch: "%s/".\n' % (tmp_dir,), err)

    def test_url_log(self):
        if False:
            while True:
                i = 10
        url = self.get_readonly_url() + 'subdir/'
        (out, err) = self.run_bzr(['log', url], retcode=3)
        self.assertEqual(u'bzr: ERROR: Not a branch: "%s".\n' % url, err)

    def test_diff_outside_tree(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('branch1')
        tree.commit('nothing')
        tree.commit('nothing')
        tmp_dir = osutils.realpath(osutils.mkdtemp())
        self.addCleanup(osutils.rmtree, tmp_dir)
        self.permit_url('file:///')
        expected_error = u'bzr: ERROR: Not a branch: "%s/branch2/".\n' % tmp_dir
        (out, err) = self.run_bzr('diff -r revno:2:branch2..revno:1', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual(expected_error, err)
        (out, err) = self.run_bzr('diff -r revno:2:branch2', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual(expected_error, err)
        (out, err) = self.run_bzr('diff -r revno:2:branch2..', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual(expected_error, err)
        (out, err) = self.run_bzr('diff', retcode=3, working_dir=tmp_dir)
        self.assertEqual('', out)
        self.assertEqual(u'bzr: ERROR: Not a branch: "%s/".\n' % tmp_dir, err)