import sys
import bzrlib.errors
from bzrlib.osutils import getcwd
from bzrlib.tests import TestCaseWithTransport, TestNotApplicable, TestSkipped
from bzrlib import urlutils
'Tests for Branch parent URL'

class TestParent(TestCaseWithTransport):

    def test_no_default_parent(self):
        if False:
            return 10
        'Branches should have no parent by default'
        b = self.make_branch('.')
        self.assertEqual(None, b.get_parent())

    def test_set_get_parent(self):
        if False:
            return 10
        'Set, re-get and reset the parent'
        b = self.make_branch('subdir')
        url = 'http://example.com/bzr/bzr.dev'
        b.set_parent(url)
        self.assertEqual(url, b.get_parent())
        self.assertEqual(url, b._get_parent_location())
        b.set_parent(None)
        self.assertEqual(None, b.get_parent())
        b.set_parent('../other_branch')
        expected_parent = urlutils.join(self.get_url('subdir'), '../other_branch')
        self.assertEqual(expected_parent, b.get_parent())
        path = urlutils.join(self.get_url('subdir'), '../yanb')
        b.set_parent(path)
        self.assertEqual('../yanb', b._get_parent_location())
        self.assertEqual(path, b.get_parent())
        self.assertRaises(bzrlib.errors.InvalidURL, b.set_parent, u'µ')
        b.set_parent(urlutils.escape(u'µ'))
        self.assertEqual('%C2%B5', b._get_parent_location())
        self.assertEqual(b.base + '%C2%B5', b.get_parent())
        if sys.platform == 'win32':
            pass
        else:
            b.lock_write()
            b._set_parent_location('/local/abs/path')
            b.unlock()
            self.assertEqual('file:///local/abs/path', b.get_parent())

    def test_get_invalid_parent(self):
        if False:
            print('Hello World!')
        b = self.make_branch('.')
        cwd = getcwd()
        n_dirs = len(cwd.split('/'))
        path = '../' * (n_dirs + 5) + 'foo'
        b.lock_write()
        b._set_parent_location(path)
        b.unlock()
        self.assertRaises(bzrlib.errors.InaccessibleParent, b.get_parent)

    def test_win32_set_parent_on_another_drive(self):
        if False:
            while True:
                i = 10
        if sys.platform != 'win32':
            raise TestSkipped('windows-specific test')
        b = self.make_branch('.')
        base_url = b.bzrdir.transport.abspath('.')
        if not base_url.startswith('file:///'):
            raise TestNotApplicable('this test should be run with local base')
        base = urlutils.local_path_from_url(base_url)
        other = 'file:///D:/path'
        if base[0] != 'C':
            other = 'file:///C:/path'
        b.set_parent(other)
        self.assertEqual(other, b._get_parent_location())