from bzrlib import conflicts, tests, workingtree
from bzrlib.tests import script, features

def make_tree_with_conflicts(test, this_path='this', other_path='other', prefix='my'):
    if False:
        i = 10
        return i + 15
    this_tree = test.make_branch_and_tree(this_path)
    test.build_tree_contents([('%s/%sfile' % (this_path, prefix), 'this content\n'), ('%s/%s_other_file' % (this_path, prefix), 'this content\n'), ('%s/%sdir/' % (this_path, prefix),)])
    this_tree.add(prefix + 'file')
    this_tree.add(prefix + '_other_file')
    this_tree.add(prefix + 'dir')
    this_tree.commit(message='new')
    other_tree = this_tree.bzrdir.sprout(other_path).open_workingtree()
    test.build_tree_contents([('%s/%sfile' % (other_path, prefix), 'contentsb\n'), ('%s/%s_other_file' % (other_path, prefix), 'contentsb\n')])
    other_tree.rename_one(prefix + 'dir', prefix + 'dir2')
    other_tree.commit(message='change')
    test.build_tree_contents([('%s/%sfile' % (this_path, prefix), 'contentsa2\n'), ('%s/%s_other_file' % (this_path, prefix), 'contentsa2\n')])
    this_tree.rename_one(prefix + 'dir', prefix + 'dir3')
    this_tree.commit(message='change')
    this_tree.merge_from_branch(other_tree.branch)
    return (this_tree, other_tree)

class TestConflicts(script.TestCaseWithTransportAndScript):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestConflicts, self).setUp()
        make_tree_with_conflicts(self, 'branch', 'other')

    def test_conflicts(self):
        if False:
            return 10
        self.run_script('$ cd branch\n$ bzr conflicts\nText conflict in my_other_file\nPath conflict: mydir3 / mydir2\nText conflict in myfile\n')

    def test_conflicts_text(self):
        if False:
            return 10
        self.run_script('$ cd branch\n$ bzr conflicts --text\nmy_other_file\nmyfile\n')

    def test_conflicts_directory(self):
        if False:
            print('Hello World!')
        self.run_script('$ bzr conflicts  -d branch\nText conflict in my_other_file\nPath conflict: mydir3 / mydir2\nText conflict in myfile\n')

class TestUnicodePaths(tests.TestCaseWithTransport):
    """Unicode characters in conflicts should be displayed properly"""
    _test_needs_features = [features.UnicodeFilenameFeature]
    encoding = 'UTF-8'

    def _as_output(self, text):
        if False:
            i = 10
            return i + 15
        return text

    def test_messages(self):
        if False:
            while True:
                i = 10
        'Conflict messages involving non-ascii paths are displayed okay'
        make_tree_with_conflicts(self, 'branch', prefix=u'§')
        (out, err) = self.run_bzr(['conflicts', '-d', 'branch'], encoding=self.encoding)
        self.assertEqual(out.decode(self.encoding), u'Text conflict in §_other_file\nPath conflict: §dir3 / §dir2\nText conflict in §file\n')
        self.assertEqual(err, '')

    def test_text_conflict_paths(self):
        if False:
            for i in range(10):
                print('nop')
        'Text conflicts on non-ascii paths are displayed okay'
        make_tree_with_conflicts(self, 'branch', prefix=u'§')
        (out, err) = self.run_bzr(['conflicts', '-d', 'branch', '--text'], encoding=self.encoding)
        self.assertEqual(out.decode(self.encoding), u'§_other_file\n§file\n')
        self.assertEqual(err, '')

class TestUnicodePathsOnAsciiTerminal(TestUnicodePaths):
    """Undisplayable unicode characters in conflicts should be escaped"""
    encoding = 'ascii'

    def setUp(self):
        if False:
            print('Hello World!')
        self.skip('Need to decide if replacing is the desired behaviour')

    def _as_output(self, text):
        if False:
            print('Hello World!')
        return text.encode(self.encoding, 'replace')