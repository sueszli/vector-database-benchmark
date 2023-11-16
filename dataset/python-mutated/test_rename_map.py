import os
from bzrlib import trace
from bzrlib.rename_map import RenameMap
from bzrlib.tests import TestCaseWithTransport

def myhash(val):
    if False:
        i = 10
        return i + 15
    'This the hash used by RenameMap.'
    return hash(val) % (1024 * 1024 * 10)

class TestRenameMap(TestCaseWithTransport):
    a_lines = 'a\nb\nc\n'.splitlines(True)
    b_lines = 'b\nc\nd\n'.splitlines(True)

    def test_add_edge_hashes(self):
        if False:
            i = 10
            return i + 15
        rn = RenameMap(None)
        rn.add_edge_hashes(self.a_lines, 'a')
        self.assertEqual(set(['a']), rn.edge_hashes[myhash(('a\n', 'b\n'))])
        self.assertEqual(set(['a']), rn.edge_hashes[myhash(('b\n', 'c\n'))])
        self.assertIs(None, rn.edge_hashes.get(myhash(('c\n', 'd\n'))))

    def test_add_file_edge_hashes(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/a', ''.join(self.a_lines))])
        tree.add('a', 'a')
        rn = RenameMap(tree)
        rn.add_file_edge_hashes(tree, ['a'])
        self.assertEqual(set(['a']), rn.edge_hashes[myhash(('a\n', 'b\n'))])
        self.assertEqual(set(['a']), rn.edge_hashes[myhash(('b\n', 'c\n'))])
        self.assertIs(None, rn.edge_hashes.get(myhash(('c\n', 'd\n'))))

    def test_hitcounts(self):
        if False:
            while True:
                i = 10
        rn = RenameMap(None)
        rn.add_edge_hashes(self.a_lines, 'a')
        rn.add_edge_hashes(self.b_lines, 'b')
        self.assertEqual({'a': 2.5, 'b': 0.5}, rn.hitcounts(self.a_lines))
        self.assertEqual({'a': 1}, rn.hitcounts(self.a_lines[:-1]))
        self.assertEqual({'b': 2.5, 'a': 0.5}, rn.hitcounts(self.b_lines))

    def test_file_match(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('tree')
        rn = RenameMap(tree)
        rn.add_edge_hashes(self.a_lines, 'aid')
        rn.add_edge_hashes(self.b_lines, 'bid')
        self.build_tree_contents([('tree/a', ''.join(self.a_lines))])
        self.build_tree_contents([('tree/b', ''.join(self.b_lines))])
        self.assertEqual({'a': 'aid', 'b': 'bid'}, rn.file_match(['a', 'b']))

    def test_file_match_no_dups(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        rn = RenameMap(tree)
        rn.add_edge_hashes(self.a_lines, 'aid')
        self.build_tree_contents([('tree/a', ''.join(self.a_lines))])
        self.build_tree_contents([('tree/b', ''.join(self.b_lines))])
        self.build_tree_contents([('tree/c', ''.join(self.b_lines))])
        self.assertEqual({'a': 'aid'}, rn.file_match(['a', 'b', 'c']))

    def test_match_directories(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        rn = RenameMap(tree)
        required_parents = rn.get_required_parents({'path1': 'a', 'path2/tr': 'b', 'path3/path4/path5': 'c'})
        self.assertEqual({'path2': set(['b']), 'path3/path4': set(['c']), 'path3': set()}, required_parents)

    def test_find_directory_renames(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        rn = RenameMap(tree)
        matches = {'path1': 'a', 'path3/path4/path5': 'c'}
        required_parents = {'path2': set(['b']), 'path3/path4': set(['c']), 'path3': set([])}
        missing_parents = {'path2-id': set(['b']), 'path4-id': set(['c']), 'path3-id': set(['path4-id'])}
        matches = rn.match_parents(required_parents, missing_parents)
        self.assertEqual({'path3/path4': 'path4-id', 'path2': 'path2-id'}, matches)

    def test_guess_renames(self):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/file'])
        tree.add('file', 'file-id')
        tree.commit('Added file')
        os.rename('tree/file', 'tree/file2')
        RenameMap.guess_renames(tree)
        self.assertEqual('file2', tree.id2path('file-id'))

    def test_guess_renames_handles_directories(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/dir/', 'tree/dir/file'])
        tree.add(['dir', 'dir/file'], ['dir-id', 'file-id'])
        tree.commit('Added file')
        os.rename('tree/dir', 'tree/dir2')
        RenameMap.guess_renames(tree)
        self.assertEqual('dir2/file', tree.id2path('file-id'))
        self.assertEqual('dir2', tree.id2path('dir-id'))

    def test_guess_renames_handles_grandparent_directories(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/topdir/', 'tree/topdir/middledir/', 'tree/topdir/middledir/file'])
        tree.add(['topdir', 'topdir/middledir', 'topdir/middledir/file'], ['topdir-id', 'middledir-id', 'file-id'])
        tree.commit('Added files.')
        os.rename('tree/topdir', 'tree/topdir2')
        RenameMap.guess_renames(tree)
        self.assertEqual('topdir2', tree.id2path('topdir-id'))

    def test_guess_renames_preserves_children(self):
        if False:
            while True:
                i = 10
        'When a directory has been moved, its children are preserved.'
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree_contents([('tree/foo/', ''), ('tree/foo/bar', 'bar'), ('tree/foo/empty', '')])
        tree.add(['foo', 'foo/bar', 'foo/empty'], ['foo-id', 'bar-id', 'empty-id'])
        tree.commit('rev1')
        os.rename('tree/foo', 'tree/baz')
        RenameMap.guess_renames(tree)
        self.assertEqual('baz/empty', tree.id2path('empty-id'))

    def test_guess_renames_dry_run(self):
        if False:
            return 10
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/file'])
        tree.add('file', 'file-id')
        tree.commit('Added file')
        os.rename('tree/file', 'tree/file2')
        RenameMap.guess_renames(tree, dry_run=True)
        self.assertEqual('file', tree.id2path('file-id'))

    @staticmethod
    def captureNotes(cmd, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        notes = []

        def my_note(fmt, *args):
            if False:
                for i in range(10):
                    print('nop')
            notes.append(fmt % args)
        old_note = trace.note
        trace.note = my_note
        try:
            result = cmd(*args, **kwargs)
        finally:
            trace.note = old_note
        return (notes, result)

    def test_guess_renames_output(self):
        if False:
            while True:
                i = 10
        'guess_renames emits output whether dry_run is True or False.'
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/file'])
        tree.add('file', 'file-id')
        tree.commit('Added file')
        os.rename('tree/file', 'tree/file2')
        notes = self.captureNotes(RenameMap.guess_renames, tree, dry_run=True)[0]
        self.assertEqual('file => file2', ''.join(notes))
        notes = self.captureNotes(RenameMap.guess_renames, tree, dry_run=False)[0]
        self.assertEqual('file => file2', ''.join(notes))