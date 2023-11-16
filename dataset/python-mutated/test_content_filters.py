"""Tests for content filtering conformance"""
import os
from bzrlib.controldir import ControlDir
from bzrlib.filters import ContentFilter
from bzrlib.switch import switch
from bzrlib.workingtree import WorkingTree
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree

def _converter_helper(chunks, fn):
    if False:
        print('Hello World!')
    result = []
    for chunk in chunks:
        result.append(getattr(chunk, fn)())
    return iter(result)

def _swapcase(chunks, context=None):
    if False:
        while True:
            i = 10
    'A converter that swaps the case of text.'
    return _converter_helper(chunks, 'swapcase')

def _uppercase(chunks, context=None):
    if False:
        print('Hello World!')
    'A converter that converts text to uppercase.'
    return _converter_helper(chunks, 'upper')

def _lowercase(chunks, context=None):
    if False:
        print('Hello World!')
    'A converter that converts text to lowercase.'
    return _converter_helper(chunks, 'lower')
_trailer_string = '\nend string\n'

def _append_text(chunks, context=None):
    if False:
        i = 10
        return i + 15
    'A content filter that appends a string to the end of the file.\n\n    This tests filters that change the length.'
    return chunks + [_trailer_string]

def _remove_appended_text(chunks, context=None):
    if False:
        while True:
            i = 10
    'Remove the appended text.'
    text = ''.join(chunks)
    if text.endswith(_trailer_string):
        text = text[:-len(_trailer_string)]
    return [text]

class TestWorkingTreeWithContentFilters(TestCaseWithWorkingTree):

    def create_cf_tree(self, txt_reader, txt_writer, dir='.'):
        if False:
            return 10
        tree = self.make_branch_and_tree(dir)

        def _content_filter_stack(path=None, file_id=None):
            if False:
                while True:
                    i = 10
            if path.endswith('.txt'):
                return [ContentFilter(txt_reader, txt_writer)]
            else:
                return []
        tree._content_filter_stack = _content_filter_stack
        self.build_tree_contents([(dir + '/file1.txt', 'Foo Txt'), (dir + '/file2.bin', 'Foo Bin')])
        tree.add(['file1.txt', 'file2.bin'])
        tree.commit('commit raw content')
        txt_fileid = tree.path2id('file1.txt')
        bin_fileid = tree.path2id('file2.bin')
        return (tree, txt_fileid, bin_fileid)

    def create_cf_tree_with_two_revisions(self, txt_reader, txt_writer, dir='.'):
        if False:
            while True:
                i = 10
        tree = self.make_branch_and_tree(dir)

        def _content_filter_stack(path=None, file_id=None):
            if False:
                for i in range(10):
                    print('nop')
            if path.endswith('.txt'):
                return [ContentFilter(txt_reader, txt_writer)]
            else:
                return []
        tree._content_filter_stack = _content_filter_stack
        self.build_tree_contents([(dir + '/file1.txt', 'Foo Txt'), (dir + '/file2.bin', 'Foo Bin'), (dir + '/file3.txt', 'Bar Txt')])
        tree.add(['file1.txt', 'file2.bin', 'file3.txt'])
        tree.commit('commit raw content')
        fileid_1 = tree.path2id('file1.txt')
        fileid_2 = tree.path2id('file2.bin')
        fileid_3 = tree.path2id('file3.txt')
        self.build_tree_contents([(dir + '/file1.txt', 'Foo ROCKS!'), (dir + '/file4.txt', 'Hello World')])
        tree.add(['file4.txt'])
        tree.remove(['file3.txt'], keep_files=False)
        tree.commit('change, add and rename stuff')
        fileid_4 = tree.path2id('file4.txt')
        return (tree, fileid_1, fileid_2, fileid_3, fileid_4)

    def patch_in_content_filter(self):
        if False:
            i = 10
            return i + 15

        def new_stack(tree, path=None, file_id=None):
            if False:
                for i in range(10):
                    print('nop')
            if path.endswith('.txt'):
                return [ContentFilter(_swapcase, _swapcase)]
            else:
                return []
        self.overrideAttr(WorkingTree, '_content_filter_stack', new_stack)

    def assert_basis_content(self, expected_content, branch, file_id):
        if False:
            for i in range(10):
                print('nop')
        basis = branch.basis_tree()
        basis.lock_read()
        try:
            self.assertEqual(expected_content, basis.get_file_text(file_id))
        finally:
            basis.unlock()

    def test_symmetric_content_filtering(self):
        if False:
            while True:
                i = 10
        (tree, txt_fileid, bin_fileid) = self.create_cf_tree(txt_reader=_swapcase, txt_writer=_swapcase)
        basis = tree.basis_tree()
        basis.lock_read()
        self.addCleanup(basis.unlock)
        if tree.supports_content_filtering():
            expected = 'fOO tXT'
        else:
            expected = 'Foo Txt'
        self.assertEqual(expected, basis.get_file_text(txt_fileid))
        self.assertEqual('Foo Bin', basis.get_file_text(bin_fileid))
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual('Foo Txt', tree.get_file(txt_fileid, filtered=False).read())
        self.assertEqual('Foo Bin', tree.get_file(bin_fileid, filtered=False).read())

    def test_readonly_content_filtering(self):
        if False:
            i = 10
            return i + 15
        (tree, txt_fileid, bin_fileid) = self.create_cf_tree(txt_reader=_uppercase, txt_writer=None)
        basis = tree.basis_tree()
        basis.lock_read()
        self.addCleanup(basis.unlock)
        if tree.supports_content_filtering():
            expected = 'FOO TXT'
        else:
            expected = 'Foo Txt'
        self.assertEqual(expected, basis.get_file_text(txt_fileid))
        self.assertEqual('Foo Bin', basis.get_file_text(bin_fileid))
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual('Foo Txt', tree.get_file(txt_fileid, filtered=False).read())
        self.assertEqual('Foo Bin', tree.get_file(bin_fileid, filtered=False).read())

    def test_branch_source_filtered_target_not(self):
        if False:
            i = 10
            return i + 15
        (source, txt_fileid, bin_fileid) = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual('Foo Txt', 'source/file1.txt')
        self.assert_basis_content('FOO TXT', source, txt_fileid)
        self.run_bzr('branch source target')
        target = WorkingTree.open('target')
        self.assertFileEqual('FOO TXT', 'target/file1.txt')
        changes = target.changes_from(source.basis_tree())
        self.assertFalse(changes.has_changed())

    def test_branch_source_not_filtered_target_is(self):
        if False:
            i = 10
            return i + 15
        (source, txt_fileid, bin_fileid) = self.create_cf_tree(txt_reader=None, txt_writer=None, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual('Foo Txt', 'source/file1.txt')
        self.assert_basis_content('Foo Txt', source, txt_fileid)
        self.patch_in_content_filter()
        self.run_bzr('branch source target')
        target = WorkingTree.open('target')
        self.assertFileEqual('fOO tXT', 'target/file1.txt')
        changes = target.changes_from(source.basis_tree())
        self.assertFalse(changes.has_changed())

    def test_path_content_summary(self):
        if False:
            return 10
        'path_content_summary should always talk about the canonical form.'
        (source, txt_fileid, bin_fileid) = self.create_cf_tree(txt_reader=_append_text, txt_writer=_remove_appended_text, dir='source')
        if not source.supports_content_filtering():
            return
        source.lock_read()
        self.addCleanup(source.unlock)
        expected_canonical_form = 'Foo Txt\nend string\n'
        self.assertEqual(source.get_file(txt_fileid, filtered=True).read(), expected_canonical_form)
        self.assertEqual(source.get_file(txt_fileid, filtered=False).read(), 'Foo Txt')
        result = source.path_content_summary('file1.txt')
        self.assertEqual(result, ('file', None, False, None))

    def test_content_filtering_applied_on_pull(self):
        if False:
            while True:
                i = 10
        (source, fileid_1, fileid_2, fileid_3, fileid_4) = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual('Foo ROCKS!', 'source/file1.txt')
        self.assert_basis_content('Foo ROCKS!', source, fileid_1)
        self.patch_in_content_filter()
        self.run_bzr('branch -r1 source target')
        target = WorkingTree.open('target')
        self.assert_basis_content('Foo Txt', target, fileid_1)
        self.assertFileEqual('fOO tXT', 'target/file1.txt')
        self.assert_basis_content('Foo Bin', target, fileid_2)
        self.assertFileEqual('Foo Bin', 'target/file2.bin')
        self.assert_basis_content('Bar Txt', target, fileid_3)
        self.assertFileEqual('bAR tXT', 'target/file3.txt')
        self.run_bzr('pull -d target')
        self.assert_basis_content('Foo ROCKS!', target, fileid_1)
        self.assertFileEqual('fOO rocks!', 'target/file1.txt')
        self.assert_basis_content('Foo Bin', target, fileid_2)
        self.assert_basis_content('Hello World', target, fileid_4)
        self.assertFileEqual('hELLO wORLD', 'target/file4.txt')

    def test_content_filtering_applied_on_merge(self):
        if False:
            return 10
        (source, fileid_1, fileid_2, fileid_3, fileid_4) = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='source')
        if not source.supports_content_filtering():
            return
        self.assert_basis_content('Foo ROCKS!', source, fileid_1)
        self.assertFileEqual('Foo ROCKS!', 'source/file1.txt')
        self.assert_basis_content('Foo Bin', source, fileid_2)
        self.assert_basis_content('Hello World', source, fileid_4)
        self.assertFileEqual('Hello World', 'source/file4.txt')
        self.patch_in_content_filter()
        self.run_bzr('branch -r1 source target')
        target = WorkingTree.open('target')
        self.assert_basis_content('Foo Txt', target, fileid_1)
        self.assertFileEqual('fOO tXT', 'target/file1.txt')
        self.assertFileEqual('Foo Bin', 'target/file2.bin')
        self.assertFileEqual('bAR tXT', 'target/file3.txt')
        self.run_bzr('merge -d target source')
        self.assertFileEqual('fOO rocks!', 'target/file1.txt')
        self.assertFileEqual('hELLO wORLD', 'target/file4.txt')
        target.commit('merge file1.txt changes from source')
        self.assert_basis_content('Foo ROCKS!', target, fileid_1)
        self.assert_basis_content('Hello World', target, fileid_4)

    def test_content_filtering_applied_on_switch(self):
        if False:
            return 10
        (source, fileid_1, fileid_2, fileid_3, fileid_4) = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='branch-a')
        if not source.supports_content_filtering():
            return
        self.patch_in_content_filter()
        self.run_bzr('branch -r1 branch-a branch-b')
        self.run_bzr('checkout --lightweight branch-b checkout')
        self.assertFileEqual('fOO tXT', 'checkout/file1.txt')
        checkout_control_dir = ControlDir.open_containing('checkout')[0]
        switch(checkout_control_dir, source.branch)
        self.assertFileEqual('fOO rocks!', 'checkout/file1.txt')
        self.assertFileEqual('hELLO wORLD', 'checkout/file4.txt')

    def test_content_filtering_applied_on_revert_delete(self):
        if False:
            i = 10
            return i + 15
        (source, txt_fileid, bin_fileid) = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual('Foo Txt', 'source/file1.txt')
        self.assert_basis_content('FOO TXT', source, txt_fileid)
        os.unlink('source/file1.txt')
        self.assertFalse(os.path.exists('source/file1.txt'))
        source.revert(['file1.txt'])
        self.assertTrue(os.path.exists('source/file1.txt'))
        self.assertFileEqual('foo txt', 'source/file1.txt')

    def test_content_filtering_applied_on_revert_rename(self):
        if False:
            print('Hello World!')
        (source, txt_fileid, bin_fileid) = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual('Foo Txt', 'source/file1.txt')
        self.assert_basis_content('FOO TXT', source, txt_fileid)
        self.build_tree_contents([('source/file1.txt', 'Foo Txt with new content')])
        source.rename_one('file1.txt', 'file1.bin')
        self.assertTrue(os.path.exists('source/file1.bin'))
        self.assertFalse(os.path.exists('source/file1.txt'))
        self.assertFileEqual('Foo Txt with new content', 'source/file1.bin')
        source.revert(['file1.bin'])
        self.assertFalse(os.path.exists('source/file1.bin'))
        self.assertTrue(os.path.exists('source/file1.txt'))
        self.assertFileEqual('foo txt', 'source/file1.txt')