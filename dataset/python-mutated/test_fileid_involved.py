import sys
import time
from bzrlib import errors, inventory, remote, revision as _mod_revision, tests, transform
from bzrlib.tests.scenarios import load_tests_apply_scenarios
from bzrlib.tests.per_repository_vf import TestCaseWithRepository, all_repository_vf_format_scenarios
load_tests = load_tests_apply_scenarios

class FileIdInvolvedWGhosts(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def create_branch_with_ghost_text(self):
        if False:
            while True:
                i = 10
        builder = self.make_branch_builder('ghost')
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('a', 'a-file-id', 'file', 'some content\n'))])
        b = builder.get_branch()
        old_rt = b.repository.revision_tree('A-id')
        new_inv = inventory.mutable_inventory_from_tree(old_rt)
        new_inv.revision_id = 'B-id'
        new_inv['a-file-id'].revision = 'ghost-id'
        new_rev = _mod_revision.Revision('B-id', timestamp=time.time(), timezone=0, message='Committing against a ghost', committer='Joe Foo <joe@foo.com>', properties={}, parent_ids=('A-id', 'ghost-id'))
        b.lock_write()
        self.addCleanup(b.unlock)
        b.repository.start_write_group()
        b.repository.add_revision('B-id', new_rev, new_inv)
        self.disable_commit_write_group_paranoia(b.repository)
        b.repository.commit_write_group()
        return b

    def disable_commit_write_group_paranoia(self, repo):
        if False:
            print('Hello World!')
        if isinstance(repo, remote.RemoteRepository):
            repo.abort_write_group()
            raise tests.TestSkipped('repository format does not support storing revisions with missing texts.')
        pack_coll = getattr(repo, '_pack_collection', None)
        if pack_coll is not None:
            pack_coll._check_new_inventories = lambda : []

    def test_file_ids_include_ghosts(self):
        if False:
            return 10
        b = self.create_branch_with_ghost_text()
        repo = b.repository
        self.assertEqual({'a-file-id': set(['ghost-id'])}, repo.fileids_altered_by_revision_ids(['B-id']))

    def test_file_ids_uses_fallbacks(self):
        if False:
            print('Hello World!')
        builder = self.make_branch_builder('source', format=self.bzrdir_format)
        repo = builder.get_branch().repository
        if not repo._format.supports_external_lookups:
            raise tests.TestNotApplicable('format does not support stacking')
        builder.start_series()
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('file', 'file-id', 'file', 'contents\n'))])
        builder.build_snapshot('B-id', ['A-id'], [('modify', ('file-id', 'new-content\n'))])
        builder.build_snapshot('C-id', ['B-id'], [('modify', ('file-id', 'yet more content\n'))])
        builder.finish_series()
        source_b = builder.get_branch()
        source_b.lock_read()
        self.addCleanup(source_b.unlock)
        base = self.make_branch('base')
        base.pull(source_b, stop_revision='B-id')
        stacked = self.make_branch('stacked')
        stacked.set_stacked_on_url('../base')
        stacked.pull(source_b, stop_revision='C-id')
        stacked.lock_read()
        self.addCleanup(stacked.unlock)
        repo = stacked.repository
        keys = {'file-id': set(['A-id'])}
        if stacked.repository.supports_rich_root():
            keys['root-id'] = set(['A-id'])
        self.assertEqual(keys, repo.fileids_altered_by_revision_ids(['A-id']))

class FileIdInvolvedBase(TestCaseWithRepository):

    def touch(self, tree, filename):
        if False:
            while True:
                i = 10
        tree.bzrdir.root_transport.append_bytes(filename, 'appended line\n')

    def compare_tree_fileids(self, branch, old_rev, new_rev):
        if False:
            return 10
        old_tree = self.branch.repository.revision_tree(old_rev)
        new_tree = self.branch.repository.revision_tree(new_rev)
        delta = new_tree.changes_from(old_tree)
        l2 = [id for (path, id, kind) in delta.added] + [id for (oldpath, newpath, id, kind, text_modified, meta_modified) in delta.renamed] + [id for (path, id, kind, text_modified, meta_modified) in delta.modified]
        return set(l2)

class TestFileIdInvolved(FileIdInvolvedBase):
    scenarios = all_repository_vf_format_scenarios()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestFileIdInvolved, self).setUp()
        main_wt = self.make_branch_and_tree('main')
        main_branch = main_wt.branch
        self.build_tree(['main/a', 'main/b', 'main/c'])
        main_wt.add(['a', 'b', 'c'], ['a-file-id-2006-01-01-abcd', 'b-file-id-2006-01-01-defg', 'c-funky<file-id>quiji%bo'])
        try:
            main_wt.commit('Commit one', rev_id='rev-A')
        except errors.IllegalPath:
            if sys.platform == 'win32':
                raise tests.TestSkipped('Old repository formats do not support file ids with <> on win32')
            raise
        bt1 = self.make_branch_and_tree('branch1')
        bt1.pull(main_branch)
        b1 = bt1.branch
        self.build_tree(['branch1/d'])
        bt1.add(['d'], ['file-d'])
        bt1.commit('branch1, Commit one', rev_id='rev-E')
        self.touch(main_wt, 'a')
        main_wt.commit('Commit two', rev_id='rev-B')
        bt2 = self.make_branch_and_tree('branch2')
        bt2.pull(main_branch)
        branch2_branch = bt2.branch
        set_executability(bt2, 'b', True)
        bt2.commit('branch2, Commit one', rev_id='rev-J')
        main_wt.merge_from_branch(b1)
        main_wt.commit('merge branch1, rev-11', rev_id='rev-C')
        bt1.rename_one('d', 'e')
        bt1.commit('branch1, commit two', rev_id='rev-F')
        self.touch(bt2, 'c')
        bt2.commit('branch2, commit two', rev_id='rev-K')
        main_wt.merge_from_branch(b1)
        self.touch(main_wt, 'b')
        main_wt.commit('merge branch1, rev-12', rev_id='rev-<D>')
        main_wt.merge_from_branch(branch2_branch)
        main_wt.commit('merge branch1, rev-22', rev_id='rev-G')
        self.branch = main_branch

    def test_fileids_altered_between_two_revs(self):
        if False:
            i = 10
            return i + 15
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        self.branch.repository.fileids_altered_by_revision_ids(['rev-J', 'rev-K'])
        self.assertEqual({'b-file-id-2006-01-01-defg': set(['rev-J']), 'c-funky<file-id>quiji%bo': set(['rev-K'])}, self.branch.repository.fileids_altered_by_revision_ids(['rev-J', 'rev-K']))
        self.assertEqual({'b-file-id-2006-01-01-defg': set(['rev-<D>']), 'file-d': set(['rev-F'])}, self.branch.repository.fileids_altered_by_revision_ids(['rev-<D>', 'rev-F']))
        self.assertEqual({'b-file-id-2006-01-01-defg': set(['rev-<D>', 'rev-G', 'rev-J']), 'c-funky<file-id>quiji%bo': set(['rev-K']), 'file-d': set(['rev-F'])}, self.branch.repository.fileids_altered_by_revision_ids(['rev-<D>', 'rev-G', 'rev-F', 'rev-K', 'rev-J']))
        self.assertEqual({'a-file-id-2006-01-01-abcd': set(['rev-B']), 'b-file-id-2006-01-01-defg': set(['rev-<D>', 'rev-G', 'rev-J']), 'c-funky<file-id>quiji%bo': set(['rev-K']), 'file-d': set(['rev-F'])}, self.branch.repository.fileids_altered_by_revision_ids(['rev-G', 'rev-F', 'rev-C', 'rev-B', 'rev-<D>', 'rev-K', 'rev-J']))

    def fileids_altered_by_revision_ids(self, revision_ids):
        if False:
            i = 10
            return i + 15
        'This is a wrapper to strip TREE_ROOT if it occurs'
        repo = self.branch.repository
        root_id = self.branch.basis_tree().get_root_id()
        result = repo.fileids_altered_by_revision_ids(revision_ids)
        if root_id in result:
            del result[root_id]
        return result

    def test_fileids_altered_by_revision_ids(self):
        if False:
            print('Hello World!')
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        self.assertEqual({'a-file-id-2006-01-01-abcd': set(['rev-A']), 'b-file-id-2006-01-01-defg': set(['rev-A']), 'c-funky<file-id>quiji%bo': set(['rev-A'])}, self.fileids_altered_by_revision_ids(['rev-A']))
        self.assertEqual({'a-file-id-2006-01-01-abcd': set(['rev-B'])}, self.branch.repository.fileids_altered_by_revision_ids(['rev-B']))
        self.assertEqual({'b-file-id-2006-01-01-defg': set(['rev-<D>'])}, self.branch.repository.fileids_altered_by_revision_ids(['rev-<D>']))

    def test_fileids_involved_full_compare(self):
        if False:
            print('Hello World!')
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        pp = []
        graph = self.branch.repository.get_graph()
        history = list(graph.iter_lefthand_ancestry(self.branch.last_revision(), [_mod_revision.NULL_REVISION]))
        history.reverse()
        if len(history) < 2:
            return
        for start in range(0, len(history) - 1):
            start_id = history[start]
            for end in range(start + 1, len(history)):
                end_id = history[end]
                unique_revs = graph.find_unique_ancestors(end_id, [start_id])
                l1 = self.branch.repository.fileids_altered_by_revision_ids(unique_revs)
                l1 = set(l1.keys())
                l2 = self.compare_tree_fileids(self.branch, start_id, end_id)
                self.assertEqual(l1, l2)

class TestFileIdInvolvedNonAscii(FileIdInvolvedBase):
    scenarios = all_repository_vf_format_scenarios()

    def test_utf8_file_ids_and_revision_ids(self):
        if False:
            return 10
        main_wt = self.make_branch_and_tree('main')
        main_branch = main_wt.branch
        self.build_tree(['main/a'])
        file_id = u'a-fíle-id'.encode('utf8')
        main_wt.add(['a'], [file_id])
        revision_id = u'rév-a'.encode('utf8')
        try:
            main_wt.commit('a', rev_id=revision_id)
        except errors.NonAsciiRevisionId:
            raise tests.TestSkipped('non-ascii revision ids not supported by %s' % self.repository_format)
        repo = main_wt.branch.repository
        repo.lock_read()
        self.addCleanup(repo.unlock)
        file_ids = repo.fileids_altered_by_revision_ids([revision_id])
        root_id = main_wt.basis_tree().get_root_id()
        if root_id in file_ids:
            self.assertEqual({file_id: set([revision_id]), root_id: set([revision_id])}, file_ids)
        else:
            self.assertEqual({file_id: set([revision_id])}, file_ids)

class TestFileIdInvolvedSuperset(FileIdInvolvedBase):
    scenarios = all_repository_vf_format_scenarios()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestFileIdInvolvedSuperset, self).setUp()
        self.branch = None
        main_wt = self.make_branch_and_tree('main')
        main_branch = main_wt.branch
        self.build_tree(['main/a', 'main/b', 'main/c'])
        main_wt.add(['a', 'b', 'c'], ['a-file-id-2006-01-01-abcd', 'b-file-id-2006-01-01-defg', 'c-funky<file-id>quiji\'"%bo'])
        try:
            main_wt.commit('Commit one', rev_id='rev-A')
        except errors.IllegalPath:
            if sys.platform == 'win32':
                raise tests.TestSkipped('Old repository formats do not support file ids with <> on win32')
            raise
        branch2_wt = self.make_branch_and_tree('branch2')
        branch2_wt.pull(main_branch)
        branch2_bzrdir = branch2_wt.bzrdir
        branch2_branch = branch2_bzrdir.open_branch()
        set_executability(branch2_wt, 'b', True)
        branch2_wt.commit('branch2, Commit one', rev_id='rev-J')
        main_wt.merge_from_branch(branch2_branch)
        set_executability(main_wt, 'b', False)
        main_wt.commit('merge branch1, rev-22', rev_id='rev-G')
        self.branch = main_branch

    def test_fileid_involved_full_compare2(self):
        if False:
            for i in range(10):
                print('nop')
        self.branch.lock_read()
        self.addCleanup(self.branch.unlock)
        graph = self.branch.repository.get_graph()
        history = list(graph.iter_lefthand_ancestry(self.branch.last_revision(), [_mod_revision.NULL_REVISION]))
        history.reverse()
        old_rev = history[0]
        new_rev = history[1]
        unique_revs = graph.find_unique_ancestors(new_rev, [old_rev])
        l1 = self.branch.repository.fileids_altered_by_revision_ids(unique_revs)
        l1 = set(l1.keys())
        l2 = self.compare_tree_fileids(self.branch, old_rev, new_rev)
        self.assertNotEqual(l2, l1)
        self.assertSubset(l2, l1)

def set_executability(wt, path, executable=True):
    if False:
        while True:
            i = 10
    "Set the executable bit for the file at path in the working tree\n\n    os.chmod() doesn't work on windows. But TreeTransform can mark or\n    unmark a file as executable.\n    "
    file_id = wt.path2id(path)
    tt = transform.TreeTransform(wt)
    try:
        tt.set_executability(executable, tt.trans_id_tree_file_id(file_id))
        tt.apply()
    finally:
        tt.finalize()