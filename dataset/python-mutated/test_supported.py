"""Tests for repositories that support CHK indices."""
from bzrlib import btree_index, errors, osutils, repository
from bzrlib.remote import RemoteRepository
from bzrlib.versionedfile import VersionedFiles
from bzrlib.tests import TestNotApplicable
from bzrlib.tests.per_repository_chk import TestCaseWithRepositoryCHK

class TestCHKSupport(TestCaseWithRepositoryCHK):

    def test_chk_bytes_attribute_is_VersionedFiles(self):
        if False:
            return 10
        repo = self.make_repository('.')
        self.assertIsInstance(repo.chk_bytes, VersionedFiles)

    def test_add_bytes_to_chk_bytes_store(self):
        if False:
            return 10
        repo = self.make_repository('.')
        repo.lock_write()
        try:
            repo.start_write_group()
            try:
                (sha1, len, _) = repo.chk_bytes.add_lines((None,), None, ['foo\n', 'bar\n'], random_id=True)
                self.assertEqual('4e48e2c9a3d2ca8a708cb0cc545700544efb5021', sha1)
                self.assertEqual(set([('sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)]), repo.chk_bytes.keys())
            except:
                repo.abort_write_group()
                raise
            else:
                repo.commit_write_group()
        finally:
            repo.unlock()
        repo.lock_read()
        try:
            self.assertEqual(set([('sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)]), repo.chk_bytes.keys())
        finally:
            repo.unlock()
        repo = repo.bzrdir.open_repository()
        repo.lock_read()
        try:
            self.assertEqual(set([('sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)]), repo.chk_bytes.keys())
        finally:
            repo.unlock()

    def test_pack_preserves_chk_bytes_store(self):
        if False:
            print('Hello World!')
        leaf_lines = ['chkleaf:\n', '0\n', '1\n', '0\n', '\n']
        leaf_sha1 = osutils.sha_strings(leaf_lines)
        node_lines = ['chknode:\n', '0\n', '1\n', '1\n', 'foo\n', '\x00sha1:%s\n' % (leaf_sha1,)]
        node_sha1 = osutils.sha_strings(node_lines)
        expected_set = set([('sha1:' + leaf_sha1,), ('sha1:' + node_sha1,)])
        repo = self.make_repository('.')
        repo.lock_write()
        try:
            repo.start_write_group()
            try:
                repo.chk_bytes.add_lines((None,), None, node_lines, random_id=True)
            except:
                repo.abort_write_group()
                raise
            else:
                repo.commit_write_group()
            repo.start_write_group()
            try:
                repo.chk_bytes.add_lines((None,), None, leaf_lines, random_id=True)
            except:
                repo.abort_write_group()
                raise
            else:
                repo.commit_write_group()
            repo.pack()
            self.assertEqual(expected_set, repo.chk_bytes.keys())
        finally:
            repo.unlock()
        repo = repo.bzrdir.open_repository()
        repo.lock_read()
        try:
            self.assertEqual(expected_set, repo.chk_bytes.keys())
        finally:
            repo.unlock()

    def test_chk_bytes_are_fully_buffered(self):
        if False:
            return 10
        repo = self.make_repository('.')
        repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        try:
            (sha1, len, _) = repo.chk_bytes.add_lines((None,), None, ['foo\n', 'bar\n'], random_id=True)
            self.assertEqual('4e48e2c9a3d2ca8a708cb0cc545700544efb5021', sha1)
            self.assertEqual(set([('sha1:4e48e2c9a3d2ca8a708cb0cc545700544efb5021',)]), repo.chk_bytes.keys())
        except:
            repo.abort_write_group()
            raise
        else:
            repo.commit_write_group()
        index = repo.chk_bytes._index._graph_index._indices[0]
        self.assertIsInstance(index, btree_index.BTreeGraphIndex)
        self.assertIs(type(index._leaf_node_cache), dict)
        repo2 = repository.Repository.open(self.get_url())
        repo2.lock_read()
        self.addCleanup(repo2.unlock)
        index = repo2.chk_bytes._index._graph_index._indices[0]
        self.assertIsInstance(index, btree_index.BTreeGraphIndex)
        self.assertIs(type(index._leaf_node_cache), dict)

class TestCommitWriteGroupIntegrityCheck(TestCaseWithRepositoryCHK):
    """Tests that commit_write_group prevents various kinds of invalid data
    from being committed to a CHK repository.
    """

    def reopen_repo_and_resume_write_group(self, repo):
        if False:
            i = 10
            return i + 15
        resume_tokens = repo.suspend_write_group()
        repo.unlock()
        reopened_repo = repo.bzrdir.open_repository()
        reopened_repo.lock_write()
        self.addCleanup(reopened_repo.unlock)
        reopened_repo.resume_write_group(resume_tokens)
        return reopened_repo

    def test_missing_chk_root_for_inventory(self):
        if False:
            return 10
        'commit_write_group fails with BzrCheckError when the chk root record\n        for a new inventory is missing.\n        '
        repo = self.make_repository('damaged-repo')
        builder = self.make_branch_builder('simple-branch')
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('file', 'file-id', 'file', 'content\n'))])
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        repo.lock_write()
        repo.start_write_group()
        text_keys = [('file-id', 'A-id'), ('root-id', 'A-id')]
        src_repo = b.repository
        repo.texts.insert_record_stream(src_repo.texts.get_record_stream(text_keys, 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([('A-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([('A-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def test_missing_chk_root_for_unchanged_inventory(self):
        if False:
            while True:
                i = 10
        'commit_write_group fails with BzrCheckError when the chk root record\n        for a new inventory is missing, even if the parent inventory is present\n        and has identical content (i.e. the same chk root).\n        \n        A stacked repository containing only a revision with an identical\n        inventory to its parent will still have the chk root records for those\n        inventories.\n\n        (In principle the chk records are unnecessary in this case, but in\n        practice bzr 2.0rc1 (at least) expects to find them.)\n        '
        repo = self.make_repository('damaged-repo')
        builder = self.make_branch_builder('simple-branch')
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None)), ('add', ('file', 'file-id', 'file', 'content\n'))])
        builder.build_snapshot('B-id', None, [])
        builder.build_snapshot('C-id', None, [])
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        inv_b = b.repository.get_inventory('B-id')
        inv_c = b.repository.get_inventory('C-id')
        if not isinstance(repo, RemoteRepository):
            self.assertEqual(inv_b.id_to_entry.key(), inv_c.id_to_entry.key())
        repo.lock_write()
        repo.start_write_group()
        src_repo = b.repository
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([('B-id',), ('C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([('C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def test_missing_chk_leaf_for_inventory(self):
        if False:
            i = 10
            return i + 15
        'commit_write_group fails with BzrCheckError when the chk root record\n        for a parent inventory of a new revision is missing.\n        '
        repo = self.make_repository('damaged-repo')
        if isinstance(repo, RemoteRepository):
            raise TestNotApplicable('Unable to obtain CHKInventory from remote repo')
        b = self.make_branch_with_multiple_chk_nodes()
        src_repo = b.repository
        src_repo.lock_read()
        self.addCleanup(src_repo.unlock)
        inv_b = src_repo.get_inventory('B-id')
        inv_c = src_repo.get_inventory('C-id')
        chk_root_keys_only = [inv_b.id_to_entry.key(), inv_b.parent_id_basename_to_file_id.key(), inv_c.id_to_entry.key(), inv_c.parent_id_basename_to_file_id.key()]
        all_chks = src_repo.chk_bytes.keys()
        key_to_drop = all_chks.difference(chk_root_keys_only).pop()
        all_chks.discard(key_to_drop)
        repo.lock_write()
        repo.start_write_group()
        repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(all_chks, 'unordered', True))
        repo.texts.insert_record_stream(src_repo.texts.get_record_stream(src_repo.texts.keys(), 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([('B-id',), ('C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([('C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def test_missing_chk_root_for_parent_inventory(self):
        if False:
            while True:
                i = 10
        'commit_write_group fails with BzrCheckError when the chk root record\n        for a parent inventory of a new revision is missing.\n        '
        repo = self.make_repository('damaged-repo')
        if isinstance(repo, RemoteRepository):
            raise TestNotApplicable('Unable to obtain CHKInventory from remote repo')
        b = self.make_branch_with_multiple_chk_nodes()
        b.lock_read()
        self.addCleanup(b.unlock)
        inv_c = b.repository.get_inventory('C-id')
        chk_keys_for_c_only = [inv_c.id_to_entry.key(), inv_c.parent_id_basename_to_file_id.key()]
        repo.lock_write()
        repo.start_write_group()
        src_repo = b.repository
        repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(chk_keys_for_c_only, 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([('B-id',), ('C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([('C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()

    def make_branch_with_multiple_chk_nodes(self):
        if False:
            print('Hello World!')
        builder = self.make_branch_builder('simple-branch')
        file_adds = []
        file_modifies = []
        for char in 'abc':
            name = char * 10000
            file_adds.append(('add', ('file-' + name, 'file-%s-id' % name, 'file', 'content %s\n' % name)))
            file_modifies.append(('modify', ('file-%s-id' % name, 'new content %s\n' % name)))
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None))] + file_adds)
        builder.build_snapshot('B-id', None, [])
        builder.build_snapshot('C-id', None, file_modifies)
        return builder.get_branch()

    def test_missing_text_record(self):
        if False:
            for i in range(10):
                print('nop')
        'commit_write_group fails with BzrCheckError when a text is missing.\n        '
        repo = self.make_repository('damaged-repo')
        b = self.make_branch_with_multiple_chk_nodes()
        src_repo = b.repository
        src_repo.lock_read()
        self.addCleanup(src_repo.unlock)
        all_texts = src_repo.texts.keys()
        all_texts.remove(('file-%s-id' % ('c' * 10000,), 'C-id'))
        repo.lock_write()
        repo.start_write_group()
        repo.chk_bytes.insert_record_stream(src_repo.chk_bytes.get_record_stream(src_repo.chk_bytes.keys(), 'unordered', True))
        repo.texts.insert_record_stream(src_repo.texts.get_record_stream(all_texts, 'unordered', True))
        repo.inventories.insert_record_stream(src_repo.inventories.get_record_stream([('B-id',), ('C-id',)], 'unordered', True))
        repo.revisions.insert_record_stream(src_repo.revisions.get_record_stream([('C-id',)], 'unordered', True))
        repo.add_fallback_repository(b.repository)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        reopened_repo = self.reopen_repo_and_resume_write_group(repo)
        self.assertRaises(errors.BzrCheckError, reopened_repo.commit_write_group)
        reopened_repo.abort_write_group()