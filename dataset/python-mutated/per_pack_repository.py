"""Tests for pack repositories.

These tests are repeated for all pack-based repository formats.
"""
from stat import S_ISDIR
from bzrlib.btree_index import BTreeGraphIndex
from bzrlib.index import GraphIndex
from bzrlib import controldir, errors, inventory, osutils, repository, revision as _mod_revision, tests, transport, ui
from bzrlib.repofmt.groupcompress_repo import RepositoryFormat2a
from bzrlib.smart import client
from bzrlib.tests import TestCaseWithTransport, TestNotApplicable
from bzrlib.transport import memory
from bzrlib.tests import test_server

class TestPackRepository(TestCaseWithTransport):
    """Tests to be repeated across all pack-based formats.

    The following are populated from the test scenario:

    :ivar format_name: Registered name fo the format to test.
    :ivar format_string: On-disk format marker.
    :ivar format_supports_external_lookups: Boolean.
    """

    def get_format(self):
        if False:
            return 10
        return controldir.format_registry.make_bzrdir(self.format_name)

    def test_attribute__fetch_order(self):
        if False:
            for i in range(10):
                print('nop')
        'Packs do not need ordered data retrieval.'
        format = self.get_format()
        repo = self.make_repository('.', format=format)
        self.assertEqual('unordered', repo._format._fetch_order)

    def test_attribute__fetch_uses_deltas(self):
        if False:
            while True:
                i = 10
        'Packs reuse deltas.'
        format = self.get_format()
        repo = self.make_repository('.', format=format)
        if isinstance(format.repository_format, RepositoryFormat2a):
            self.assertEqual(False, repo._format._fetch_uses_deltas)
        else:
            self.assertEqual(True, repo._format._fetch_uses_deltas)

    def test_disk_layout(self):
        if False:
            while True:
                i = 10
        format = self.get_format()
        repo = self.make_repository('.', format=format)
        repo.lock_write()
        repo.unlock()
        t = repo.bzrdir.get_repository_transport(None)
        self.check_format(t)
        self.check_databases(t)

    def check_format(self, t):
        if False:
            i = 10
            return i + 15
        self.assertEqualDiff(self.format_string, t.get('format').read())

    def assertHasNoKndx(self, t, knit_name):
        if False:
            return 10
        'Assert that knit_name has no index on t.'
        self.assertFalse(t.has(knit_name + '.kndx'))

    def assertHasNoKnit(self, t, knit_name):
        if False:
            while True:
                i = 10
        'Assert that knit_name exists on t.'
        self.assertFalse(t.has(knit_name + '.knit'))

    def check_databases(self, t):
        if False:
            print('Hello World!')
        'check knit content for a repository.'
        self.assertHasNoKndx(t, 'inventory')
        self.assertHasNoKnit(t, 'inventory')
        self.assertHasNoKndx(t, 'revisions')
        self.assertHasNoKnit(t, 'revisions')
        self.assertHasNoKndx(t, 'signatures')
        self.assertHasNoKnit(t, 'signatures')
        self.assertFalse(t.has('knits'))
        self.assertEqual([], list(self.index_class(t, 'pack-names', None).iter_all_entries()))
        self.assertTrue(S_ISDIR(t.stat('packs').st_mode))
        self.assertTrue(S_ISDIR(t.stat('upload').st_mode))
        self.assertTrue(S_ISDIR(t.stat('indices').st_mode))
        self.assertTrue(S_ISDIR(t.stat('obsolete_packs').st_mode))

    def test_shared_disk_layout(self):
        if False:
            for i in range(10):
                print('nop')
        format = self.get_format()
        repo = self.make_repository('.', shared=True, format=format)
        t = repo.bzrdir.get_repository_transport(None)
        self.check_format(t)
        self.assertEqualDiff('', t.get('shared-storage').read())
        self.check_databases(t)

    def test_shared_no_tree_disk_layout(self):
        if False:
            i = 10
            return i + 15
        format = self.get_format()
        repo = self.make_repository('.', shared=True, format=format)
        repo.set_make_working_trees(False)
        t = repo.bzrdir.get_repository_transport(None)
        self.check_format(t)
        self.assertEqualDiff('', t.get('shared-storage').read())
        self.assertEqualDiff('', t.get('no-working-trees').read())
        repo.set_make_working_trees(True)
        self.assertFalse(t.has('no-working-trees'))
        self.check_databases(t)

    def test_adding_revision_creates_pack_indices(self):
        if False:
            i = 10
            return i + 15
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        trans = tree.branch.repository.bzrdir.get_repository_transport(None)
        self.assertEqual([], list(self.index_class(trans, 'pack-names', None).iter_all_entries()))
        tree.commit('foobarbaz')
        index = self.index_class(trans, 'pack-names', None)
        index_nodes = list(index.iter_all_entries())
        self.assertEqual(1, len(index_nodes))
        node = index_nodes[0]
        name = node[1][0]
        pack_value = node[2]
        sizes = [int(digits) for digits in pack_value.split(' ')]
        for (size, suffix) in zip(sizes, ['.rix', '.iix', '.tix', '.six']):
            stat = trans.stat('indices/%s%s' % (name, suffix))
            self.assertEqual(size, stat.st_size)

    def test_pulling_nothing_leads_to_no_new_names(self):
        if False:
            while True:
                i = 10
        format = self.get_format()
        tree1 = self.make_branch_and_tree('1', format=format)
        tree2 = self.make_branch_and_tree('2', format=format)
        tree1.branch.repository.fetch(tree2.branch.repository)
        trans = tree1.branch.repository.bzrdir.get_repository_transport(None)
        self.assertEqual([], list(self.index_class(trans, 'pack-names', None).iter_all_entries()))

    def test_commit_across_pack_shape_boundary_autopacks(self):
        if False:
            while True:
                i = 10
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        trans = tree.branch.repository.bzrdir.get_repository_transport(None)
        for x in range(9):
            tree.commit('commit %s' % x)
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(9, len(list(index.iter_all_entries())))
        trans.put_bytes('obsolete_packs/foo', '123')
        trans.put_bytes('obsolete_packs/bar', '321')
        tree.commit('commit triggering pack')
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(1, len(list(index.iter_all_entries())))
        tree = tree.bzrdir.open_workingtree()
        check_result = tree.branch.repository.check([tree.branch.last_revision()])
        nb_files = 5
        if tree.branch.repository._format.supports_chks:
            nb_files += 1
        obsolete_files = list(trans.list_dir('obsolete_packs'))
        self.assertFalse('foo' in obsolete_files)
        self.assertFalse('bar' in obsolete_files)
        self.assertEqual(10 * nb_files, len(obsolete_files))
        large_pack_name = list(index.iter_all_entries())[0][1][0]
        tree.commit('commit not triggering pack')
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(2, len(list(index.iter_all_entries())))
        pack_names = [node[1][0] for node in index.iter_all_entries()]
        self.assertTrue(large_pack_name in pack_names)

    def test_commit_write_group_returns_new_pack_names(self):
        if False:
            return 10
        self.vfs_transport_factory = memory.MemoryServer
        format = self.get_format()
        repo = self.make_repository('foo', format=format)
        repo.lock_write()
        try:
            for pos in range(10):
                revid = str(pos)
                repo.start_write_group()
                try:
                    inv = inventory.Inventory(revision_id=revid)
                    inv.root.revision = revid
                    repo.texts.add_lines((inv.root.file_id, revid), [], [])
                    rev = _mod_revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', revision_id=revid)
                    rev.parent_ids = ()
                    repo.add_revision(revid, rev, inv=inv)
                except:
                    repo.abort_write_group()
                    raise
                else:
                    old_names = repo._pack_collection._names.keys()
                    result = repo.commit_write_group()
                    cur_names = repo._pack_collection._names.keys()
                    new_names = list(set(cur_names) - set(old_names))
                    self.assertEqual(new_names, result)
        finally:
            repo.unlock()

    def test_fail_obsolete_deletion(self):
        if False:
            while True:
                i = 10
        format = self.get_format()
        server = test_server.FakeNFSServer()
        self.start_server(server)
        t = transport.get_transport_from_url(server.get_url())
        bzrdir = self.get_format().initialize_on_transport(t)
        repo = bzrdir.create_repository()
        repo_transport = bzrdir.get_repository_transport(None)
        self.assertTrue(repo_transport.has('obsolete_packs'))
        repo_transport.put_bytes('obsolete_packs/.nfsblahblah', 'contents')
        repo._pack_collection._clear_obsolete_packs()
        self.assertTrue(repo_transport.has('obsolete_packs/.nfsblahblah'))

    def test_pack_collection_sets_sibling_indices(self):
        if False:
            print('Hello World!')
        'The CombinedGraphIndex objects in the pack collection are all\n        siblings of each other, so that search-order reorderings will be copied\n        to each other.\n        '
        repo = self.make_repository('repo')
        pack_coll = repo._pack_collection
        indices = set([pack_coll.revision_index, pack_coll.inventory_index, pack_coll.text_index, pack_coll.signature_index])
        if pack_coll.chk_index is not None:
            indices.add(pack_coll.chk_index)
        combined_indices = set((idx.combined_index for idx in indices))
        for combined_index in combined_indices:
            self.assertEqual(combined_indices.difference([combined_index]), combined_index._sibling_indices)

    def test_pack_after_two_commits_packs_everything(self):
        if False:
            for i in range(10):
                print('nop')
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        trans = tree.branch.repository.bzrdir.get_repository_transport(None)
        tree.commit('start')
        tree.commit('more work')
        tree.branch.repository.pack()
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(1, len(list(index.iter_all_entries())))
        self.assertEqual(2, len(tree.branch.repository.all_revision_ids()))

    def test_pack_preserves_all_inventories(self):
        if False:
            while True:
                i = 10
        format = self.get_format()
        builder = self.make_branch_builder('source', format=format)
        builder.start_series()
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None))])
        builder.build_snapshot('B-id', None, [('add', ('file', 'file-id', 'file', 'B content\n'))])
        builder.build_snapshot('C-id', None, [('modify', ('file-id', 'C content\n'))])
        builder.finish_series()
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        repo = self.make_repository('repo', shared=True, format=format)
        repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.fetch(b.repository, revision_id='B-id')
        inv = b.repository.iter_inventories(['C-id']).next()
        repo.start_write_group()
        repo.add_inventory('C-id', inv, ['B-id'])
        repo.commit_write_group()
        self.assertEqual([('A-id',), ('B-id',), ('C-id',)], sorted(repo.inventories.keys()))
        repo.pack()
        self.assertEqual([('A-id',), ('B-id',), ('C-id',)], sorted(repo.inventories.keys()))
        self.assertEqual(inv, repo.iter_inventories(['C-id']).next())

    def test_pack_layout(self):
        if False:
            i = 10
            return i + 15
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        trans = tree.branch.repository.bzrdir.get_repository_transport(None)
        tree.commit('start', rev_id='1')
        tree.commit('more work', rev_id='2')
        tree.branch.repository.pack()
        tree.lock_read()
        self.addCleanup(tree.unlock)
        pack = tree.branch.repository._pack_collection.get_pack_by_name(tree.branch.repository._pack_collection.names()[0])
        for (_1, key, val, refs) in pack.revision_index.iter_all_entries():
            if type(format.repository_format) is RepositoryFormat2a:
                pos = map(int, val.split())
            else:
                pos = int(val[1:].split()[0])
            if key == ('1',):
                pos_1 = pos
            else:
                pos_2 = pos
        self.assertTrue(pos_2 < pos_1, 'rev 1 came before rev 2 %s > %s' % (pos_1, pos_2))

    def test_pack_repositories_support_multiple_write_locks(self):
        if False:
            return 10
        format = self.get_format()
        self.make_repository('.', shared=True, format=format)
        r1 = repository.Repository.open('.')
        r2 = repository.Repository.open('.')
        r1.lock_write()
        self.addCleanup(r1.unlock)
        r2.lock_write()
        r2.unlock()

    def _add_text(self, repo, fileid):
        if False:
            for i in range(10):
                print('nop')
        'Add a text to the repository within a write group.'
        repo.texts.add_lines((fileid, 'samplerev+' + fileid), [], ['smaplerev+' + fileid])

    def test_concurrent_writers_merge_new_packs(self):
        if False:
            print('Hello World!')
        format = self.get_format()
        self.make_repository('.', shared=True, format=format)
        r1 = repository.Repository.open('.')
        r2 = repository.Repository.open('.')
        r1.lock_write()
        try:
            list(r1.all_revision_ids())
            r2.lock_write()
            try:
                list(r2.all_revision_ids())
                r1.start_write_group()
                try:
                    r2.start_write_group()
                    try:
                        self._add_text(r1, 'fileidr1')
                        self._add_text(r2, 'fileidr2')
                    except:
                        r2.abort_write_group()
                        raise
                except:
                    r1.abort_write_group()
                    raise
                try:
                    r1.commit_write_group()
                except:
                    r1.abort_write_group()
                    r2.abort_write_group()
                    raise
                r2.commit_write_group()
                r1._pack_collection.reset()
                r1._pack_collection.ensure_loaded()
                r2._pack_collection.ensure_loaded()
                self.assertEqual(r1._pack_collection.names(), r2._pack_collection.names())
                self.assertEqual(2, len(r1._pack_collection.names()))
            finally:
                r2.unlock()
        finally:
            r1.unlock()

    def test_concurrent_writer_second_preserves_dropping_a_pack(self):
        if False:
            print('Hello World!')
        format = self.get_format()
        self.make_repository('.', shared=True, format=format)
        r1 = repository.Repository.open('.')
        r2 = repository.Repository.open('.')
        r1.lock_write()
        try:
            r1.start_write_group()
            try:
                self._add_text(r1, 'fileidr1')
            except:
                r1.abort_write_group()
                raise
            else:
                r1.commit_write_group()
            r1._pack_collection.ensure_loaded()
            name_to_drop = r1._pack_collection.all_packs()[0].name
        finally:
            r1.unlock()
        r1.lock_write()
        try:
            list(r1.all_revision_ids())
            r2.lock_write()
            try:
                list(r2.all_revision_ids())
                r1._pack_collection.ensure_loaded()
                try:
                    r2.start_write_group()
                    try:
                        r1._pack_collection._remove_pack_from_memory(r1._pack_collection.get_pack_by_name(name_to_drop))
                        self._add_text(r2, 'fileidr2')
                    except:
                        r2.abort_write_group()
                        raise
                except:
                    r1._pack_collection.reset()
                    raise
                try:
                    r1._pack_collection._save_pack_names()
                    r1._pack_collection.reset()
                except:
                    r2.abort_write_group()
                    raise
                try:
                    r2.commit_write_group()
                except:
                    r2.abort_write_group()
                    raise
                r1._pack_collection.ensure_loaded()
                r2._pack_collection.ensure_loaded()
                self.assertEqual(r1._pack_collection.names(), r2._pack_collection.names())
                self.assertEqual(1, len(r1._pack_collection.names()))
                self.assertFalse(name_to_drop in r1._pack_collection.names())
            finally:
                r2.unlock()
        finally:
            r1.unlock()

    def test_concurrent_pack_triggers_reload(self):
        if False:
            print('Hello World!')
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        try:
            rev1 = tree.commit('one')
            rev2 = tree.commit('two')
            r2 = repository.Repository.open('tree')
            r2.lock_read()
            try:
                tree.branch.repository.pack()
                self.assertEqual({rev2: (rev1,)}, r2.get_parent_map([rev2]))
            finally:
                r2.unlock()
        finally:
            tree.unlock()

    def test_concurrent_pack_during_get_record_reloads(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        try:
            rev1 = tree.commit('one')
            rev2 = tree.commit('two')
            keys = [(rev1,), (rev2,)]
            r2 = repository.Repository.open('tree')
            r2.lock_read()
            try:
                packed = False
                result = {}
                record_stream = r2.revisions.get_record_stream(keys, 'unordered', False)
                for record in record_stream:
                    result[record.key] = record
                    if not packed:
                        tree.branch.repository.pack()
                        packed = True
                self.assertEqual(sorted(keys), sorted(result.keys()))
            finally:
                r2.unlock()
        finally:
            tree.unlock()

    def test_concurrent_pack_during_autopack(self):
        if False:
            for i in range(10):
                print('nop')
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        try:
            for i in xrange(9):
                tree.commit('rev %d' % (i,))
            r2 = repository.Repository.open('tree')
            r2.lock_write()
            try:
                autopack_count = [0]
                r1 = tree.branch.repository
                orig = r1._pack_collection.pack_distribution

                def trigger_during_auto(*args, **kwargs):
                    if False:
                        i = 10
                        return i + 15
                    ret = orig(*args, **kwargs)
                    if not autopack_count[0]:
                        r2.pack()
                    autopack_count[0] += 1
                    return ret
                r1._pack_collection.pack_distribution = trigger_during_auto
                tree.commit('autopack-rev')
                self.assertEqual([2], autopack_count)
            finally:
                r2.unlock()
        finally:
            tree.unlock()

    def test_lock_write_does_not_physically_lock(self):
        if False:
            while True:
                i = 10
        repo = self.make_repository('.', format=self.get_format())
        repo.lock_write()
        self.addCleanup(repo.unlock)
        self.assertFalse(repo.get_physical_lock_status())

    def prepare_for_break_lock(self):
        if False:
            for i in range(10):
                print('nop')
        ui.ui_factory = ui.CannedInputUIFactory([True])

    def test_break_lock_breaks_physical_lock(self):
        if False:
            for i in range(10):
                print('nop')
        repo = self.make_repository('.', format=self.get_format())
        repo._pack_collection.lock_names()
        repo.control_files.leave_in_place()
        repo.unlock()
        repo2 = repository.Repository.open('.')
        self.assertTrue(repo.get_physical_lock_status())
        self.prepare_for_break_lock()
        repo2.break_lock()
        self.assertFalse(repo.get_physical_lock_status())

    def test_broken_physical_locks_error_on__unlock_names_lock(self):
        if False:
            return 10
        repo = self.make_repository('.', format=self.get_format())
        repo._pack_collection.lock_names()
        self.assertTrue(repo.get_physical_lock_status())
        repo2 = repository.Repository.open('.')
        self.prepare_for_break_lock()
        repo2.break_lock()
        self.assertRaises(errors.LockBroken, repo._pack_collection._unlock_names)

    def test_fetch_without_find_ghosts_ignores_ghosts(self):
        if False:
            while True:
                i = 10
        has_ghost = self.make_repository('has_ghost', format=self.get_format())
        missing_ghost = self.make_repository('missing_ghost', format=self.get_format())

        def add_commit(repo, revision_id, parent_ids):
            if False:
                for i in range(10):
                    print('nop')
            repo.lock_write()
            repo.start_write_group()
            inv = inventory.Inventory(revision_id=revision_id)
            inv.root.revision = revision_id
            root_id = inv.root.file_id
            sha1 = repo.add_inventory(revision_id, inv, [])
            repo.texts.add_lines((root_id, revision_id), [], [])
            rev = _mod_revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=revision_id)
            rev.parent_ids = parent_ids
            repo.add_revision(revision_id, rev)
            repo.commit_write_group()
            repo.unlock()
        add_commit(has_ghost, 'ghost', [])
        add_commit(has_ghost, 'references', ['ghost'])
        add_commit(missing_ghost, 'references', ['ghost'])
        add_commit(has_ghost, 'tip', ['references'])
        missing_ghost.fetch(has_ghost, 'tip')
        rev = missing_ghost.get_revision('tip')
        inv = missing_ghost.get_inventory('tip')
        self.assertRaises(errors.NoSuchRevision, missing_ghost.get_revision, 'ghost')
        self.assertRaises(errors.NoSuchRevision, missing_ghost.get_inventory, 'ghost')

    def make_write_ready_repo(self):
        if False:
            print('Hello World!')
        format = self.get_format()
        if isinstance(format.repository_format, RepositoryFormat2a):
            raise TestNotApplicable('No missing compression parents')
        repo = self.make_repository('.', format=format)
        repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        self.addCleanup(repo.abort_write_group)
        return repo

    def test_missing_inventories_compression_parent_prevents_commit(self):
        if False:
            while True:
                i = 10
        repo = self.make_write_ready_repo()
        key = ('junk',)
        repo.inventories._index._missing_compression_parents.add(key)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)

    def test_missing_revisions_compression_parent_prevents_commit(self):
        if False:
            i = 10
            return i + 15
        repo = self.make_write_ready_repo()
        key = ('junk',)
        repo.revisions._index._missing_compression_parents.add(key)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)

    def test_missing_signatures_compression_parent_prevents_commit(self):
        if False:
            return 10
        repo = self.make_write_ready_repo()
        key = ('junk',)
        repo.signatures._index._missing_compression_parents.add(key)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)

    def test_missing_text_compression_parent_prevents_commit(self):
        if False:
            while True:
                i = 10
        repo = self.make_write_ready_repo()
        key = ('some', 'junk')
        repo.texts._index._missing_compression_parents.add(key)
        self.assertRaises(errors.BzrCheckError, repo.commit_write_group)
        e = self.assertRaises(errors.BzrCheckError, repo.commit_write_group)

    def test_supports_external_lookups(self):
        if False:
            i = 10
            return i + 15
        repo = self.make_repository('.', format=self.get_format())
        self.assertEqual(self.format_supports_external_lookups, repo._format.supports_external_lookups)

    def _lock_write(self, write_lockable):
        if False:
            i = 10
            return i + 15
        'Lock write_lockable, add a cleanup and return the result.\n        \n        :param write_lockable: An object with a lock_write method.\n        :return: The result of write_lockable.lock_write().\n        '
        result = write_lockable.lock_write()
        self.addCleanup(result.unlock)
        return result

    def test_abort_write_group_does_not_raise_when_suppressed(self):
        if False:
            i = 10
            return i + 15
        "Similar to per_repository.test_write_group's test of the same name.\n\n        Also requires that the exception is logged.\n        "
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo', format=self.get_format())
        token = self._lock_write(repo).repository_token
        repo.start_write_group()
        self.get_transport('').rename('repo', 'foo')
        self.assertEqual(None, repo.abort_write_group(suppress_errors=True))
        log = self.get_log()
        self.assertContainsRe(log, 'abort_write_group failed')
        self.assertContainsRe(log, 'INFO  bzr: ERROR \\(ignored\\):')
        if token is not None:
            repo.leave_lock_in_place()

    def test_abort_write_group_does_raise_when_not_suppressed(self):
        if False:
            print('Hello World!')
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo', format=self.get_format())
        token = self._lock_write(repo).repository_token
        repo.start_write_group()
        self.get_transport('').rename('repo', 'foo')
        self.assertRaises(Exception, repo.abort_write_group)
        if token is not None:
            repo.leave_lock_in_place()

    def test_suspend_write_group(self):
        if False:
            for i in range(10):
                print('nop')
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo', format=self.get_format())
        token = self._lock_write(repo).repository_token
        repo.start_write_group()
        repo.texts.add_lines(('file-id', 'revid'), (), ['lines'])
        wg_tokens = repo.suspend_write_group()
        expected_pack_name = wg_tokens[0] + '.pack'
        expected_names = [wg_tokens[0] + ext for ext in ('.rix', '.iix', '.tix', '.six')]
        if repo.chk_bytes is not None:
            expected_names.append(wg_tokens[0] + '.cix')
        expected_names.append(expected_pack_name)
        upload_transport = repo._pack_collection._upload_transport
        limbo_files = upload_transport.list_dir('')
        self.assertEqual(sorted(expected_names), sorted(limbo_files))
        md5 = osutils.md5(upload_transport.get_bytes(expected_pack_name))
        self.assertEqual(wg_tokens[0], md5.hexdigest())

    def test_resume_chk_bytes(self):
        if False:
            i = 10
            return i + 15
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo', format=self.get_format())
        if repo.chk_bytes is None:
            raise TestNotApplicable('no chk_bytes for this repository')
        token = self._lock_write(repo).repository_token
        repo.start_write_group()
        text = 'a bit of text\n'
        key = ('sha1:' + osutils.sha_string(text),)
        repo.chk_bytes.add_lines(key, (), [text])
        wg_tokens = repo.suspend_write_group()
        same_repo = repo.bzrdir.open_repository()
        same_repo.lock_write()
        self.addCleanup(same_repo.unlock)
        same_repo.resume_write_group(wg_tokens)
        self.assertEqual([key], list(same_repo.chk_bytes.keys()))
        self.assertEqual(text, same_repo.chk_bytes.get_record_stream([key], 'unordered', True).next().get_bytes_as('fulltext'))
        same_repo.abort_write_group()
        self.assertEqual([], list(same_repo.chk_bytes.keys()))

    def test_resume_write_group_then_abort(self):
        if False:
            print('Hello World!')
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo', format=self.get_format())
        token = self._lock_write(repo).repository_token
        repo.start_write_group()
        text_key = ('file-id', 'revid')
        repo.texts.add_lines(text_key, (), ['lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = repo.bzrdir.open_repository()
        same_repo.lock_write()
        self.addCleanup(same_repo.unlock)
        same_repo.resume_write_group(wg_tokens)
        same_repo.abort_write_group()
        self.assertEqual([], same_repo._pack_collection._upload_transport.list_dir(''))
        self.assertEqual([], same_repo._pack_collection._pack_transport.list_dir(''))

    def test_commit_resumed_write_group(self):
        if False:
            for i in range(10):
                print('nop')
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo', format=self.get_format())
        token = self._lock_write(repo).repository_token
        repo.start_write_group()
        text_key = ('file-id', 'revid')
        repo.texts.add_lines(text_key, (), ['lines'])
        wg_tokens = repo.suspend_write_group()
        same_repo = repo.bzrdir.open_repository()
        same_repo.lock_write()
        self.addCleanup(same_repo.unlock)
        same_repo.resume_write_group(wg_tokens)
        same_repo.commit_write_group()
        expected_pack_name = wg_tokens[0] + '.pack'
        expected_names = [wg_tokens[0] + ext for ext in ('.rix', '.iix', '.tix', '.six')]
        if repo.chk_bytes is not None:
            expected_names.append(wg_tokens[0] + '.cix')
        self.assertEqual([], same_repo._pack_collection._upload_transport.list_dir(''))
        index_names = repo._pack_collection._index_transport.list_dir('')
        self.assertEqual(sorted(expected_names), sorted(index_names))
        pack_names = repo._pack_collection._pack_transport.list_dir('')
        self.assertEqual([expected_pack_name], pack_names)

    def test_resume_malformed_token(self):
        if False:
            while True:
                i = 10
        self.vfs_transport_factory = memory.MemoryServer
        repo = self.make_repository('repo', format=self.get_format())
        token = self._lock_write(repo).repository_token
        repo.start_write_group()
        text_key = ('file-id', 'revid')
        repo.texts.add_lines(text_key, (), ['lines'])
        wg_tokens = repo.suspend_write_group()
        new_repo = self.make_repository('new_repo', format=self.get_format())
        token = self._lock_write(new_repo).repository_token
        hacked_wg_token = '../../../../repo/.bzr/repository/upload/' + wg_tokens[0]
        self.assertRaises(errors.UnresumableWriteGroup, new_repo.resume_write_group, [hacked_wg_token])

class TestPackRepositoryStacking(TestCaseWithTransport):
    """Tests for stacking pack repositories"""

    def setUp(self):
        if False:
            print('Hello World!')
        if not self.format_supports_external_lookups:
            raise TestNotApplicable("%r doesn't support stacking" % (self.format_name,))
        super(TestPackRepositoryStacking, self).setUp()

    def get_format(self):
        if False:
            print('Hello World!')
        return controldir.format_registry.make_bzrdir(self.format_name)

    def test_stack_checks_rich_root_compatibility(self):
        if False:
            i = 10
            return i + 15
        repo = self.make_repository('repo', format=self.get_format())
        if repo.supports_rich_root():
            if getattr(repo._format, 'supports_tree_reference', False):
                matching_format_name = 'pack-0.92-subtree'
            elif repo._format.supports_chks:
                matching_format_name = '2a'
            else:
                matching_format_name = 'rich-root-pack'
            mismatching_format_name = 'pack-0.92'
        else:
            if repo._format.supports_chks:
                raise AssertionError('no non-rich-root CHK formats known')
            else:
                matching_format_name = 'pack-0.92'
            mismatching_format_name = 'pack-0.92-subtree'
        base = self.make_repository('base', format=matching_format_name)
        repo.add_fallback_repository(base)
        bad_repo = self.make_repository('mismatch', format=mismatching_format_name)
        e = self.assertRaises(errors.IncompatibleRepositories, repo.add_fallback_repository, bad_repo)
        self.assertContainsRe(str(e), '(?m)KnitPackRepository.*/mismatch/.*\\nis not compatible with\\n.*Repository.*/repo/.*\\ndifferent rich-root support')

    def test_stack_checks_serializers_compatibility(self):
        if False:
            while True:
                i = 10
        repo = self.make_repository('repo', format=self.get_format())
        if getattr(repo._format, 'supports_tree_reference', False):
            matching_format_name = 'pack-0.92-subtree'
            mismatching_format_name = 'rich-root-pack'
        elif repo.supports_rich_root():
            if repo._format.supports_chks:
                matching_format_name = '2a'
            else:
                matching_format_name = 'rich-root-pack'
            mismatching_format_name = 'pack-0.92-subtree'
        else:
            raise TestNotApplicable('No formats use non-v5 serializer without having rich-root also set')
        base = self.make_repository('base', format=matching_format_name)
        repo.add_fallback_repository(base)
        bad_repo = self.make_repository('mismatch', format=mismatching_format_name)
        e = self.assertRaises(errors.IncompatibleRepositories, repo.add_fallback_repository, bad_repo)
        self.assertContainsRe(str(e), '(?m)KnitPackRepository.*/mismatch/.*\\nis not compatible with\\n.*Repository.*/repo/.*\\ndifferent serializers')

    def test_adding_pack_does_not_record_pack_names_from_other_repositories(self):
        if False:
            return 10
        base = self.make_branch_and_tree('base', format=self.get_format())
        base.commit('foo')
        referencing = self.make_branch_and_tree('repo', format=self.get_format())
        referencing.branch.repository.add_fallback_repository(base.branch.repository)
        local_tree = referencing.branch.create_checkout('local')
        local_tree.commit('bar')
        new_instance = referencing.bzrdir.open_repository()
        new_instance.lock_read()
        self.addCleanup(new_instance.unlock)
        new_instance._pack_collection.ensure_loaded()
        self.assertEqual(1, len(new_instance._pack_collection.all_packs()))

    def test_autopack_only_considers_main_repo_packs(self):
        if False:
            return 10
        format = self.get_format()
        base = self.make_branch_and_tree('base', format=format)
        base.commit('foo')
        tree = self.make_branch_and_tree('repo', format=format)
        tree.branch.repository.add_fallback_repository(base.branch.repository)
        trans = tree.branch.repository.bzrdir.get_repository_transport(None)
        local_tree = tree.branch.create_checkout('local')
        for x in range(9):
            local_tree.commit('commit %s' % x)
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(9, len(list(index.iter_all_entries())))
        local_tree.commit('commit triggering pack')
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(1, len(list(index.iter_all_entries())))
        tree = tree.bzrdir.open_workingtree()
        check_result = tree.branch.repository.check([tree.branch.last_revision()])
        nb_files = 5
        if tree.branch.repository._format.supports_chks:
            nb_files += 1
        obsolete_files = list(trans.list_dir('obsolete_packs'))
        self.assertFalse('foo' in obsolete_files)
        self.assertFalse('bar' in obsolete_files)
        self.assertEqual(10 * nb_files, len(obsolete_files))
        large_pack_name = list(index.iter_all_entries())[0][1][0]
        local_tree.commit('commit not triggering pack')
        index = self.index_class(trans, 'pack-names', None)
        self.assertEqual(2, len(list(index.iter_all_entries())))
        pack_names = [node[1][0] for node in index.iter_all_entries()]
        self.assertTrue(large_pack_name in pack_names)

class TestKeyDependencies(TestCaseWithTransport):

    def get_format(self):
        if False:
            for i in range(10):
                print('nop')
        return controldir.format_registry.make_bzrdir(self.format_name)

    def create_source_and_target(self):
        if False:
            i = 10
            return i + 15
        builder = self.make_branch_builder('source', format=self.get_format())
        builder.start_series()
        builder.build_snapshot('A-id', None, [('add', ('', 'root-id', 'directory', None))])
        builder.build_snapshot('B-id', ['A-id', 'ghost-id'], [])
        builder.finish_series()
        repo = self.make_repository('target', format=self.get_format())
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        repo.lock_write()
        self.addCleanup(repo.unlock)
        return (b.repository, repo)

    def test_key_dependencies_cleared_on_abort(self):
        if False:
            return 10
        (source_repo, target_repo) = self.create_source_and_target()
        target_repo.start_write_group()
        try:
            stream = source_repo.revisions.get_record_stream([('B-id',)], 'unordered', True)
            target_repo.revisions.insert_record_stream(stream)
            key_refs = target_repo.revisions._index._key_dependencies
            self.assertEqual([('B-id',)], sorted(key_refs.get_referrers()))
        finally:
            target_repo.abort_write_group()
        self.assertEqual([], sorted(key_refs.get_referrers()))

    def test_key_dependencies_cleared_on_suspend(self):
        if False:
            print('Hello World!')
        (source_repo, target_repo) = self.create_source_and_target()
        target_repo.start_write_group()
        try:
            stream = source_repo.revisions.get_record_stream([('B-id',)], 'unordered', True)
            target_repo.revisions.insert_record_stream(stream)
            key_refs = target_repo.revisions._index._key_dependencies
            self.assertEqual([('B-id',)], sorted(key_refs.get_referrers()))
        finally:
            target_repo.suspend_write_group()
        self.assertEqual([], sorted(key_refs.get_referrers()))

    def test_key_dependencies_cleared_on_commit(self):
        if False:
            for i in range(10):
                print('nop')
        (source_repo, target_repo) = self.create_source_and_target()
        target_repo.start_write_group()
        try:
            for vf_name in ['texts', 'chk_bytes', 'inventories']:
                source_vf = getattr(source_repo, vf_name, None)
                if source_vf is None:
                    continue
                target_vf = getattr(target_repo, vf_name)
                stream = source_vf.get_record_stream(source_vf.keys(), 'unordered', True)
                target_vf.insert_record_stream(stream)
            stream = source_repo.revisions.get_record_stream([('B-id',)], 'unordered', True)
            target_repo.revisions.insert_record_stream(stream)
            key_refs = target_repo.revisions._index._key_dependencies
            self.assertEqual([('B-id',)], sorted(key_refs.get_referrers()))
        finally:
            target_repo.commit_write_group()
        self.assertEqual([], sorted(key_refs.get_referrers()))

class TestSmartServerAutopack(TestCaseWithTransport):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestSmartServerAutopack, self).setUp()
        self.smart_server = test_server.SmartTCPServer_for_testing()
        self.start_server(self.smart_server, self.get_server())
        client._SmartClient.hooks.install_named_hook('call', self.capture_hpss_call, None)
        self.hpss_calls = []

    def capture_hpss_call(self, params):
        if False:
            i = 10
            return i + 15
        self.hpss_calls.append(params.method)

    def get_format(self):
        if False:
            print('Hello World!')
        return controldir.format_registry.make_bzrdir(self.format_name)

    def test_autopack_or_streaming_rpc_is_used_when_using_hpss(self):
        if False:
            for i in range(10):
                print('nop')
        format = self.get_format()
        tree = self.make_branch_and_tree('local', format=format)
        self.make_branch_and_tree('remote', format=format)
        remote_branch_url = self.smart_server.get_url() + 'remote'
        remote_branch = controldir.ControlDir.open(remote_branch_url).open_branch()
        for x in range(9):
            tree.commit('commit %s' % x)
            tree.branch.push(remote_branch)
        self.hpss_calls = []
        tree.commit('commit triggering pack')
        tree.branch.push(remote_branch)
        autopack_calls = len([call for call in self.hpss_calls if call == 'PackRepository.autopack'])
        streaming_calls = len([call for call in self.hpss_calls if call in ('Repository.insert_stream', 'Repository.insert_stream_1.19')])
        if autopack_calls:
            self.assertEqual(1, autopack_calls)
            self.assertEqual(0, streaming_calls)
        else:
            self.assertEqual(0, autopack_calls)
            self.assertEqual(2, streaming_calls)

def load_tests(basic_tests, module, loader):
    if False:
        print('Hello World!')
    scenarios_params = [dict(format_name='pack-0.92', format_string='Bazaar pack repository format 1 (needs bzr 0.92)\n', format_supports_external_lookups=False, index_class=GraphIndex), dict(format_name='pack-0.92-subtree', format_string='Bazaar pack repository format 1 with subtree support (needs bzr 0.92)\n', format_supports_external_lookups=False, index_class=GraphIndex), dict(format_name='1.6', format_string='Bazaar RepositoryFormatKnitPack5 (bzr 1.6)\n', format_supports_external_lookups=True, index_class=GraphIndex), dict(format_name='1.6.1-rich-root', format_string='Bazaar RepositoryFormatKnitPack5RichRoot (bzr 1.6.1)\n', format_supports_external_lookups=True, index_class=GraphIndex), dict(format_name='1.9', format_string='Bazaar RepositoryFormatKnitPack6 (bzr 1.9)\n', format_supports_external_lookups=True, index_class=BTreeGraphIndex), dict(format_name='1.9-rich-root', format_string='Bazaar RepositoryFormatKnitPack6RichRoot (bzr 1.9)\n', format_supports_external_lookups=True, index_class=BTreeGraphIndex), dict(format_name='2a', format_string='Bazaar repository format 2a (needs bzr 1.16 or later)\n', format_supports_external_lookups=True, index_class=BTreeGraphIndex)]
    scenarios = [(s['format_name'], s) for s in scenarios_params]
    return tests.multiply_tests(basic_tests, scenarios, loader.suiteClass())