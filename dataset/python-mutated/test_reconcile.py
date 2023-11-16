"""Tests for reconciliation of repositories."""
import bzrlib
from bzrlib import errors
from bzrlib.bzrdir import BzrDir
from bzrlib.inventory import Inventory
from bzrlib.reconcile import reconcile, Reconciler
from bzrlib.revision import Revision
from bzrlib.tests import TestSkipped
from bzrlib.tests.per_repository_vf.helpers import TestCaseWithBrokenRevisionIndex
from bzrlib.tests.per_repository_vf import TestCaseWithRepository, all_repository_vf_format_scenarios
from bzrlib.tests.matchers import MatchesAncestry
from bzrlib.tests.scenarios import load_tests_apply_scenarios
from bzrlib.uncommit import uncommit
load_tests = load_tests_apply_scenarios

class TestReconcile(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def checkUnreconciled(self, d, reconciler):
        if False:
            print('Hello World!')
        'Check that d did not get reconciled.'
        self.assertEqual(0, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        self.checkNoBackupInventory(d)

    def checkNoBackupInventory(self, aBzrDir):
        if False:
            print('Hello World!')
        'Check that there is no backup inventory in aBzrDir.'
        repo = aBzrDir.open_repository()
        for path in repo.control_transport.list_dir('.'):
            self.assertFalse('inventory.backup' in path)

class TestBadRevisionParents(TestCaseWithBrokenRevisionIndex):
    scenarios = all_repository_vf_format_scenarios()

    def test_aborts_if_bad_parents_in_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Reconcile refuses to proceed if the revision index is wrong when\n        checked against the revision texts, so that it does not generate broken\n        data.\n\n        Ideally reconcile would fix this, but until we implement that we just\n        make sure we safely detect this problem.\n        '
        repo = self.make_repo_with_extra_ghost_index()
        reconciler = repo.reconcile(thorough=True)
        self.assertTrue(reconciler.aborted, 'reconcile should have aborted due to bad parents.')

    def test_does_not_abort_on_clean_repo(self):
        if False:
            while True:
                i = 10
        repo = self.make_repository('.')
        reconciler = repo.reconcile(thorough=True)
        self.assertFalse(reconciler.aborted, 'reconcile should not have aborted on an unbroken repository.')

class TestsNeedingReweave(TestReconcile):

    def setUp(self):
        if False:
            return 10
        super(TestsNeedingReweave, self).setUp()
        t = self.get_transport()
        repo = self.make_repository('inventory_without_revision')
        repo.lock_write()
        repo.start_write_group()
        inv = Inventory(revision_id='missing')
        inv.root.revision = 'missing'
        repo.add_inventory('missing', inv, [])
        repo.commit_write_group()
        repo.unlock()

        def add_commit(repo, revision_id, parent_ids):
            if False:
                while True:
                    i = 10
            repo.lock_write()
            repo.start_write_group()
            inv = Inventory(revision_id=revision_id)
            inv.root.revision = revision_id
            root_id = inv.root.file_id
            sha1 = repo.add_inventory(revision_id, inv, parent_ids)
            repo.texts.add_lines((root_id, revision_id), [], [])
            rev = bzrlib.revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id=revision_id)
            rev.parent_ids = parent_ids
            repo.add_revision(revision_id, rev)
            repo.commit_write_group()
            repo.unlock()
        repo = self.make_repository('inventory_without_revision_and_ghost')
        repo.lock_write()
        repo.start_write_group()
        repo.add_inventory('missing', inv, [])
        repo.commit_write_group()
        repo.unlock()
        add_commit(repo, 'references_missing', ['missing'])
        repo = self.make_repository('inventory_one_ghost')
        add_commit(repo, 'ghost', ['the_ghost'])
        t.copy_tree('inventory_one_ghost', 'inventory_ghost_present')
        bzrdir_url = self.get_url('inventory_ghost_present')
        bzrdir = BzrDir.open(bzrdir_url)
        repo = bzrdir.open_repository()
        add_commit(repo, 'the_ghost', [])

    def checkEmptyReconcile(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Check a reconcile on an empty repository.'
        self.make_repository('empty')
        d = BzrDir.open(self.get_url('empty'))
        reconciler = d.find_repository().reconcile(**kwargs)
        self.assertEqual(0, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        self.checkNoBackupInventory(d)

    def test_reconcile_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.checkEmptyReconcile()

    def test_repo_has_reconcile_does_inventory_gc_attribute(self):
        if False:
            i = 10
            return i + 15
        repo = self.make_repository('repo')
        self.assertNotEqual(None, repo._reconcile_does_inventory_gc)

    def test_reconcile_empty_thorough(self):
        if False:
            for i in range(10):
                print('nop')
        self.checkEmptyReconcile(thorough=True)

    def test_convenience_reconcile_inventory_without_revision_reconcile(self):
        if False:
            print('Hello World!')
        bzrdir_url = self.get_url('inventory_without_revision')
        bzrdir = BzrDir.open(bzrdir_url)
        repo = bzrdir.open_repository()
        if not repo._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')
        reconcile(bzrdir)
        repo = bzrdir.open_repository()
        self.check_missing_was_removed(repo)

    def test_reweave_inventory_without_revision(self):
        if False:
            for i in range(10):
                print('nop')
        d_url = self.get_url('inventory_without_revision')
        d = BzrDir.open(d_url)
        repo = d.open_repository()
        if not repo._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')
        self.checkUnreconciled(d, repo.reconcile())
        reconciler = repo.reconcile(thorough=True)
        self.assertEqual(0, reconciler.inconsistent_parents)
        self.assertEqual(1, reconciler.garbage_inventories)
        self.check_missing_was_removed(repo)

    def check_thorough_reweave_missing_revision(self, aBzrDir, reconcile, **kwargs):
        if False:
            i = 10
            return i + 15
        repo = aBzrDir.open_repository()
        if not repo.has_revision('missing'):
            expected_inconsistent_parents = 0
        else:
            expected_inconsistent_parents = 1
        reconciler = reconcile(**kwargs)
        self.assertEqual(expected_inconsistent_parents, reconciler.inconsistent_parents)
        self.assertEqual(1, reconciler.garbage_inventories)
        repo = aBzrDir.open_repository()
        self.check_missing_was_removed(repo)
        self.assertFalse(repo.has_revision('missing'))

    def check_missing_was_removed(self, repo):
        if False:
            return 10
        if repo._reconcile_backsup_inventory:
            backed_up = False
            for path in repo.control_transport.list_dir('.'):
                if 'inventory.backup' in path:
                    backed_up = True
            self.assertTrue(backed_up)
        self.assertRaises(errors.NoSuchRevision, repo.get_inventory, 'missing')

    def test_reweave_inventory_without_revision_reconciler(self):
        if False:
            i = 10
            return i + 15
        d_url = self.get_url('inventory_without_revision_and_ghost')
        d = BzrDir.open(d_url)
        if not d.open_repository()._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')

        def reconcile():
            if False:
                print('Hello World!')
            reconciler = Reconciler(d)
            reconciler.reconcile()
            return reconciler
        self.check_thorough_reweave_missing_revision(d, reconcile)

    def test_reweave_inventory_without_revision_and_ghost(self):
        if False:
            while True:
                i = 10
        d_url = self.get_url('inventory_without_revision_and_ghost')
        d = BzrDir.open(d_url)
        repo = d.open_repository()
        if not repo._reconcile_does_inventory_gc:
            raise TestSkipped('Irrelevant test')
        self.check_thorough_reweave_missing_revision(d, repo.reconcile, thorough=True)

    def test_reweave_inventory_preserves_a_revision_with_ghosts(self):
        if False:
            while True:
                i = 10
        d = BzrDir.open(self.get_url('inventory_one_ghost'))
        reconciler = d.open_repository().reconcile(thorough=True)
        self.assertEqual(0, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        repo = d.open_repository()
        repo.get_inventory('ghost')
        self.assertThat(['ghost', 'the_ghost'], MatchesAncestry(repo, 'ghost'))

    def test_reweave_inventory_fixes_ancestryfor_a_present_ghost(self):
        if False:
            for i in range(10):
                print('nop')
        d = BzrDir.open(self.get_url('inventory_ghost_present'))
        repo = d.open_repository()
        m = MatchesAncestry(repo, 'ghost')
        if m.match(['the_ghost', 'ghost']) is None:
            return
        self.assertThat(['ghost'], m)
        reconciler = repo.reconcile()
        self.assertEqual(1, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        repo = d.open_repository()
        repo.get_inventory('ghost')
        repo.get_inventory('the_ghost')
        self.assertThat(['the_ghost', 'ghost'], MatchesAncestry(repo, 'ghost'))
        self.assertThat(['the_ghost'], MatchesAncestry(repo, 'the_ghost'))

    def test_text_from_ghost_revision(self):
        if False:
            for i in range(10):
                print('nop')
        repo = self.make_repository('text-from-ghost')
        inv = Inventory(revision_id='final-revid')
        inv.root.revision = 'root-revid'
        ie = inv.add_path('bla', 'file', 'myfileid')
        ie.revision = 'ghostrevid'
        ie.text_size = 42
        ie.text_sha1 = 'bee68c8acd989f5f1765b4660695275948bf5c00'
        rev = bzrlib.revision.Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', revision_id='final-revid')
        repo.lock_write()
        try:
            repo.start_write_group()
            try:
                repo.add_revision('final-revid', rev, inv)
                try:
                    repo.texts.add_lines(('myfileid', 'ghostrevid'), (('myfileid', 'ghost-text-parent'),), ['line1\n', 'line2\n'])
                except errors.RevisionNotPresent:
                    raise TestSkipped('text ghost parents not supported')
                if repo.supports_rich_root():
                    root_id = inv.root.file_id
                    repo.texts.add_lines((inv.root.file_id, inv.root.revision), [], [])
            finally:
                repo.commit_write_group()
        finally:
            repo.unlock()
        repo.reconcile(thorough=True)

class TestReconcileWithIncorrectRevisionCache(TestReconcile):
    """Ancestry data gets cached in knits and weaves should be reconcilable.

    This class tests that reconcile can correct invalid caches (such as after
    a reconcile).
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.reduceLockdirTimeout()
        super(TestReconcileWithIncorrectRevisionCache, self).setUp()
        t = self.get_transport()
        self.first_tree = self.make_branch_and_tree('wrong-first-parent')
        self.second_tree = self.make_branch_and_tree('reversed-secondary-parents')
        for t in [self.first_tree, self.second_tree]:
            t.commit('1', rev_id='1')
            uncommit(t.branch, tree=t)
            t.commit('2', rev_id='2')
            uncommit(t.branch, tree=t)
            t.commit('3', rev_id='3')
            uncommit(t.branch, tree=t)
        repo_secondary = self.second_tree.branch.repository
        repo = self.first_tree.branch.repository
        repo.lock_write()
        repo.start_write_group()
        inv = Inventory(revision_id='wrong-first-parent')
        inv.root.revision = 'wrong-first-parent'
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, 'wrong-first-parent'), [], [])
        sha1 = repo.add_inventory('wrong-first-parent', inv, ['2', '1'])
        rev = Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id='wrong-first-parent')
        rev.parent_ids = ['1', '2']
        repo.add_revision('wrong-first-parent', rev)
        repo.commit_write_group()
        repo.unlock()
        repo = repo_secondary
        repo.lock_write()
        repo.start_write_group()
        inv = Inventory(revision_id='wrong-secondary-parent')
        inv.root.revision = 'wrong-secondary-parent'
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, 'wrong-secondary-parent'), [], [])
        sha1 = repo.add_inventory('wrong-secondary-parent', inv, ['1', '3', '2'])
        rev = Revision(timestamp=0, timezone=None, committer='Foo Bar <foo@example.com>', message='Message', inventory_sha1=sha1, revision_id='wrong-secondary-parent')
        rev.parent_ids = ['1', '2', '3']
        repo.add_revision('wrong-secondary-parent', rev)
        repo.commit_write_group()
        repo.unlock()

    def test_reconcile_wrong_order(self):
        if False:
            i = 10
            return i + 15
        repo = self.first_tree.branch.repository
        repo.lock_read()
        try:
            g = repo.get_graph()
            if g.get_parent_map(['wrong-first-parent'])['wrong-first-parent'] == ('1', '2'):
                raise TestSkipped('wrong-first-parent is not setup for testing')
        finally:
            repo.unlock()
        self.checkUnreconciled(repo.bzrdir, repo.reconcile())
        reconciler = repo.reconcile(thorough=True)
        self.assertEqual(1, reconciler.inconsistent_parents)
        self.assertEqual(0, reconciler.garbage_inventories)
        repo.lock_read()
        self.addCleanup(repo.unlock)
        g = repo.get_graph()
        self.assertEqual({'wrong-first-parent': ('1', '2')}, g.get_parent_map(['wrong-first-parent']))

    def test_reconcile_wrong_order_secondary_inventory(self):
        if False:
            return 10
        repo = self.second_tree.branch.repository
        self.checkUnreconciled(repo.bzrdir, repo.reconcile())
        self.checkUnreconciled(repo.bzrdir, repo.reconcile(thorough=True))