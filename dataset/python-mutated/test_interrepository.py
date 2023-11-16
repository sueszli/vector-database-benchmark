"""Tests for InterRepository implementastions."""
import sys
import bzrlib
import bzrlib.errors as errors
import bzrlib.gpg
from bzrlib.inventory import Inventory
from bzrlib.revision import NULL_REVISION
from bzrlib.tests import TestNotApplicable, TestSkipped
from bzrlib.tests.matchers import MatchesAncestry
from bzrlib.tests.per_interrepository import TestCaseWithInterRepository

def check_repo_format_for_funky_id_on_win32(repo):
    if False:
        print('Hello World!')
    if not repo._format.supports_funky_characters and sys.platform == 'win32':
        raise TestSkipped('funky chars not allowed on this platform in repository %s' % repo.__class__.__name__)

class TestInterRepository(TestCaseWithInterRepository):

    def test_interrepository_get_returns_correct_optimiser(self):
        if False:
            return 10
        pass

class TestCaseWithComplexRepository(TestCaseWithInterRepository):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestCaseWithComplexRepository, self).setUp()
        tree_a = self.make_branch_and_tree('a')
        self.bzrdir = tree_a.branch.bzrdir
        tree_a.branch.repository.lock_write()
        tree_a.branch.repository.start_write_group()
        inv_file = tree_a.branch.repository.inventories
        inv_file.add_lines(('orphan',), [], [])
        tree_a.branch.repository.commit_write_group()
        tree_a.branch.repository.unlock()
        tree_a.commit('rev1', rev_id='rev1', allow_pointless=True)
        tree_a.commit('rev2', rev_id='rev2', allow_pointless=True)
        tree_a.branch.repository.lock_write()
        tree_a.branch.repository.start_write_group()
        tree_a.branch.repository.sign_revision('rev2', bzrlib.gpg.LoopbackGPGStrategy(None))
        tree_a.branch.repository.commit_write_group()
        tree_a.branch.repository.unlock()

    def test_search_missing_revision_ids(self):
        if False:
            return 10
        repo_b = self.make_to_repository('rev1_only')
        repo_a = self.bzrdir.open_repository()
        repo_b.fetch(repo_a, 'rev1')
        self.assertFalse(repo_b.has_revision('rev2'))
        result = repo_b.search_missing_revision_ids(repo_a)
        self.assertEqual(set(['rev2']), result.get_keys())
        self.assertEqual(('search', set(['rev2']), set(['rev1']), 1), result.get_recipe())

    def test_search_missing_revision_ids_absent_requested_raises(self):
        if False:
            print('Hello World!')
        repo_b = self.make_to_repository('target')
        repo_a = self.bzrdir.open_repository()
        self.assertFalse(repo_a.has_revision('pizza'))
        self.assertFalse(repo_b.has_revision('pizza'))
        self.assertRaises(errors.NoSuchRevision, repo_b.search_missing_revision_ids, repo_a, revision_ids=['pizza'], find_ghosts=True)
        self.assertRaises(errors.NoSuchRevision, repo_b.search_missing_revision_ids, repo_a, revision_ids=['pizza'], find_ghosts=False)
        self.callDeprecated(['search_missing_revision_ids(revision_id=...) was deprecated in 2.4.  Use revision_ids=[...] instead.'], self.assertRaises, errors.NoSuchRevision, repo_b.search_missing_revision_ids, repo_a, revision_id='pizza', find_ghosts=False)

    def test_search_missing_revision_ids_revision_limited(self):
        if False:
            for i in range(10):
                print('nop')
        repo_b = self.make_to_repository('empty')
        repo_a = self.bzrdir.open_repository()
        result = repo_b.search_missing_revision_ids(repo_a, revision_ids=['rev1'])
        self.assertEqual(set(['rev1']), result.get_keys())
        self.assertEqual(('search', set(['rev1']), set([NULL_REVISION]), 1), result.get_recipe())

    def test_search_missing_revision_ids_limit(self):
        if False:
            print('Hello World!')
        repo_b = self.make_to_repository('rev1_only')
        repo_a = self.bzrdir.open_repository()
        self.assertFalse(repo_b.has_revision('rev2'))
        result = repo_b.search_missing_revision_ids(repo_a, limit=1)
        self.assertEqual(('search', set(['rev1']), set(['null:']), 1), result.get_recipe())

    def test_fetch_fetches_signatures_too(self):
        if False:
            i = 10
            return i + 15
        from_repo = self.bzrdir.open_repository()
        from_signature = from_repo.get_signature_text('rev2')
        to_repo = self.make_to_repository('target')
        to_repo.fetch(from_repo)
        to_signature = to_repo.get_signature_text('rev2')
        self.assertEqual(from_signature, to_signature)

class TestCaseWithGhosts(TestCaseWithInterRepository):

    def test_fetch_all_fixes_up_ghost(self):
        if False:
            print('Hello World!')
        has_ghost = self.make_repository('has_ghost')
        missing_ghost = self.make_repository('missing_ghost')
        if [True, True] != [repo._format.supports_ghosts for repo in (has_ghost, missing_ghost)]:
            raise TestNotApplicable('Need ghost support.')

        def add_commit(repo, revision_id, parent_ids):
            if False:
                for i in range(10):
                    print('nop')
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
        add_commit(has_ghost, 'ghost', [])
        add_commit(has_ghost, 'references', ['ghost'])
        add_commit(missing_ghost, 'references', ['ghost'])
        add_commit(has_ghost, 'tip', ['references'])
        missing_ghost.fetch(has_ghost, 'tip', find_ghosts=True)
        rev = missing_ghost.get_revision('tip')
        inv = missing_ghost.get_inventory('tip')
        rev = missing_ghost.get_revision('ghost')
        inv = missing_ghost.get_inventory('ghost')
        self.assertThat(['ghost', 'references', 'tip'], MatchesAncestry(missing_ghost, 'tip'))