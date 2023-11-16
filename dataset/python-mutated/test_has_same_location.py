"""Tests for implementations of Repository.has_same_location."""
from bzrlib import controldir, transport
from bzrlib.remote import RemoteRepositoryFormat
from bzrlib.tests import TestNotApplicable
from bzrlib.tests.per_repository import TestCaseWithRepository

class TestHasSameLocation(TestCaseWithRepository):
    """Tests for Repository.has_same_location method."""

    def assertSameRepo(self, a, b):
        if False:
            while True:
                i = 10
        "Asserts that two objects are the same repository.\n\n        This method does the comparison both ways (`a.has_same_location(b)` as\n        well as `b.has_same_location(a)`) to make sure both objects'\n        `has_same_location` methods give the same results.\n        "
        self.assertTrue(a.has_same_location(b), '%r is not the same repository as %r' % (a, b))
        self.assertTrue(b.has_same_location(a), '%r is the same as %r, but not vice versa' % (a, b))

    def assertDifferentRepo(self, a, b):
        if False:
            while True:
                i = 10
        "Asserts that two objects are the not same repository.\n\n        This method does the comparison both ways (`a.has_same_location(b)` as\n        well as `b.has_same_location(a)`) to make sure both objects'\n        `has_same_location` methods give the same results.\n\n        :seealso: assertDifferentRepo\n        "
        self.assertFalse(a.has_same_location(b), '%r is not the same repository as %r' % (a, b))
        self.assertFalse(b.has_same_location(a), '%r is the same as %r, but not vice versa' % (a, b))

    def test_same_repo_instance(self):
        if False:
            print('Hello World!')
        'A repository object is the same repository as itself.'
        repo = self.make_repository('.')
        self.assertSameRepo(repo, repo)

    def test_same_repo_location(self):
        if False:
            for i in range(10):
                print('nop')
        'Different repository objects for the same location are the same.'
        repo = self.make_repository('.')
        reopened_repo = repo.bzrdir.open_repository()
        self.assertFalse(repo is reopened_repo, 'This test depends on reopened_repo being a different instance of the same repo.')
        self.assertSameRepo(repo, reopened_repo)

    def test_different_repos_not_equal(self):
        if False:
            print('Hello World!')
        'Repositories at different locations are not the same.'
        repo_one = self.make_repository('one')
        repo_two = self.make_repository('two')
        self.assertDifferentRepo(repo_one, repo_two)

    def test_same_bzrdir_different_control_files_not_equal(self):
        if False:
            while True:
                i = 10
        'Repositories in the same bzrdir, but with different control files,\n        are not the same.\n\n        This can happens e.g. when upgrading a repository.  This test mimics how\n        CopyConverter creates a second repository in one bzrdir.\n        '
        repo = self.make_repository('repo')
        if repo.control_transport.base == repo.bzrdir.control_transport.base:
            raise TestNotApplicable('%r has repository files directly in the bzrdir' % (repo,))
        repo.control_transport.copy_tree('.', '../repository.backup')
        backup_transport = repo.control_transport.clone('../repository.backup')
        if isinstance(repo._format, RemoteRepositoryFormat):
            raise TestNotApplicable("remote repositories don't support overriding transport")
        backup_repo = repo._format.open(repo.bzrdir, _override_transport=backup_transport)
        self.assertDifferentRepo(repo, backup_repo)

    def test_different_format_not_equal(self):
        if False:
            i = 10
            return i + 15
        'Different format repositories are comparable and not the same.\n\n        Comparing different format repository objects should give a negative\n        result, rather than trigger an exception (which could happen with a\n        naive __eq__ implementation, e.g. due to missing attributes).\n        '
        repo = self.make_repository('repo')
        other_repo = self.make_repository('other', format='default')
        if repo._format == other_repo._format:
            transport.get_transport_from_url(self.get_vfs_only_url()).delete_tree('other')
            other_repo = self.make_repository('other', format='knit')
        other_bzrdir = controldir.ControlDir.open(self.get_vfs_only_url('other'))
        other_repo = other_bzrdir.open_repository()
        self.assertDifferentRepo(repo, other_repo)