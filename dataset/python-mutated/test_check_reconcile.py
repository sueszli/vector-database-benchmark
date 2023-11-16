"""Tests that use BrokenRepoScenario objects.

That is, tests for reconcile and check.
"""
from bzrlib import osutils
from bzrlib.inventory import Inventory, InventoryFile
from bzrlib.revision import NULL_REVISION, Revision
from bzrlib.tests import TestNotApplicable, multiply_scenarios
from bzrlib.tests.per_repository_vf import TestCaseWithRepository, all_repository_vf_format_scenarios
from bzrlib.tests.scenarios import load_tests_apply_scenarios
load_tests = load_tests_apply_scenarios

class BrokenRepoScenario(object):
    """Base class for defining scenarios for testing check and reconcile.

    A subclass needs to define the following methods:
        :populate_repository: a method to use to populate a repository with
            sample revisions, inventories and file versions.
        :all_versions_after_reconcile: all the versions in repository after
            reconcile.  run_test verifies that the text of each of these
            versions of the file is unchanged by the reconcile.
        :populated_parents: a list of (parents list, revision).  Each version
            of the file is verified to have the given parents before running
            the reconcile.  i.e. this is used to assert that the repo from the
            factory is what we expect.
        :corrected_parents: a list of (parents list, revision).  Each version
            of the file is verified to have the given parents after the
            reconcile.  i.e. this is used to assert that reconcile made the
            changes we expect it to make.

    A subclass may define the following optional method as well:
        :corrected_fulltexts: a list of file versions that should be stored as
            fulltexts (not deltas) after reconcile.  run_test will verify that
            this occurs.
    """

    def __init__(self, test_case):
        if False:
            i = 10
            return i + 15
        self.test_case = test_case

    def make_one_file_inventory(self, repo, revision, parents, inv_revision=None, root_revision=None, file_contents=None, make_file_version=True):
        if False:
            print('Hello World!')
        return self.test_case.make_one_file_inventory(repo, revision, parents, inv_revision=inv_revision, root_revision=root_revision, file_contents=file_contents, make_file_version=make_file_version)

    def add_revision(self, repo, revision_id, inv, parent_ids):
        if False:
            for i in range(10):
                print('nop')
        return self.test_case.add_revision(repo, revision_id, inv, parent_ids)

    def corrected_fulltexts(self):
        if False:
            for i in range(10):
                print('nop')
        return []

    def repository_text_key_index(self):
        if False:
            print('Hello World!')
        result = {}
        if self.versioned_root:
            result.update(self.versioned_repository_text_keys())
        result.update(self.repository_text_keys())
        return result

class UndamagedRepositoryScenario(BrokenRepoScenario):
    """A scenario where the repository has no damage.

    It has a single revision, 'rev1a', with a single file.
    """

    def all_versions_after_reconcile(self):
        if False:
            while True:
                i = 10
        return ('rev1a',)

    def populated_parents(self):
        if False:
            return 10
        return (((), 'rev1a'),)

    def corrected_parents(self):
        if False:
            print('Hello World!')
        return self.populated_parents()

    def check_regexes(self, repo):
        if False:
            while True:
                i = 10
        return ['0 unreferenced text versions']

    def populate_repository(self, repo):
        if False:
            for i in range(10):
                print('nop')
        inv = self.make_one_file_inventory(repo, 'rev1a', [], root_revision='rev1a')
        self.add_revision(repo, 'rev1a', inv, [])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            i = 10
            return i + 15
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'rev1a'): True})
        result.update({('a-file-id', 'rev1a'): True})
        return result

    def repository_text_keys(self):
        if False:
            i = 10
            return i + 15
        return {('a-file-id', 'rev1a'): [NULL_REVISION]}

    def versioned_repository_text_keys(self):
        if False:
            print('Hello World!')
        return {('TREE_ROOT', 'rev1a'): [NULL_REVISION]}

class FileParentIsNotInRevisionAncestryScenario(BrokenRepoScenario):
    """A scenario where a revision 'rev2' has 'a-file' with a
    parent 'rev1b' that is not in the revision ancestry.

    Reconcile should remove 'rev1b' from the parents list of 'a-file' in
    'rev2', preserving 'rev1a' as a parent.
    """

    def all_versions_after_reconcile(self):
        if False:
            i = 10
            return i + 15
        return ('rev1a', 'rev2')

    def populated_parents(self):
        if False:
            while True:
                i = 10
        return (((), 'rev1a'), ((), 'rev1b'), (('rev1a', 'rev1b'), 'rev2'))

    def corrected_parents(self):
        if False:
            for i in range(10):
                print('nop')
        return (((), 'rev1a'), (None, 'rev1b'), (('rev1a',), 'rev2'))

    def check_regexes(self, repo):
        if False:
            i = 10
            return i + 15
        return ["\\* a-file-id version rev2 has parents \\('rev1a', 'rev1b'\\) but should have \\('rev1a',\\)", '1 unreferenced text versions']

    def populate_repository(self, repo):
        if False:
            while True:
                i = 10
        inv = self.make_one_file_inventory(repo, 'rev1a', [], root_revision='rev1a')
        self.add_revision(repo, 'rev1a', inv, [])
        inv = self.make_one_file_inventory(repo, 'rev1b', [], root_revision='rev1b')
        repo.add_inventory('rev1b', inv, [])
        inv = self.make_one_file_inventory(repo, 'rev2', ['rev1a', 'rev1b'])
        self.add_revision(repo, 'rev2', inv, ['rev1a'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            i = 10
            return i + 15
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'rev1a'): True, ('TREE_ROOT', 'rev2'): True})
        result.update({('a-file-id', 'rev1a'): True, ('a-file-id', 'rev2'): True})
        return result

    def repository_text_keys(self):
        if False:
            i = 10
            return i + 15
        return {('a-file-id', 'rev1a'): [NULL_REVISION], ('a-file-id', 'rev2'): [('a-file-id', 'rev1a')]}

    def versioned_repository_text_keys(self):
        if False:
            while True:
                i = 10
        return {('TREE_ROOT', 'rev1a'): [NULL_REVISION], ('TREE_ROOT', 'rev2'): [('TREE_ROOT', 'rev1a')]}

class FileParentHasInaccessibleInventoryScenario(BrokenRepoScenario):
    """A scenario where a revision 'rev3' containing 'a-file' modified in
    'rev3', and with a parent which is in the revision ancestory, but whose
    inventory cannot be accessed at all.

    Reconcile should remove the file version parent whose inventory is
    inaccessbile (i.e. remove 'rev1c' from the parents of a-file's rev3).
    """

    def all_versions_after_reconcile(self):
        if False:
            while True:
                i = 10
        return ('rev2', 'rev3')

    def populated_parents(self):
        if False:
            print('Hello World!')
        return (((), 'rev2'), (('rev1c',), 'rev3'))

    def corrected_parents(self):
        if False:
            print('Hello World!')
        return (((), 'rev2'), ((), 'rev3'))

    def check_regexes(self, repo):
        if False:
            i = 10
            return i + 15
        return ["\\* a-file-id version rev3 has parents \\('rev1c',\\) but should have \\(\\)"]

    def populate_repository(self, repo):
        if False:
            i = 10
            return i + 15
        inv = self.make_one_file_inventory(repo, 'rev2', [])
        self.add_revision(repo, 'rev2', inv, [])
        self.make_one_file_inventory(repo, 'rev1c', [])
        inv = self.make_one_file_inventory(repo, 'rev3', ['rev1c'])
        self.add_revision(repo, 'rev3', inv, ['rev1c', 'rev1a'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            return 10
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'rev2'): True, ('TREE_ROOT', 'rev3'): True})
        result.update({('a-file-id', 'rev2'): True, ('a-file-id', 'rev3'): True})
        return result

    def repository_text_keys(self):
        if False:
            return 10
        return {('a-file-id', 'rev2'): [NULL_REVISION], ('a-file-id', 'rev3'): [NULL_REVISION]}

    def versioned_repository_text_keys(self):
        if False:
            print('Hello World!')
        return {('TREE_ROOT', 'rev2'): [NULL_REVISION], ('TREE_ROOT', 'rev3'): [NULL_REVISION]}

class FileParentsNotReferencedByAnyInventoryScenario(BrokenRepoScenario):
    """A scenario where a repository with file 'a-file' which has extra
    per-file versions that are not referenced by any inventory (even though
    they have the same ID as actual revisions).  The inventory of 'rev2'
    references 'rev1a' of 'a-file', but there is a 'rev2' of 'some-file' stored
    and erroneously referenced by later per-file versions (revisions 'rev4' and
    'rev5').

    Reconcile should remove the file parents that are not referenced by any
    inventory.
    """

    def all_versions_after_reconcile(self):
        if False:
            print('Hello World!')
        return ('rev1a', 'rev2c', 'rev4', 'rev5')

    def populated_parents(self):
        if False:
            print('Hello World!')
        return [(('rev1a',), 'rev2'), (('rev1a',), 'rev2b'), (('rev2',), 'rev3'), (('rev2',), 'rev4'), (('rev2', 'rev2c'), 'rev5')]

    def corrected_parents(self):
        if False:
            while True:
                i = 10
        return ((None, 'rev2'), (None, 'rev2b'), (('rev1a',), 'rev3'), (('rev1a',), 'rev4'), (('rev2c',), 'rev5'))

    def check_regexes(self, repo):
        if False:
            i = 10
            return i + 15
        if repo.supports_rich_root():
            count = 9
        else:
            count = 3
        return ['unreferenced version: {rev2} in a-file-id', 'unreferenced version: {rev2b} in a-file-id', "a-file-id version rev3 has parents \\('rev2',\\) but should have \\('rev1a',\\)", "a-file-id version rev5 has parents \\('rev2', 'rev2c'\\) but should have \\('rev2c',\\)", "a-file-id version rev4 has parents \\('rev2',\\) but should have \\('rev1a',\\)", '%d inconsistent parents' % count]

    def populate_repository(self, repo):
        if False:
            while True:
                i = 10
        inv = self.make_one_file_inventory(repo, 'rev1a', [], root_revision='rev1a')
        self.add_revision(repo, 'rev1a', inv, [])
        self.make_one_file_inventory(repo, 'rev2', ['rev1a'], inv_revision='rev1a')
        self.add_revision(repo, 'rev2', inv, ['rev1a'])
        inv = self.make_one_file_inventory(repo, 'rev3', ['rev2'])
        self.add_revision(repo, 'rev3', inv, ['rev1c', 'rev1a'])
        inv = self.make_one_file_inventory(repo, 'rev2b', ['rev1a'], inv_revision='rev1a')
        self.add_revision(repo, 'rev2b', inv, ['rev1a'])
        inv = self.make_one_file_inventory(repo, 'rev4', ['rev2'])
        self.add_revision(repo, 'rev4', inv, ['rev2', 'rev2b'])
        inv = self.make_one_file_inventory(repo, 'rev2c', ['rev1a'])
        self.add_revision(repo, 'rev2c', inv, ['rev1a'])
        inv = self.make_one_file_inventory(repo, 'rev5', ['rev2', 'rev2c'])
        self.add_revision(repo, 'rev5', inv, ['rev2', 'rev2c'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            for i in range(10):
                print('nop')
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'rev1a'): True, ('TREE_ROOT', 'rev2'): True, ('TREE_ROOT', 'rev2b'): True, ('TREE_ROOT', 'rev2c'): True, ('TREE_ROOT', 'rev3'): True, ('TREE_ROOT', 'rev4'): True, ('TREE_ROOT', 'rev5'): True})
        result.update({('a-file-id', 'rev1a'): True, ('a-file-id', 'rev2c'): True, ('a-file-id', 'rev3'): True, ('a-file-id', 'rev4'): True, ('a-file-id', 'rev5'): True})
        return result

    def repository_text_keys(self):
        if False:
            for i in range(10):
                print('nop')
        return {('a-file-id', 'rev1a'): [NULL_REVISION], ('a-file-id', 'rev2c'): [('a-file-id', 'rev1a')], ('a-file-id', 'rev3'): [('a-file-id', 'rev1a')], ('a-file-id', 'rev4'): [('a-file-id', 'rev1a')], ('a-file-id', 'rev5'): [('a-file-id', 'rev2c')]}

    def versioned_repository_text_keys(self):
        if False:
            return 10
        return {('TREE_ROOT', 'rev1a'): [NULL_REVISION], ('TREE_ROOT', 'rev2'): [('TREE_ROOT', 'rev1a')], ('TREE_ROOT', 'rev2b'): [('TREE_ROOT', 'rev1a')], ('TREE_ROOT', 'rev2c'): [('TREE_ROOT', 'rev1a')], ('TREE_ROOT', 'rev3'): [('TREE_ROOT', 'rev1a')], ('TREE_ROOT', 'rev4'): [('TREE_ROOT', 'rev2'), ('TREE_ROOT', 'rev2b')], ('TREE_ROOT', 'rev5'): [('TREE_ROOT', 'rev2'), ('TREE_ROOT', 'rev2c')]}

class UnreferencedFileParentsFromNoOpMergeScenario(BrokenRepoScenario):
    """
    rev1a and rev1b with identical contents
    rev2 revision has parents of [rev1a, rev1b]
    There is a a-file:rev2 file version, not referenced by the inventory.
    """

    def all_versions_after_reconcile(self):
        if False:
            return 10
        return ('rev1a', 'rev1b', 'rev2', 'rev4')

    def populated_parents(self):
        if False:
            for i in range(10):
                print('nop')
        return (((), 'rev1a'), ((), 'rev1b'), (('rev1a', 'rev1b'), 'rev2'), (None, 'rev3'), (('rev2',), 'rev4'))

    def corrected_parents(self):
        if False:
            print('Hello World!')
        return (((), 'rev1a'), ((), 'rev1b'), ((), 'rev2'), (None, 'rev3'), (('rev2',), 'rev4'))

    def corrected_fulltexts(self):
        if False:
            for i in range(10):
                print('nop')
        return ['rev2']

    def check_regexes(self, repo):
        if False:
            return 10
        return []

    def populate_repository(self, repo):
        if False:
            while True:
                i = 10
        inv1a = self.make_one_file_inventory(repo, 'rev1a', [], root_revision='rev1a')
        self.add_revision(repo, 'rev1a', inv1a, [])
        file_contents = repo.texts.get_record_stream([('a-file-id', 'rev1a')], 'unordered', False).next().get_bytes_as('fulltext')
        inv = self.make_one_file_inventory(repo, 'rev1b', [], root_revision='rev1b', file_contents=file_contents)
        self.add_revision(repo, 'rev1b', inv, [])
        inv = self.make_one_file_inventory(repo, 'rev2', ['rev1a', 'rev1b'], inv_revision='rev1a', file_contents=file_contents)
        self.add_revision(repo, 'rev2', inv, ['rev1a', 'rev1b'])
        inv = self.make_one_file_inventory(repo, 'rev3', ['rev2'], inv_revision='rev2', file_contents=file_contents, make_file_version=False)
        self.add_revision(repo, 'rev3', inv, ['rev2'])
        inv = self.make_one_file_inventory(repo, 'rev4', ['rev2'])
        self.add_revision(repo, 'rev4', inv, ['rev3'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            i = 10
            return i + 15
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'rev1a'): True, ('TREE_ROOT', 'rev1b'): True, ('TREE_ROOT', 'rev2'): True, ('TREE_ROOT', 'rev3'): True, ('TREE_ROOT', 'rev4'): True})
        result.update({('a-file-id', 'rev1a'): True, ('a-file-id', 'rev1b'): True, ('a-file-id', 'rev2'): False, ('a-file-id', 'rev4'): True})
        return result

    def repository_text_keys(self):
        if False:
            return 10
        return {('a-file-id', 'rev1a'): [NULL_REVISION], ('a-file-id', 'rev1b'): [NULL_REVISION], ('a-file-id', 'rev2'): [NULL_REVISION], ('a-file-id', 'rev4'): [('a-file-id', 'rev2')]}

    def versioned_repository_text_keys(self):
        if False:
            print('Hello World!')
        return {('TREE_ROOT', 'rev1a'): [NULL_REVISION], ('TREE_ROOT', 'rev1b'): [NULL_REVISION], ('TREE_ROOT', 'rev2'): [('TREE_ROOT', 'rev1a'), ('TREE_ROOT', 'rev1b')], ('TREE_ROOT', 'rev3'): [('TREE_ROOT', 'rev2')], ('TREE_ROOT', 'rev4'): [('TREE_ROOT', 'rev3')]}

class TooManyParentsScenario(BrokenRepoScenario):
    """A scenario where 'broken-revision' of 'a-file' claims to have parents
    ['good-parent', 'bad-parent'].  However 'bad-parent' is in the ancestry of
    'good-parent', so the correct parent list for that file version are is just
    ['good-parent'].
    """

    def all_versions_after_reconcile(self):
        if False:
            while True:
                i = 10
        return ('bad-parent', 'good-parent', 'broken-revision')

    def populated_parents(self):
        if False:
            print('Hello World!')
        return (((), 'bad-parent'), (('bad-parent',), 'good-parent'), (('good-parent', 'bad-parent'), 'broken-revision'))

    def corrected_parents(self):
        if False:
            for i in range(10):
                print('nop')
        return (((), 'bad-parent'), (('bad-parent',), 'good-parent'), (('good-parent',), 'broken-revision'))

    def check_regexes(self, repo):
        if False:
            i = 10
            return i + 15
        if repo.supports_rich_root():
            count = 3
        else:
            count = 1
        return ('     %d inconsistent parents' % count, "      \\* a-file-id version broken-revision has parents \\('good-parent', 'bad-parent'\\) but should have \\('good-parent',\\)")

    def populate_repository(self, repo):
        if False:
            i = 10
            return i + 15
        inv = self.make_one_file_inventory(repo, 'bad-parent', (), root_revision='bad-parent')
        self.add_revision(repo, 'bad-parent', inv, ())
        inv = self.make_one_file_inventory(repo, 'good-parent', ('bad-parent',))
        self.add_revision(repo, 'good-parent', inv, ('bad-parent',))
        inv = self.make_one_file_inventory(repo, 'broken-revision', ('good-parent', 'bad-parent'))
        self.add_revision(repo, 'broken-revision', inv, ('good-parent',))
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            for i in range(10):
                print('nop')
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'bad-parent'): True, ('TREE_ROOT', 'broken-revision'): True, ('TREE_ROOT', 'good-parent'): True})
        result.update({('a-file-id', 'bad-parent'): True, ('a-file-id', 'broken-revision'): True, ('a-file-id', 'good-parent'): True})
        return result

    def repository_text_keys(self):
        if False:
            i = 10
            return i + 15
        return {('a-file-id', 'bad-parent'): [NULL_REVISION], ('a-file-id', 'broken-revision'): [('a-file-id', 'good-parent')], ('a-file-id', 'good-parent'): [('a-file-id', 'bad-parent')]}

    def versioned_repository_text_keys(self):
        if False:
            i = 10
            return i + 15
        return {('TREE_ROOT', 'bad-parent'): [NULL_REVISION], ('TREE_ROOT', 'broken-revision'): [('TREE_ROOT', 'good-parent')], ('TREE_ROOT', 'good-parent'): [('TREE_ROOT', 'bad-parent')]}

class ClaimedFileParentDidNotModifyFileScenario(BrokenRepoScenario):
    """A scenario where the file parent is the same as the revision parent, but
    should not be because that revision did not modify the file.

    Specifically, the parent revision of 'current' is
    'modified-something-else', which does not modify 'a-file', but the
    'current' version of 'a-file' erroneously claims that
    'modified-something-else' is the parent file version.
    """

    def all_versions_after_reconcile(self):
        if False:
            i = 10
            return i + 15
        return ('basis', 'current')

    def populated_parents(self):
        if False:
            for i in range(10):
                print('nop')
        return (((), 'basis'), (('basis',), 'modified-something-else'), (('modified-something-else',), 'current'))

    def corrected_parents(self):
        if False:
            return 10
        return (((), 'basis'), (None, 'modified-something-else'), (('basis',), 'current'))

    def check_regexes(self, repo):
        if False:
            print('Hello World!')
        if repo.supports_rich_root():
            count = 3
        else:
            count = 1
        return ('%d inconsistent parents' % count, "\\* a-file-id version current has parents \\('modified-something-else',\\) but should have \\('basis',\\)")

    def populate_repository(self, repo):
        if False:
            while True:
                i = 10
        inv = self.make_one_file_inventory(repo, 'basis', ())
        self.add_revision(repo, 'basis', inv, ())
        inv = self.make_one_file_inventory(repo, 'modified-something-else', ('basis',), inv_revision='basis')
        self.add_revision(repo, 'modified-something-else', inv, ('basis',))
        inv = self.make_one_file_inventory(repo, 'current', ('modified-something-else',))
        self.add_revision(repo, 'current', inv, ('modified-something-else',))
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            return 10
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'basis'): True, ('TREE_ROOT', 'current'): True, ('TREE_ROOT', 'modified-something-else'): True})
        result.update({('a-file-id', 'basis'): True, ('a-file-id', 'current'): True})
        return result

    def repository_text_keys(self):
        if False:
            return 10
        return {('a-file-id', 'basis'): [NULL_REVISION], ('a-file-id', 'current'): [('a-file-id', 'basis')]}

    def versioned_repository_text_keys(self):
        if False:
            while True:
                i = 10
        return {('TREE_ROOT', 'basis'): ['null:'], ('TREE_ROOT', 'current'): [('TREE_ROOT', 'modified-something-else')], ('TREE_ROOT', 'modified-something-else'): [('TREE_ROOT', 'basis')]}

class IncorrectlyOrderedParentsScenario(BrokenRepoScenario):
    """A scenario where the set parents of a version of a file are correct, but
    the order of those parents is incorrect.

    This defines a 'broken-revision-1-2' and a 'broken-revision-2-1' which both
    have their file version parents reversed compared to the revision parents,
    which is invalid.  (We use two revisions with opposite orderings of the
    same parents to make sure that accidentally relying on dictionary/set
    ordering cannot make the test pass; the assumption is that while dict/set
    iteration order is arbitrary, it is also consistent within a single test).
    """

    def all_versions_after_reconcile(self):
        if False:
            print('Hello World!')
        return ['parent-1', 'parent-2', 'broken-revision-1-2', 'broken-revision-2-1']

    def populated_parents(self):
        if False:
            while True:
                i = 10
        return (((), 'parent-1'), ((), 'parent-2'), (('parent-2', 'parent-1'), 'broken-revision-1-2'), (('parent-1', 'parent-2'), 'broken-revision-2-1'))

    def corrected_parents(self):
        if False:
            i = 10
            return i + 15
        return (((), 'parent-1'), ((), 'parent-2'), (('parent-1', 'parent-2'), 'broken-revision-1-2'), (('parent-2', 'parent-1'), 'broken-revision-2-1'))

    def check_regexes(self, repo):
        if False:
            return 10
        if repo.supports_rich_root():
            count = 4
        else:
            count = 2
        return ('%d inconsistent parents' % count, "\\* a-file-id version broken-revision-1-2 has parents \\('parent-2', 'parent-1'\\) but should have \\('parent-1', 'parent-2'\\)", "\\* a-file-id version broken-revision-2-1 has parents \\('parent-1', 'parent-2'\\) but should have \\('parent-2', 'parent-1'\\)")

    def populate_repository(self, repo):
        if False:
            print('Hello World!')
        inv = self.make_one_file_inventory(repo, 'parent-1', [])
        self.add_revision(repo, 'parent-1', inv, [])
        inv = self.make_one_file_inventory(repo, 'parent-2', [])
        self.add_revision(repo, 'parent-2', inv, [])
        inv = self.make_one_file_inventory(repo, 'broken-revision-1-2', ['parent-2', 'parent-1'])
        self.add_revision(repo, 'broken-revision-1-2', inv, ['parent-1', 'parent-2'])
        inv = self.make_one_file_inventory(repo, 'broken-revision-2-1', ['parent-1', 'parent-2'])
        self.add_revision(repo, 'broken-revision-2-1', inv, ['parent-2', 'parent-1'])
        self.versioned_root = repo.supports_rich_root()

    def repository_text_key_references(self):
        if False:
            i = 10
            return i + 15
        result = {}
        if self.versioned_root:
            result.update({('TREE_ROOT', 'broken-revision-1-2'): True, ('TREE_ROOT', 'broken-revision-2-1'): True, ('TREE_ROOT', 'parent-1'): True, ('TREE_ROOT', 'parent-2'): True})
        result.update({('a-file-id', 'broken-revision-1-2'): True, ('a-file-id', 'broken-revision-2-1'): True, ('a-file-id', 'parent-1'): True, ('a-file-id', 'parent-2'): True})
        return result

    def repository_text_keys(self):
        if False:
            for i in range(10):
                print('nop')
        return {('a-file-id', 'broken-revision-1-2'): [('a-file-id', 'parent-1'), ('a-file-id', 'parent-2')], ('a-file-id', 'broken-revision-2-1'): [('a-file-id', 'parent-2'), ('a-file-id', 'parent-1')], ('a-file-id', 'parent-1'): [NULL_REVISION], ('a-file-id', 'parent-2'): [NULL_REVISION]}

    def versioned_repository_text_keys(self):
        if False:
            for i in range(10):
                print('nop')
        return {('TREE_ROOT', 'broken-revision-1-2'): [('TREE_ROOT', 'parent-1'), ('TREE_ROOT', 'parent-2')], ('TREE_ROOT', 'broken-revision-2-1'): [('TREE_ROOT', 'parent-2'), ('TREE_ROOT', 'parent-1')], ('TREE_ROOT', 'parent-1'): [NULL_REVISION], ('TREE_ROOT', 'parent-2'): [NULL_REVISION]}
all_broken_scenario_classes = [UndamagedRepositoryScenario, FileParentIsNotInRevisionAncestryScenario, FileParentHasInaccessibleInventoryScenario, FileParentsNotReferencedByAnyInventoryScenario, TooManyParentsScenario, ClaimedFileParentDidNotModifyFileScenario, IncorrectlyOrderedParentsScenario, UnreferencedFileParentsFromNoOpMergeScenario]

def broken_scenarios_for_all_formats():
    if False:
        return 10
    format_scenarios = all_repository_vf_format_scenarios()
    broken_scenarios = [(s.__name__, {'scenario_class': s}) for s in all_broken_scenario_classes]
    return multiply_scenarios(format_scenarios, broken_scenarios)

class TestFileParentReconciliation(TestCaseWithRepository):
    """Tests for how reconcile corrects errors in parents of file versions."""
    scenarios = broken_scenarios_for_all_formats()

    def make_populated_repository(self, factory):
        if False:
            while True:
                i = 10
        'Create a new repository populated by the given factory.'
        repo = self.make_repository('broken-repo')
        repo.lock_write()
        try:
            repo.start_write_group()
            try:
                factory(repo)
                repo.commit_write_group()
                return repo
            except:
                repo.abort_write_group()
                raise
        finally:
            repo.unlock()

    def add_revision(self, repo, revision_id, inv, parent_ids):
        if False:
            return 10
        'Add a revision with a given inventory and parents to a repository.\n\n        :param repo: a repository.\n        :param revision_id: the revision ID for the new revision.\n        :param inv: an inventory (such as created by\n            `make_one_file_inventory`).\n        :param parent_ids: the parents for the new revision.\n        '
        inv.revision_id = revision_id
        inv.root.revision = revision_id
        if repo.supports_rich_root():
            root_id = inv.root.file_id
            repo.texts.add_lines((root_id, revision_id), [], [])
        repo.add_inventory(revision_id, inv, parent_ids)
        revision = Revision(revision_id, committer='jrandom@example.com', timestamp=0, inventory_sha1='', timezone=0, message='foo', parent_ids=parent_ids)
        repo.add_revision(revision_id, revision, inv)

    def make_one_file_inventory(self, repo, revision, parents, inv_revision=None, root_revision=None, file_contents=None, make_file_version=True):
        if False:
            i = 10
            return i + 15
        "Make an inventory containing a version of a file with ID 'a-file'.\n\n        The file's ID will be 'a-file', and its filename will be 'a file name',\n        stored at the tree root.\n\n        :param repo: a repository to add the new file version to.\n        :param revision: the revision ID of the new inventory.\n        :param parents: the parents for this revision of 'a-file'.\n        :param inv_revision: if not None, the revision ID to store in the\n            inventory entry.  Otherwise, this defaults to revision.\n        :param root_revision: if not None, the inventory's root.revision will\n            be set to this.\n        :param file_contents: if not None, the contents of this file version.\n            Otherwise a unique default (based on revision ID) will be\n            generated.\n        "
        inv = Inventory(revision_id=revision)
        if root_revision is not None:
            inv.root.revision = root_revision
        file_id = 'a-file-id'
        entry = InventoryFile(file_id, 'a file name', 'TREE_ROOT')
        if inv_revision is not None:
            entry.revision = inv_revision
        else:
            entry.revision = revision
        entry.text_size = 0
        if file_contents is None:
            file_contents = '%sline\n' % entry.revision
        entry.text_sha1 = osutils.sha_string(file_contents)
        inv.add(entry)
        if make_file_version:
            repo.texts.add_lines((file_id, revision), [(file_id, parent) for parent in parents], [file_contents])
        return inv

    def require_repo_suffers_text_parent_corruption(self, repo):
        if False:
            while True:
                i = 10
        if not repo._reconcile_fixes_text_parents:
            raise TestNotApplicable('Format does not support text parent reconciliation')

    def file_parents(self, repo, revision_id):
        if False:
            return 10
        key = ('a-file-id', revision_id)
        parent_map = repo.texts.get_parent_map([key])
        return tuple((parent[-1] for parent in parent_map[key]))

    def assertFileVersionAbsent(self, repo, revision_id):
        if False:
            return 10
        self.assertEqual({}, repo.texts.get_parent_map([('a-file-id', revision_id)]))

    def assertParentsMatch(self, expected_parents_for_versions, repo, when_description):
        if False:
            i = 10
            return i + 15
        for (expected_parents, version) in expected_parents_for_versions:
            if expected_parents is None:
                self.assertFileVersionAbsent(repo, version)
            else:
                found_parents = self.file_parents(repo, version)
                self.assertEqual(expected_parents, found_parents, '%s reconcile %s has parents %s, should have %s.' % (when_description, version, found_parents, expected_parents))

    def prepare_test_repository(self):
        if False:
            i = 10
            return i + 15
        'Prepare a repository to test with from the test scenario.\n\n        :return: A repository, and the scenario instance.\n        '
        scenario = self.scenario_class(self)
        repo = self.make_populated_repository(scenario.populate_repository)
        self.require_repo_suffers_text_parent_corruption(repo)
        return (repo, scenario)

    def shas_for_versions_of_file(self, repo, versions):
        if False:
            i = 10
            return i + 15
        "Get the SHA-1 hashes of the versions of 'a-file' in the repository.\n\n        :param repo: the repository to get the hashes from.\n        :param versions: a list of versions to get hashes for.\n\n        :returns: A dict of `{version: hash}`.\n        "
        keys = [('a-file-id', version) for version in versions]
        return repo.texts.get_sha1s(keys)

    def test_reconcile_behaviour(self):
        if False:
            while True:
                i = 10
        'Populate a repository and reconcile it, verifying the state before\n        and after.\n        '
        (repo, scenario) = self.prepare_test_repository()
        repo.lock_read()
        try:
            self.assertParentsMatch(scenario.populated_parents(), repo, 'before')
            vf_shas = self.shas_for_versions_of_file(repo, scenario.all_versions_after_reconcile())
        finally:
            repo.unlock()
        result = repo.reconcile(thorough=True)
        repo.lock_read()
        try:
            self.assertParentsMatch(scenario.corrected_parents(), repo, 'after')
            self.assertEqual(vf_shas, self.shas_for_versions_of_file(repo, scenario.all_versions_after_reconcile()))
            for file_version in scenario.corrected_fulltexts():
                key = ('a-file-id', file_version)
                self.assertEqual({key: ()}, repo.texts.get_parent_map([key]))
                self.assertIsInstance(repo.texts.get_record_stream([key], 'unordered', True).next().get_bytes_as('fulltext'), str)
        finally:
            repo.unlock()

    def test_check_behaviour(self):
        if False:
            i = 10
            return i + 15
        'Populate a repository and check it, and verify the output.'
        (repo, scenario) = self.prepare_test_repository()
        check_result = repo.check()
        check_result.report_results(verbose=True)
        log = self.get_log()
        for pattern in scenario.check_regexes(repo):
            self.assertContainsRe(log, pattern)

    def test_find_text_key_references(self):
        if False:
            print('Hello World!')
        'Test that find_text_key_references finds erroneous references.'
        (repo, scenario) = self.prepare_test_repository()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual(scenario.repository_text_key_references(), repo.find_text_key_references())

    def test__generate_text_key_index(self):
        if False:
            while True:
                i = 10
        'Test that the generated text key index has all entries.'
        (repo, scenario) = self.prepare_test_repository()
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual(scenario.repository_text_key_index(), repo._generate_text_key_index())