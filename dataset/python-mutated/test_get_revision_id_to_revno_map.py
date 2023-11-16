"""Tests for Branch.get_revision_id_to_revno_map()"""
from bzrlib.symbol_versioning import deprecated_in
from bzrlib.tests.per_branch import TestCaseWithBranch

class TestRevisionIdToDottedRevno(TestCaseWithBranch):

    def test_simple_revno(self):
        if False:
            while True:
                i = 10
        tree = self.create_tree_with_merge()
        the_branch = tree.bzrdir.open_branch()
        self.assertEqual({'rev-1': (1,), 'rev-2': (2,), 'rev-3': (3,), 'rev-1.1.1': (1, 1, 1)}, the_branch.get_revision_id_to_revno_map())

class TestCaching(TestCaseWithBranch):
    """Tests for the caching of branches' dotted revno generation.

    When locked, branches should avoid regenerating revision_id=>dotted revno
    mapping.

    When not locked, obviously the revision_id => dotted revno will need to be
    regenerated or reread each time.

    We test if revision_history is using the cache by instrumenting the branch's
    _gen_revno_map method, which is called by get_revision_id_to_revno_map.
    """

    def get_instrumented_branch(self):
        if False:
            print('Hello World!')
        'Get a branch and monkey patch it to log calls to _gen_revno_map.\n\n        :returns: a tuple of (the branch, list that calls will be logged to)\n        '
        tree = self.create_tree_with_merge()
        calls = []
        real_func = tree.branch._gen_revno_map

        def wrapper():
            if False:
                return 10
            calls.append('_gen_revno_map')
            return real_func()
        tree.branch._gen_revno_map = wrapper
        return (tree.branch, calls)

    def test_unlocked(self):
        if False:
            print('Hello World!')
        'Repeated calls will call _gen_revno_map each time.'
        (branch, calls) = self.get_instrumented_branch()
        branch.get_revision_id_to_revno_map()
        branch.get_revision_id_to_revno_map()
        branch.get_revision_id_to_revno_map()
        self.assertEqual(['_gen_revno_map'] * 3, calls)

    def test_locked(self):
        if False:
            for i in range(10):
                print('nop')
        'Repeated calls will only call _gen_revno_map once.\n        '
        (branch, calls) = self.get_instrumented_branch()
        branch.lock_read()
        try:
            branch.get_revision_id_to_revno_map()
            self.assertEqual(['_gen_revno_map'], calls)
        finally:
            branch.unlock()

    def test_set_last_revision_info_when_locked(self):
        if False:
            for i in range(10):
                print('nop')
        'Calling set_last_revision_info should reset the cache.'
        (branch, calls) = self.get_instrumented_branch()
        branch.lock_write()
        try:
            self.assertEqual({'rev-1': (1,), 'rev-2': (2,), 'rev-3': (3,), 'rev-1.1.1': (1, 1, 1)}, branch.get_revision_id_to_revno_map())
            branch.set_last_revision_info(2, 'rev-2')
            self.assertEqual({'rev-1': (1,), 'rev-2': (2,)}, branch.get_revision_id_to_revno_map())
            self.assertEqual({'rev-1': (1,), 'rev-2': (2,)}, branch.get_revision_id_to_revno_map())
            self.assertEqual(['_gen_revno_map'] * 2, calls)
        finally:
            branch.unlock()