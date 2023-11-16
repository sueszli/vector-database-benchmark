"""Tests variations of case-insensitive and case-preserving file-systems."""
import os
from bzrlib import osutils, tests
from bzrlib.tests import KnownFailure
from bzrlib.osutils import canonical_relpath, pathjoin
from bzrlib.tests.script import run_script
from bzrlib.tests.features import CaseInsCasePresFilenameFeature

class TestCICPBase(tests.TestCaseWithTransport):
    """Base class for tests on a case-insensitive, case-preserving filesystem.
    """
    _test_needs_features = [CaseInsCasePresFilenameFeature]

    def _make_mixed_case_tree(self):
        if False:
            print('Hello World!')
        'Make a working tree with mixed-case filenames.'
        wt = self.make_branch_and_tree('.')
        self.build_tree(['CamelCaseParent/', 'lowercaseparent/'])
        self.build_tree_contents([('CamelCaseParent/CamelCase', 'camel case'), ('lowercaseparent/lowercase', 'lower case'), ('lowercaseparent/mixedCase', 'mixedCasecase')])
        return wt

class TestAdd(TestCICPBase):

    def test_add_simple(self):
        if False:
            print('Hello World!')
        'Test add always uses the case of the filename reported by the os.'
        wt = self.make_branch_and_tree('.')
        self.build_tree(['CamelCase'])
        run_script(self, '\n            $ bzr add camelcase\n            adding CamelCase\n            ')

    def test_add_subdir(self):
        if False:
            while True:
                i = 10
        'test_add_simple but with subdirectories tested too.'
        wt = self.make_branch_and_tree('.')
        self.build_tree(['CamelCaseParent/', 'CamelCaseParent/CamelCase'])
        run_script(self, '\n            $ bzr add camelcaseparent/camelcase\n            adding CamelCaseParent\n            adding CamelCaseParent/CamelCase\n            ')

    def test_add_implied(self):
        if False:
            i = 10
            return i + 15
        'test add with no args sees the correct names.'
        wt = self.make_branch_and_tree('.')
        self.build_tree(['CamelCaseParent/', 'CamelCaseParent/CamelCase'])
        run_script(self, '\n            $ bzr add\n            adding CamelCaseParent\n            adding CamelCaseParent/CamelCase\n            ')

    def test_re_add(self):
        if False:
            for i in range(10):
                print('nop')
        "Test than when a file has 'unintentionally' changed case, we can't\n        add a new entry using the new case."
        wt = self.make_branch_and_tree('.')
        self.build_tree(['MixedCase'])
        run_script(self, '\n            $ bzr add MixedCase\n            adding MixedCase\n            ')
        osutils.rename('MixedCase', 'mixedcase')
        run_script(self, '\n            $ bzr add mixedcase\n            ')

    def test_re_add_dir(self):
        if False:
            return 10
        "Test than when a file has 'unintentionally' changed case, we can't\n        add a new entry using the new case."
        wt = self.make_branch_and_tree('.')
        self.build_tree(['MixedCaseParent/', 'MixedCaseParent/MixedCase'])
        run_script(self, '\n            $ bzr add MixedCaseParent\n            adding MixedCaseParent\n            adding MixedCaseParent/MixedCase\n            ')
        osutils.rename('MixedCaseParent', 'mixedcaseparent')
        run_script(self, '\n            $ bzr add mixedcaseparent\n            ')

    def test_add_not_found(self):
        if False:
            i = 10
            return i + 15
        "Test add when the input file doesn't exist."
        wt = self.make_branch_and_tree('.')
        self.build_tree(['MixedCaseParent/', 'MixedCaseParent/MixedCase'])
        expected_fname = pathjoin(wt.basedir, 'MixedCaseParent', 'notfound')
        run_script(self, '\n            $ bzr add mixedcaseparent/notfound\n            2>bzr: ERROR: No such file: %s\n            ' % (repr(expected_fname),))

class TestMove(TestCICPBase):

    def test_mv_newname(self):
        if False:
            print('Hello World!')
        wt = self._make_mixed_case_tree()
        run_script(self, '\n            $ bzr add -q\n            $ bzr ci -qm message\n            $ bzr mv camelcaseparent/camelcase camelcaseparent/NewCamelCase\n            CamelCaseParent/CamelCase => CamelCaseParent/NewCamelCase\n            ')

    def test_mv_newname_after(self):
        if False:
            i = 10
            return i + 15
        wt = self._make_mixed_case_tree()
        run_script(self, '\n            $ bzr add -q\n            $ bzr ci -qm message\n            $ mv CamelCaseParent/CamelCase CamelCaseParent/NewCamelCase\n            $ bzr mv --after camelcaseparent/camelcase camelcaseparent/newcamelcase\n            CamelCaseParent/CamelCase => CamelCaseParent/NewCamelCase\n            ')

    def test_mv_newname_exists(self):
        if False:
            return 10
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        run_script(self, '\n            $ bzr mv camelcaseparent/camelcase LOWERCASEPARENT/LOWERCASE\n            2>bzr: ERROR: Could not move CamelCase => lowercase: lowercaseparent/lowercase is already versioned.\n            ')

    def test_mv_newname_exists_after(self):
        if False:
            return 10
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        os.unlink('CamelCaseParent/CamelCase')
        osutils.rename('lowercaseparent/lowercase', 'lowercaseparent/LOWERCASE')
        run_script(self, '\n            $ bzr mv --after camelcaseparent/camelcase LOWERCASEPARENT/LOWERCASE\n            2>bzr: ERROR: Could not move CamelCase => lowercase: lowercaseparent/lowercase is already versioned.\n            ')

    def test_mv_newname_root(self):
        if False:
            i = 10
            return i + 15
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        run_script(self, '\n            $ bzr mv camelcaseparent NewCamelCaseParent\n            CamelCaseParent => NewCamelCaseParent\n            ')

    def test_mv_newname_root_after(self):
        if False:
            print('Hello World!')
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        run_script(self, '\n            $ mv CamelCaseParent NewCamelCaseParent\n            $ bzr mv --after camelcaseparent NewCamelCaseParent\n            CamelCaseParent => NewCamelCaseParent\n            ')

    def test_mv_newcase(self):
        if False:
            print('Hello World!')
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        run_script(self, '\n            $ bzr mv camelcaseparent/camelcase camelcaseparent/camelCase\n            CamelCaseParent/CamelCase => CamelCaseParent/camelCase\n            ')
        self.failUnlessEqual(canonical_relpath(wt.basedir, 'camelcaseparent/camelcase'), 'CamelCaseParent/camelCase')

    def test_mv_newcase_after(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        osutils.rename('CamelCaseParent/CamelCase', 'CamelCaseParent/camelCase')
        run_script(self, '\n            $ bzr mv --after camelcaseparent/camelcase camelcaseparent/camelCase\n            CamelCaseParent/CamelCase => CamelCaseParent/camelCase\n            ')
        self.failUnlessEqual(canonical_relpath(wt.basedir, 'camelcaseparent/camelcase'), 'CamelCaseParent/camelCase')

    def test_mv_multiple(self):
        if False:
            print('Hello World!')
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        run_script(self, '\n            $ bzr mv LOWercaseparent/LOWercase LOWercaseparent/MIXEDCase camelcaseparent\n            lowercaseparent/lowercase => CamelCaseParent/lowercase\n            lowercaseparent/mixedCase => CamelCaseParent/mixedCase\n            ')

class TestMisc(TestCICPBase):

    def test_status(self):
        if False:
            print('Hello World!')
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        run_script(self, '\n            $ bzr status camelcaseparent/camelcase LOWERCASEPARENT/LOWERCASE\n            added:\n              CamelCaseParent/\n              CamelCaseParent/CamelCase\n              lowercaseparent/\n              lowercaseparent/lowercase\n            ')

    def test_ci(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        got = self.run_bzr('ci -m message camelcaseparent LOWERCASEPARENT')[1]
        for expected in ['CamelCaseParent', 'lowercaseparent', 'CamelCaseParent/CamelCase', 'lowercaseparent/lowercase']:
            self.assertContainsRe(got, 'added ' + expected + '\n')

    def test_rm(self):
        if False:
            for i in range(10):
                print('nop')
        wt = self._make_mixed_case_tree()
        self.run_bzr('add')
        self.run_bzr('ci -m message')
        got = self.run_bzr('rm camelcaseparent LOWERCASEPARENT')[1]
        for expected in ['lowercaseparent/lowercase', 'CamelCaseParent/CamelCase']:
            self.assertContainsRe(got, 'deleted ' + expected + '\n')