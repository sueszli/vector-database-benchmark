"""Tests for eol conversion."""
import sys
from cStringIO import StringIO
from bzrlib import rules, status
from bzrlib.tests import TestSkipped
from bzrlib.tests.per_workingtree import TestCaseWithWorkingTree
from bzrlib.workingtree import WorkingTree
_sample_text = 'hello\nworld\r\n'
_sample_text_on_win = 'hello\r\nworld\r\n'
_sample_text_on_unix = 'hello\nworld\n'
_sample_binary = 'hello\nworld\r\n\x00'
_sample_clean_lf = _sample_text_on_unix
_sample_clean_crlf = _sample_text_on_win
_LF_IN_REPO = ['native', 'lf', 'crlf']
_CRLF_IN_REPO = ['%s-with-crlf-in-repo' % (f,) for f in _LF_IN_REPO]

class TestEolConversion(TestCaseWithWorkingTree):

    def setUp(self):
        if False:
            print('Hello World!')
        fmt = self.workingtree_format
        f = getattr(fmt, 'supports_content_filtering')
        if f is None:
            raise TestSkipped("format %s doesn't declare whether it supports content filtering, assuming not" % fmt)
        if not f():
            raise TestSkipped("format %s doesn't support content filtering" % fmt)
        super(TestEolConversion, self).setUp()

    def patch_rules_searcher(self, eol):
        if False:
            for i in range(10):
                print('nop')
        'Patch in a custom rules searcher with a given eol setting.'
        if eol is None:
            WorkingTree._get_rules_searcher = self.real_rules_searcher
        else:

            def custom_eol_rules_searcher(tree, default_searcher):
                if False:
                    for i in range(10):
                        print('nop')
                return rules._IniBasedRulesSearcher(['[name *]\n', 'eol=%s\n' % eol])
            WorkingTree._get_rules_searcher = custom_eol_rules_searcher

    def prepare_tree(self, content, eol=None):
        if False:
            print('Hello World!')
        'Prepare a working tree and commit some content.'
        self.real_rules_searcher = self.overrideAttr(WorkingTree, '_get_rules_searcher')
        self.patch_rules_searcher(eol)
        t = self.make_branch_and_tree('tree1')
        self.build_tree_contents([('tree1/file1', content)])
        t.add('file1', 'file1-id')
        t.commit('add file1')
        basis = t.basis_tree()
        basis.lock_read()
        self.addCleanup(basis.unlock)
        return (t, basis)

    def assertNewContentForSetting(self, wt, eol, expected_unix, expected_win, roundtrip):
        if False:
            for i in range(10):
                print('nop')
        'Clone a working tree and check the convenience content.\n        \n        If roundtrip is True, status and commit should see no changes.\n        '
        if expected_win is None:
            expected_win = expected_unix
        self.patch_rules_searcher(eol)
        wt2 = wt.bzrdir.sprout('tree-%s' % eol).open_workingtree()
        content = wt2.get_file('file1-id', filtered=False).read()
        if sys.platform == 'win32':
            self.assertEqual(expected_win, content)
        else:
            self.assertEqual(expected_unix, content)
        if roundtrip:
            status_io = StringIO()
            status.show_tree_status(wt2, to_file=status_io)
            self.assertEqual('', status_io.getvalue())

    def assertContent(self, wt, basis, expected_raw, expected_unix, expected_win, roundtrip_to=None):
        if False:
            return 10
        'Check the committed content and content in cloned trees.\n        \n        :param roundtrip_to: the set of formats (excluding exact) we\n          can round-trip to or None for all\n        '
        basis_content = basis.get_file('file1-id').read()
        self.assertEqual(expected_raw, basis_content)
        self.assertNewContentForSetting(wt, None, expected_raw, expected_raw, roundtrip=True)
        self.assertNewContentForSetting(wt, 'exact', expected_raw, expected_raw, roundtrip=True)
        if roundtrip_to is None:
            roundtrip_to = _LF_IN_REPO + _CRLF_IN_REPO
        self.assertNewContentForSetting(wt, 'native', expected_unix, expected_win, 'native' in roundtrip_to)
        self.assertNewContentForSetting(wt, 'lf', expected_unix, expected_unix, 'lf' in roundtrip_to)
        self.assertNewContentForSetting(wt, 'crlf', expected_win, expected_win, 'crlf' in roundtrip_to)
        self.assertNewContentForSetting(wt, 'native-with-crlf-in-repo', expected_unix, expected_win, 'native-with-crlf-in-repo' in roundtrip_to)
        self.assertNewContentForSetting(wt, 'lf-with-crlf-in-repo', expected_unix, expected_unix, 'lf-with-crlf-in-repo' in roundtrip_to)
        self.assertNewContentForSetting(wt, 'crlf-with-crlf-in-repo', expected_win, expected_win, 'crlf-with-crlf-in-repo' in roundtrip_to)

    def test_eol_no_rules_binary(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_binary)
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_exact_binary(self):
        if False:
            i = 10
            return i + 15
        (wt, basis) = self.prepare_tree(_sample_binary, eol='exact')
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_native_binary(self):
        if False:
            for i in range(10):
                print('nop')
        (wt, basis) = self.prepare_tree(_sample_binary, eol='native')
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_lf_binary(self):
        if False:
            i = 10
            return i + 15
        (wt, basis) = self.prepare_tree(_sample_binary, eol='lf')
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_crlf_binary(self):
        if False:
            i = 10
            return i + 15
        (wt, basis) = self.prepare_tree(_sample_binary, eol='crlf')
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_native_with_crlf_in_repo_binary(self):
        if False:
            return 10
        (wt, basis) = self.prepare_tree(_sample_binary, eol='native-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_lf_with_crlf_in_repo_binary(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_binary, eol='lf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_crlf_with_crlf_in_repo_binary(self):
        if False:
            for i in range(10):
                print('nop')
        (wt, basis) = self.prepare_tree(_sample_binary, eol='crlf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)

    def test_eol_no_rules_dirty(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_text)
        self.assertContent(wt, basis, _sample_text, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_exact_dirty(self):
        if False:
            return 10
        (wt, basis) = self.prepare_tree(_sample_text, eol='exact')
        self.assertContent(wt, basis, _sample_text, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_native_dirty(self):
        if False:
            for i in range(10):
                print('nop')
        (wt, basis) = self.prepare_tree(_sample_text, eol='native')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_lf_dirty(self):
        if False:
            return 10
        (wt, basis) = self.prepare_tree(_sample_text, eol='lf')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_crlf_dirty(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_text, eol='crlf')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_native_with_crlf_in_repo_dirty(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_text, eol='native-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_lf_with_crlf_in_repo_dirty(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_text, eol='lf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_crlf_with_crlf_in_repo_dirty(self):
        if False:
            i = 10
            return i + 15
        (wt, basis) = self.prepare_tree(_sample_text, eol='crlf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=[])

    def test_eol_no_rules_clean_lf(self):
        if False:
            i = 10
            return i + 15
        (wt, basis) = self.prepare_tree(_sample_clean_lf)
        self.assertContent(wt, basis, _sample_clean_lf, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_no_rules_clean_crlf(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_clean_crlf)
        self.assertContent(wt, basis, _sample_clean_crlf, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)

    def test_eol_exact_clean_lf(self):
        if False:
            for i in range(10):
                print('nop')
        (wt, basis) = self.prepare_tree(_sample_clean_lf, eol='exact')
        self.assertContent(wt, basis, _sample_clean_lf, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_exact_clean_crlf(self):
        if False:
            i = 10
            return i + 15
        (wt, basis) = self.prepare_tree(_sample_clean_crlf, eol='exact')
        self.assertContent(wt, basis, _sample_clean_crlf, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)

    def test_eol_native_clean_lf(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_clean_lf, eol='native')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_native_clean_crlf(self):
        if False:
            print('Hello World!')
        (wt, basis) = self.prepare_tree(_sample_clean_crlf, eol='native')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_lf_clean_lf(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_clean_lf, eol='lf')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_lf_clean_crlf(self):
        if False:
            for i in range(10):
                print('nop')
        (wt, basis) = self.prepare_tree(_sample_clean_crlf, eol='lf')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_crlf_clean_lf(self):
        if False:
            return 10
        (wt, basis) = self.prepare_tree(_sample_clean_lf, eol='crlf')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_crlf_clean_crlf(self):
        if False:
            print('Hello World!')
        (wt, basis) = self.prepare_tree(_sample_clean_crlf, eol='crlf')
        self.assertContent(wt, basis, _sample_text_on_unix, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)

    def test_eol_native_with_crlf_in_repo_clean_lf(self):
        if False:
            return 10
        (wt, basis) = self.prepare_tree(_sample_clean_lf, eol='native-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)

    def test_eol_native_with_crlf_in_repo_clean_crlf(self):
        if False:
            for i in range(10):
                print('nop')
        (wt, basis) = self.prepare_tree(_sample_clean_crlf, eol='native-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)

    def test_eol_lf_with_crlf_in_repo_clean_lf(self):
        if False:
            for i in range(10):
                print('nop')
        (wt, basis) = self.prepare_tree(_sample_clean_lf, eol='lf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)

    def test_eol_lf_with_crlf_in_repo_clean_crlf(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_clean_crlf, eol='lf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)

    def test_eol_crlf_with_crlf_in_repo_clean_lf(self):
        if False:
            while True:
                i = 10
        (wt, basis) = self.prepare_tree(_sample_clean_lf, eol='crlf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)

    def test_eol_crlf_with_crlf_in_repo_clean_crlf(self):
        if False:
            return 10
        (wt, basis) = self.prepare_tree(_sample_clean_crlf, eol='crlf-with-crlf-in-repo')
        self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)