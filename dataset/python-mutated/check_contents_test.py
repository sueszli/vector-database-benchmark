from absl.testing import absltest
from xla.build_tools.lint import check_contents
from xla.build_tools.lint import diff_parser

class CheckDiffsTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        base_path = 'third_party/xla/build_tools/lint'
        with open(f'{base_path}/testdata/bad_cc.diff') as f:
            cls.bad_cc_hunks = diff_parser.parse_hunks(f.read())
        with open(f'{base_path}/testdata/important_cc.diff') as f:
            cls.important_cc_hunks = diff_parser.parse_hunks(f.read())

    def test_check_good_diff(self):
        if False:
            while True:
                i = 10
        locs = check_contents.check_diffs(self.bad_cc_hunks, prohibited_regex='Make_Unique', suppression_regex='OK')
        self.assertEmpty(locs, 0)

    def test_check_suppressed_diff_without_suppressions(self):
        if False:
            i = 10
            return i + 15
        locs = check_contents.check_diffs(self.bad_cc_hunks, prohibited_regex='Make_Unique')
        expected_locs = [check_contents.RegexLocation(path='src/dir/bad.cc', line_number=3, line_contents='using Make_Unique = std::make_unique; // OK', matched_text='Make_Unique'), check_contents.RegexLocation(path='src/dir/bad.cc', line_number=6, line_contents='  return Make_Unique<int>(a + b); // OK. Fixed now!', matched_text='Make_Unique')]
        self.assertEqual(locs, expected_locs)

    def test_check_suppressed_diff_with_path_regexes(self):
        if False:
            print('Hello World!')
        filtered_hunks = check_contents.filter_hunks_by_path(self.bad_cc_hunks, path_regexes=['src/important\\..*'], path_regex_exclusions=[])
        self.assertLen(filtered_hunks, 1)
        locs = check_contents.check_diffs(filtered_hunks, prohibited_regex='Make_Unique')
        self.assertEmpty(locs)

    def test_check_suppressed_diff_with_exclusions(self):
        if False:
            i = 10
            return i + 15
        filtered_hunks = check_contents.filter_hunks_by_path(self.bad_cc_hunks, path_regexes=[], path_regex_exclusions=['src/dir/.*'])
        self.assertLen(filtered_hunks, 1)
        locs = check_contents.check_diffs(filtered_hunks, prohibited_regex='Make_Unique')
        self.assertEmpty(locs)

    def test_check_suppressed_diff_with_suppression(self):
        if False:
            while True:
                i = 10
        filtered_hunks = check_contents.filter_hunks_by_path(self.bad_cc_hunks, path_regexes=[], path_regex_exclusions=[])
        self.assertEqual(self.bad_cc_hunks, filtered_hunks)
        locs = check_contents.check_diffs(filtered_hunks, prohibited_regex='Make_Unique', suppression_regex='OK')
        self.assertEmpty(locs)
if __name__ == '__main__':
    absltest.main()