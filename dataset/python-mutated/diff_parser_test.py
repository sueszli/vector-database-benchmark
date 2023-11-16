from absl.testing import absltest
from xla.build_tools.lint import diff_parser

class ParseDiffTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        base_path = 'third_party/xla/build_tools/lint'
        with open(f'{base_path}/testdata/bad_cc.diff') as f:
            cls.bad_cc_diff = f.read()
        with open(f'{base_path}/testdata/important_cc.diff') as f:
            cls.important_cc_diff = f.read()
        with open(f'{base_path}/testdata/crosstool.diff') as f:
            cls.crosstool_diff = f.read()

    def test_parse_important_cc_diff(self):
        if False:
            return 10
        hunks = diff_parser.parse_hunks(self.important_cc_diff)
        self.assertLen(hunks, 1)
        [hunk] = hunks
        self.assertEqual(hunk.file, 'src/important.cc')
        self.assertEqual(hunk.start, 1)
        self.assertEqual(hunk.length, 3)
        expected_lines = ['+// Here we care if we find prohibited regexes.', '+std::unique_ptr<int> add(int a, int b) {', '+  return std::make_unique<int>(a + b);', '+}']
        self.assertEqual(hunk.lines, expected_lines)

    def test_parse_bad_cc_diff(self):
        if False:
            print('Hello World!')
        hunks = diff_parser.parse_hunks(self.bad_cc_diff)
        self.assertLen(hunks, 2)
        (bad_cc_hunk, important_cc_hunk) = hunks
        self.assertEqual(bad_cc_hunk.file, 'src/dir/bad.cc')
        self.assertEqual(bad_cc_hunk.start, 1)
        self.assertEqual(bad_cc_hunk.length, 7)
        expected_lines = ['+// This code is bad!', '+', '+using Make_Unique = std::make_unique; // OK', '+', '+std::unique_ptr<int> add(int a, int b) {', '+  return Make_Unique<int>(a + b); // OK. Fixed now!', '+}']
        self.assertEqual(bad_cc_hunk.lines, expected_lines)
        self.assertEqual(important_cc_hunk.file, 'src/important.cc')
        self.assertEqual(important_cc_hunk.start, 1)
        self.assertEqual(important_cc_hunk.length, 5)
        expected_lines = ['+// Here we care if we find prohibited regexes.', '+', '+std::unique_ptr<int> add(int a, int b) {', '+  return std::make_unique<int>(a + b);', '+}']
        self.assertEqual(important_cc_hunk.lines, expected_lines)

    def test_parse_crosstool_diff(self):
        if False:
            print('Hello World!')
        hunks = diff_parser.parse_hunks(self.crosstool_diff)
        self.assertLen(hunks, 3)
        (small_hunk, big_hunk, literal_cc_hunk) = hunks
        self.assertEqual(small_hunk.file, 'third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl')
        self.assertEqual(small_hunk.start, 24)
        self.assertEqual(small_hunk.length, 7)
        self.assertEqual(big_hunk.file, 'third_party/gpus/crosstool/cc_toolchain_config.bzl.tpl')
        self.assertEqual(big_hunk.start, 300)
        self.assertEqual(big_hunk.length, 45)
        self.assertEqual(literal_cc_hunk.file, 'xla/literal.cc')
        self.assertEqual(literal_cc_hunk.start, 47)
        self.assertEqual(literal_cc_hunk.length, 7)

    def test_added_lines(self):
        if False:
            print('Hello World!')
        hunks = diff_parser.parse_hunks(self.crosstool_diff)
        (small_hunk, big_hunk, literal_cc_hunk) = hunks
        line_numbers = lambda hunk: [line_no for (line_no, _) in hunk.added_lines()]
        self.assertEqual(line_numbers(small_hunk), [27])
        self.assertEqual(line_numbers(big_hunk), list(range(303, 342)))
        self.assertEqual(line_numbers(literal_cc_hunk), [50])
if __name__ == '__main__':
    absltest.main()