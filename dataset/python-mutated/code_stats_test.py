import os
import sys
from io import StringIO
from keras.testing import test_case
from keras.utils.code_stats import count_loc

class TestCountLoc(test_case.TestCase):

    def setUp(self):
        if False:
            return 10
        self.test_dir = 'test_directory'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        if False:
            while True:
                i = 10
        for (root, dirs, files) in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    def create_file(self, filename, content):
        if False:
            print('Hello World!')
        with open(os.path.join(self.test_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)

    def test_count_loc_valid_python(self):
        if False:
            while True:
                i = 10
        self.create_file('sample.py', "# This is a test file\n\nprint('Hello')\n")
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_exclude_test_files(self):
        if False:
            print('Hello World!')
        self.create_file('sample_test.py', "print('Hello')\n")
        loc = count_loc(self.test_dir, exclude=('_test',))
        self.assertEqual(loc, 0)

    def test_other_extensions(self):
        if False:
            i = 10
            return i + 15
        self.create_file('sample.txt', 'Hello\n')
        loc = count_loc(self.test_dir, extensions=('.py',))
        self.assertEqual(loc, 0)

    def test_comment_lines(self):
        if False:
            print('Hello World!')
        self.create_file('sample.py', "# Comment\nprint('Hello')\n# Another comment\n")
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_empty_file(self):
        if False:
            i = 10
            return i + 15
        self.create_file('empty.py', '')
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 0)

    def test_whitespace_only(self):
        if False:
            return 10
        self.create_file('whitespace.py', '     \n\t\n')
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 0)

    def test_inline_comments_after_code(self):
        if False:
            i = 10
            return i + 15
        content = 'print("Hello") # This is an inline comment'
        self.create_file('inline_comment_sample.py', content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_directory_structure(self):
        if False:
            i = 10
            return i + 15
        content1 = 'print("Hello from file1")'
        content2 = 'print("Hello from file2")'
        os.mkdir(os.path.join(self.test_dir, 'subdir'))
        self.create_file('sample1.py', content1)
        self.create_file(os.path.join('subdir', 'sample2.py'), content2)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 2)

    def test_normal_directory_name(self):
        if False:
            i = 10
            return i + 15
        content = 'print("Hello from a regular directory")'
        os.makedirs(os.path.join(self.test_dir, 'some_test_dir'))
        self.create_file(os.path.join('some_test_dir', 'sample.py'), content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_exclude_directory_name(self):
        if False:
            print('Hello World!')
        content = 'print("Hello from an excluded directory")'
        os.makedirs(os.path.join(self.test_dir, 'dir_test'))
        self.create_file(os.path.join('dir_test', 'sample.py'), content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 0)

    def test_verbose_output(self):
        if False:
            return 10
        content = 'print("Hello")'
        self.create_file('sample.py', content)
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        count_loc(self.test_dir, verbose=1)
        output = sys.stdout.getvalue()
        sys.stdout = original_stdout
        self.assertIn('Count LoCs in', output)

    def test_multiline_string_same_line(self):
        if False:
            i = 10
            return i + 15
        content = '"""This is a multiline string ending on the same line"""\n        print("Outside string")'
        self.create_file('same_line_multiline.py', content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_multiline_string_ends_on_same_line(self):
        if False:
            i = 10
            return i + 15
        content = '"""a multiline string end on same line"""\nprint("Outstr")'
        self.create_file('same_line_multiline.py', content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 1)

    def test_multiline_string_ends_in_middle_of_line(self):
        if False:
            return 10
        content = 'print("Start")\n        """This is a multiline string ending in the middle of a line"""\n        """This is another multiline string."""\n        print("End")'
        self.create_file('multiline_in_middle.py', content)
        loc = count_loc(self.test_dir)
        self.assertEqual(loc, 2)

    def test_line_starting_with_triple_quotes_not_ending(self):
        if False:
            for i in range(10):
                print('nop')
        content = '"""\nThis is a multiline string\n'
        self.create_file('test_file_2.py', content)
        path = os.path.join(self.test_dir, 'test_file_2.py')
        self.assertEqual(count_loc(path), 0)

    def test_line_starting_and_ending_with_triple_quotes(self):
        if False:
            while True:
                i = 10
        content = '"""This is a one-liner docstring."""\n'
        self.create_file('test_file_3.py', content)
        path = os.path.join(self.test_dir, 'test_file_3.py')
        self.assertEqual(count_loc(path), 0)

    def test_string_open_true_line_starting_with_triple_quotes(self):
        if False:
            return 10
        content = '"""\nEnd of the multiline string."""\n'
        self.create_file('test_file_4.py', content)
        path = os.path.join(self.test_dir, 'test_file_4.py')
        self.assertEqual(count_loc(path), 0)