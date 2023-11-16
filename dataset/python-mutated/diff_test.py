"""Tests for the diff and difffull modules."""
import tempfile
from fire import testutils
from examples.diff import diff
from examples.diff import difffull

class DiffTest(testutils.BaseTestCase):
    """The purpose of these tests is to ensure the difflib wrappers works.

  It is not the goal of these tests to exhaustively test difflib functionality.
  """

    def setUp(self):
        if False:
            return 10
        self.file1 = file1 = tempfile.NamedTemporaryFile()
        self.file2 = file2 = tempfile.NamedTemporaryFile()
        file1.write(b'test\ntest1\n')
        file2.write(b'test\ntest2\nextraline\n')
        file1.flush()
        file2.flush()
        self.diff = diff.DiffLibWrapper(file1.name, file2.name)

    def testSetUp(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.diff.fromlines, ['test\n', 'test1\n'])
        self.assertEqual(self.diff.tolines, ['test\n', 'test2\n', 'extraline\n'])

    def testUnifiedDiff(self):
        if False:
            i = 10
            return i + 15
        results = list(self.diff.unified_diff())
        self.assertTrue(results[0].startswith('--- ' + self.file1.name))
        self.assertTrue(results[1].startswith('+++ ' + self.file2.name))
        self.assertEqual(results[2:], ['@@ -1,2 +1,3 @@\n', ' test\n', '-test1\n', '+test2\n', '+extraline\n'])

    def testContextDiff(self):
        if False:
            return 10
        expected_lines = ['***************\n', '*** 1,2 ****\n', '  test\n', '! test1\n', '--- 1,3 ----\n', '  test\n', '! test2\n', '! extraline\n']
        results = list(self.diff.context_diff())
        self.assertEqual(results[2:], expected_lines)

    def testNDiff(self):
        if False:
            return 10
        expected_lines = ['  test\n', '- test1\n', '?     ^\n', '+ test2\n', '?     ^\n', '+ extraline\n']
        results = list(self.diff.ndiff())
        self.assertEqual(results, expected_lines)

    def testMakeDiff(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(''.join(self.diff.make_file()).startswith('\n<!DOC'))

    def testDiffFull(self):
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(difffull)
        self.assertIsNotNone(difffull.difflib)
if __name__ == '__main__':
    testutils.main()