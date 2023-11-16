from bzrlib.textmerge import Merge2
from bzrlib.tests import TestCase

class TestMerge2(TestCase):

    def test_agreed(self):
        if False:
            i = 10
            return i + 15
        lines = 'a\nb\nc\nd\ne\nf\n'.splitlines(True)
        mlines = list(Merge2(lines, lines).merge_lines()[0])
        self.assertEqualDiff(mlines, lines)

    def test_conflict(self):
        if False:
            print('Hello World!')
        lines_a = 'a\nb\nc\nd\ne\nf\ng\nh\n'.splitlines(True)
        lines_b = 'z\nb\nx\nd\ne\ne\nf\ng\ny\n'.splitlines(True)
        expected = '<\na\n=\nz\n>\nb\n<\nc\n=\nx\n>\nd\ne\n<\n=\ne\n>\nf\ng\n<\nh\n=\ny\n>\n'
        m2 = Merge2(lines_a, lines_b, '<\n', '>\n', '=\n')
        mlines = m2.merge_lines()[0]
        self.assertEqualDiff(''.join(mlines), expected)
        mlines = m2.merge_lines(reprocess=True)[0]
        self.assertEqualDiff(''.join(mlines), expected)

    def test_reprocess(self):
        if False:
            while True:
                i = 10
        struct = [('a', 'b'), ('c',), ('def', 'geh'), ('i',)]
        expect = [('a', 'b'), ('c',), ('d', 'g'), ('e',), ('f', 'h'), ('i',)]
        result = Merge2.reprocess_struct(struct)
        self.assertEqual(list(result), expect)