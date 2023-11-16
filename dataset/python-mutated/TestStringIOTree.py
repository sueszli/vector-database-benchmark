import unittest
from Cython import StringIOTree as stringtree
code = "\ncdef int spam                   # line 1\n\ncdef ham():\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n\ndef eggs():\n    pass\n\ncpdef bacon():\n    print spam\n    print 'scotch'\n    print 'tea?'\n    print 'or coffee?'          # line 16\n"
linemap = dict(enumerate(code.splitlines()))

class TestStringIOTree(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.tree = stringtree.StringIOTree()

    def test_markers(self):
        if False:
            print('Hello World!')
        assert not self.tree.allmarkers()

    def test_insertion(self):
        if False:
            print('Hello World!')
        self.write_lines((1, 2, 3))
        line_4_to_6_insertion_point = self.tree.insertion_point()
        self.write_lines((7, 8))
        line_9_to_13_insertion_point = self.tree.insertion_point()
        self.write_lines((14, 15, 16))
        line_4_insertion_point = line_4_to_6_insertion_point.insertion_point()
        self.write_lines((5, 6), tree=line_4_to_6_insertion_point)
        line_9_to_12_insertion_point = line_9_to_13_insertion_point.insertion_point()
        self.write_line(13, tree=line_9_to_13_insertion_point)
        self.write_line(4, tree=line_4_insertion_point)
        self.write_line(9, tree=line_9_to_12_insertion_point)
        line_10_insertion_point = line_9_to_12_insertion_point.insertion_point()
        self.write_line(11, tree=line_9_to_12_insertion_point)
        self.write_line(10, tree=line_10_insertion_point)
        self.write_line(12, tree=line_9_to_12_insertion_point)
        self.assertEqual(self.tree.allmarkers(), list(range(1, 17)))
        self.assertEqual(code.strip(), self.tree.getvalue().strip())

    def write_lines(self, linenos, tree=None):
        if False:
            for i in range(10):
                print('nop')
        for lineno in linenos:
            self.write_line(lineno, tree=tree)

    def write_line(self, lineno, tree=None):
        if False:
            return 10
        if tree is None:
            tree = self.tree
        tree.markers.append(lineno)
        tree.write(linemap[lineno] + '\n')