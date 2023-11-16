from io import StringIO
from antlr4.tree.ParseTreePattern import ParseTreePattern
from antlr4.tree.Tree import ParseTree

class ParseTreeMatch(object):
    __slots__ = ('tree', 'pattern', 'labels', 'mismatchedNode')

    def __init__(self, tree: ParseTree, pattern: ParseTreePattern, labels: dict, mismatchedNode: ParseTree):
        if False:
            while True:
                i = 10
        if tree is None:
            raise Exception('tree cannot be null')
        if pattern is None:
            raise Exception('pattern cannot be null')
        if labels is None:
            raise Exception('labels cannot be null')
        self.tree = tree
        self.pattern = pattern
        self.labels = labels
        self.mismatchedNode = mismatchedNode

    def get(self, label: str):
        if False:
            while True:
                i = 10
        parseTrees = self.labels.get(label, None)
        if parseTrees is None or len(parseTrees) == 0:
            return None
        else:
            return parseTrees[len(parseTrees) - 1]

    def getAll(self, label: str):
        if False:
            return 10
        nodes = self.labels.get(label, None)
        if nodes is None:
            return list()
        else:
            return nodes

    def succeeded(self):
        if False:
            print('Hello World!')
        return self.mismatchedNode is None

    def __str__(self):
        if False:
            while True:
                i = 10
        with StringIO() as buf:
            buf.write('Match ')
            buf.write('succeeded' if self.succeeded() else 'failed')
            buf.write('; found ')
            buf.write(str(len(self.labels)))
            buf.write(' labels')
            return buf.getvalue()