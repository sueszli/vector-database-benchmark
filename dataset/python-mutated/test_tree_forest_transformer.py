from __future__ import absolute_import
import unittest
from lark import Lark
from lark.lexer import Token
from lark.tree import Tree
from lark.visitors import Visitor, Transformer, Discard
from lark.parsers.earley_forest import TreeForestTransformer, handles_ambiguity

class TestTreeForestTransformer(unittest.TestCase):
    grammar = '\n    start: ab bc cd\n    !ab: "A" "B"?\n    !bc: "B"? "C"?\n    !cd: "C"? "D"\n    '
    parser = Lark(grammar, parser='earley', ambiguity='forest')
    forest = parser.parse('ABCD')

    def test_identity_resolve_ambiguity(self):
        if False:
            for i in range(10):
                print('nop')
        l = Lark(self.grammar, parser='earley', ambiguity='resolve')
        tree1 = l.parse('ABCD')
        tree2 = TreeForestTransformer(resolve_ambiguity=True).transform(self.forest)
        self.assertEqual(tree1, tree2)

    def test_identity_explicit_ambiguity(self):
        if False:
            print('Hello World!')
        l = Lark(self.grammar, parser='earley', ambiguity='explicit')
        tree1 = l.parse('ABCD')
        tree2 = TreeForestTransformer(resolve_ambiguity=False).transform(self.forest)
        self.assertEqual(tree1, tree2)

    def test_tree_class(self):
        if False:
            for i in range(10):
                print('nop')

        class CustomTree(Tree):
            pass

        class TreeChecker(Visitor):

            def __default__(self, tree):
                if False:
                    print('Hello World!')
                assert isinstance(tree, CustomTree)
        tree = TreeForestTransformer(resolve_ambiguity=False, tree_class=CustomTree).transform(self.forest)
        TreeChecker().visit(tree)

    def test_token_calls(self):
        if False:
            print('Hello World!')
        visited = [False] * 4

        class CustomTransformer(TreeForestTransformer):

            def A(self, node):
                if False:
                    while True:
                        i = 10
                assert node.type == 'A'
                visited[0] = True

            def B(self, node):
                if False:
                    while True:
                        i = 10
                assert node.type == 'B'
                visited[1] = True

            def C(self, node):
                if False:
                    while True:
                        i = 10
                assert node.type == 'C'
                visited[2] = True

            def D(self, node):
                if False:
                    return 10
                assert node.type == 'D'
                visited[3] = True
        tree = CustomTransformer(resolve_ambiguity=False).transform(self.forest)
        assert visited == [True] * 4

    def test_default_token(self):
        if False:
            while True:
                i = 10
        token_count = [0]

        class CustomTransformer(TreeForestTransformer):

            def __default_token__(self, node):
                if False:
                    print('Hello World!')
                token_count[0] += 1
                assert isinstance(node, Token)
        tree = CustomTransformer(resolve_ambiguity=True).transform(self.forest)
        self.assertEqual(token_count[0], 4)

    def test_rule_calls(self):
        if False:
            return 10
        visited_start = [False]
        visited_ab = [False]
        visited_bc = [False]
        visited_cd = [False]

        class CustomTransformer(TreeForestTransformer):

            def start(self, data):
                if False:
                    print('Hello World!')
                visited_start[0] = True

            def ab(self, data):
                if False:
                    i = 10
                    return i + 15
                visited_ab[0] = True

            def bc(self, data):
                if False:
                    return 10
                visited_bc[0] = True

            def cd(self, data):
                if False:
                    while True:
                        i = 10
                visited_cd[0] = True
        tree = CustomTransformer(resolve_ambiguity=False).transform(self.forest)
        self.assertTrue(visited_start[0])
        self.assertTrue(visited_ab[0])
        self.assertTrue(visited_bc[0])
        self.assertTrue(visited_cd[0])

    def test_default_rule(self):
        if False:
            return 10
        rule_count = [0]

        class CustomTransformer(TreeForestTransformer):

            def __default__(self, name, data):
                if False:
                    while True:
                        i = 10
                rule_count[0] += 1
        tree = CustomTransformer(resolve_ambiguity=True).transform(self.forest)
        self.assertEqual(rule_count[0], 4)

    def test_default_ambig(self):
        if False:
            return 10
        ambig_count = [0]

        class CustomTransformer(TreeForestTransformer):

            def __default_ambig__(self, name, data):
                if False:
                    i = 10
                    return i + 15
                if len(data) > 1:
                    ambig_count[0] += 1
        tree = CustomTransformer(resolve_ambiguity=False).transform(self.forest)
        self.assertEqual(ambig_count[0], 1)

    def test_handles_ambiguity(self):
        if False:
            return 10

        class CustomTransformer(TreeForestTransformer):

            @handles_ambiguity
            def start(self, data):
                if False:
                    while True:
                        i = 10
                assert isinstance(data, list)
                assert len(data) == 4
                for tree in data:
                    assert tree.data == 'start'
                return 'handled'

            @handles_ambiguity
            def ab(self, data):
                if False:
                    print('Hello World!')
                assert isinstance(data, list)
                assert len(data) == 1
                assert data[0].data == 'ab'
        tree = CustomTransformer(resolve_ambiguity=False).transform(self.forest)
        self.assertEqual(tree, 'handled')

    def test_discard(self):
        if False:
            return 10

        class CustomTransformer(TreeForestTransformer):

            def bc(self, data):
                if False:
                    while True:
                        i = 10
                return Discard

            def D(self, node):
                if False:
                    print('Hello World!')
                return Discard

        class TreeChecker(Transformer):

            def bc(self, children):
                if False:
                    return 10
                assert False

            def D(self, token):
                if False:
                    while True:
                        i = 10
                assert False
        tree = CustomTransformer(resolve_ambiguity=False).transform(self.forest)
        TreeChecker(visit_tokens=True).transform(tree)

    def test_aliases(self):
        if False:
            i = 10
            return i + 15
        visited_ambiguous = [False]
        visited_full = [False]

        class CustomTransformer(TreeForestTransformer):

            @handles_ambiguity
            def start(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                for tree in data:
                    assert tree.data == 'ambiguous' or tree.data == 'full'

            def ambiguous(self, data):
                if False:
                    return 10
                visited_ambiguous[0] = True
                assert len(data) == 3
                assert data[0].data == 'ab'
                assert data[1].data == 'bc'
                assert data[2].data == 'cd'
                return self.tree_class('ambiguous', data)

            def full(self, data):
                if False:
                    print('Hello World!')
                visited_full[0] = True
                assert len(data) == 1
                assert data[0].data == 'abcd'
                return self.tree_class('full', data)
        grammar = '\n        start: ab bc cd -> ambiguous\n            | abcd -> full\n        !ab: "A" "B"?\n        !bc: "B"? "C"?\n        !cd: "C"? "D"\n        !abcd: "ABCD"\n        '
        l = Lark(grammar, parser='earley', ambiguity='forest')
        forest = l.parse('ABCD')
        tree = CustomTransformer(resolve_ambiguity=False).transform(forest)
        self.assertTrue(visited_ambiguous[0])
        self.assertTrue(visited_full[0])

    def test_transformation(self):
        if False:
            while True:
                i = 10

        class CustomTransformer(TreeForestTransformer):

            def __default__(self, name, data):
                if False:
                    return 10
                result = []
                for item in data:
                    if isinstance(item, list):
                        result += item
                    else:
                        result.append(item)
                return result

            def __default_token__(self, node):
                if False:
                    print('Hello World!')
                return node.lower()

            def __default_ambig__(self, name, data):
                if False:
                    while True:
                        i = 10
                return data[0]
        result = CustomTransformer(resolve_ambiguity=False).transform(self.forest)
        expected = ['a', 'b', 'c', 'd']
        self.assertEqual(result, expected)
if __name__ == '__main__':
    unittest.main()