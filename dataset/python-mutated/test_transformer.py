"""Tests for templates module."""
import re
import gast
import unittest
from nvidia.dali._autograph.pyct import anno
from nvidia.dali._autograph.pyct import origin_info
from nvidia.dali._autograph.pyct import parser
from nvidia.dali._autograph.pyct import transformer

class TransformerTest(unittest.TestCase):

    def _simple_context(self):
        if False:
            print('Hello World!')
        entity_info = transformer.EntityInfo(name='Test_fn', source_code=None, source_file=None, future_features=(), namespace=None)
        return transformer.Context(entity_info, None, None)

    def assertSameAnno(self, first, second, key):
        if False:
            i = 10
            return i + 15
        self.assertIs(anno.getanno(first, key), anno.getanno(second, key))

    def assertDifferentAnno(self, first, second, key):
        if False:
            while True:
                i = 10
        self.assertIsNot(anno.getanno(first, key), anno.getanno(second, key))

    def test_state_tracking(self):
        if False:
            return 10

        class LoopState(object):
            pass

        class CondState(object):
            pass

        class TestTransformer(transformer.Base):

            def visit(self, node):
                if False:
                    print('Hello World!')
                anno.setanno(node, 'loop_state', self.state[LoopState].value)
                anno.setanno(node, 'cond_state', self.state[CondState].value)
                return super(TestTransformer, self).visit(node)

            def visit_While(self, node):
                if False:
                    while True:
                        i = 10
                self.state[LoopState].enter()
                node = self.generic_visit(node)
                self.state[LoopState].exit()
                return node

            def visit_If(self, node):
                if False:
                    while True:
                        i = 10
                self.state[CondState].enter()
                node = self.generic_visit(node)
                self.state[CondState].exit()
                return node
        tr = TestTransformer(self._simple_context())

        def test_function(a):
            if False:
                for i in range(10):
                    print('nop')
            a = 1
            while a:
                _ = 'a'
                if a > 2:
                    _ = 'b'
                    while True:
                        raise '1'
                if a > 3:
                    _ = 'c'
                    while True:
                        raise '1'
        (node, _) = parser.parse_entity(test_function, future_features=())
        node = tr.visit(node)
        fn_body = node.body
        outer_while_body = fn_body[1].body
        self.assertSameAnno(fn_body[0], outer_while_body[0], 'cond_state')
        self.assertDifferentAnno(fn_body[0], outer_while_body[0], 'loop_state')
        first_if_body = outer_while_body[1].body
        self.assertDifferentAnno(outer_while_body[0], first_if_body[0], 'cond_state')
        self.assertSameAnno(outer_while_body[0], first_if_body[0], 'loop_state')
        first_inner_while_body = first_if_body[1].body
        self.assertSameAnno(first_if_body[0], first_inner_while_body[0], 'cond_state')
        self.assertDifferentAnno(first_if_body[0], first_inner_while_body[0], 'loop_state')
        second_if_body = outer_while_body[2].body
        self.assertDifferentAnno(first_if_body[0], second_if_body[0], 'cond_state')
        self.assertSameAnno(first_if_body[0], second_if_body[0], 'loop_state')
        second_inner_while_body = second_if_body[1].body
        self.assertDifferentAnno(first_inner_while_body[0], second_inner_while_body[0], 'cond_state')
        self.assertDifferentAnno(first_inner_while_body[0], second_inner_while_body[0], 'loop_state')

    def test_state_tracking_context_manager(self):
        if False:
            while True:
                i = 10

        class CondState(object):
            pass

        class TestTransformer(transformer.Base):

            def visit(self, node):
                if False:
                    while True:
                        i = 10
                anno.setanno(node, 'cond_state', self.state[CondState].value)
                return super(TestTransformer, self).visit(node)

            def visit_If(self, node):
                if False:
                    return 10
                with self.state[CondState]:
                    return self.generic_visit(node)
        tr = TestTransformer(self._simple_context())

        def test_function(a):
            if False:
                print('Hello World!')
            a = 1
            if a > 2:
                _ = 'b'
                if a < 5:
                    _ = 'c'
                _ = 'd'
        (node, _) = parser.parse_entity(test_function, future_features=())
        node = tr.visit(node)
        fn_body = node.body
        outer_if_body = fn_body[1].body
        self.assertDifferentAnno(fn_body[0], outer_if_body[0], 'cond_state')
        self.assertSameAnno(outer_if_body[0], outer_if_body[2], 'cond_state')
        inner_if_body = outer_if_body[1].body
        self.assertDifferentAnno(inner_if_body[0], outer_if_body[0], 'cond_state')

    def test_visit_block_postprocessing(self):
        if False:
            i = 10
            return i + 15

        class TestTransformer(transformer.Base):

            def _process_body_item(self, node):
                if False:
                    return 10
                if isinstance(node, gast.Assign) and node.value.id == 'y':
                    if_node = gast.If(gast.Name('x', ctx=gast.Load(), annotation=None, type_comment=None), [node], [])
                    return (if_node, if_node.body)
                return (node, None)

            def visit_FunctionDef(self, node):
                if False:
                    while True:
                        i = 10
                node.body = self.visit_block(node.body, after_visit=self._process_body_item)
                return node

        def test_function(x, y):
            if False:
                for i in range(10):
                    print('nop')
            z = x
            z = y
            return z
        tr = TestTransformer(self._simple_context())
        (node, _) = parser.parse_entity(test_function, future_features=())
        node = tr.visit(node)
        self.assertEqual(len(node.body), 2)
        self.assertIsInstance(node.body[0], gast.Assign)
        self.assertIsInstance(node.body[1], gast.If)
        self.assertIsInstance(node.body[1].body[0], gast.Assign)
        self.assertIsInstance(node.body[1].body[1], gast.Return)

    def test_robust_error_on_list_visit(self):
        if False:
            for i in range(10):
                print('nop')

        class BrokenTransformer(transformer.Base):

            def visit_If(self, node):
                if False:
                    print('Hello World!')
                self.visit(node.body)
                return node

        def test_function(x):
            if False:
                return 10
            if x > 0:
                return x
        tr = BrokenTransformer(self._simple_context())
        (node, _) = parser.parse_entity(test_function, future_features=())
        with self.assertRaises(ValueError) as cm:
            node = tr.visit(node)
        obtained_message = str(cm.exception)
        expected_message = 'expected "ast.AST", got "\\<(type|class) \\\'list\\\'\\>"'
        self.assertRegex(obtained_message, expected_message)

    def test_robust_error_on_ast_corruption(self):
        if False:
            print('Hello World!')

        class NotANode(object):
            pass

        class BrokenTransformer(transformer.Base):

            def visit_If(self, node):
                if False:
                    i = 10
                    return i + 15
                node.body = NotANode()
                raise ValueError('I blew up')

        def test_function(x):
            if False:
                return 10
            if x > 0:
                return x
        tr = BrokenTransformer(self._simple_context())
        (node, _) = parser.parse_entity(test_function, future_features=())
        with self.assertRaises(ValueError) as cm:
            node = tr.visit(node)
        obtained_message = str(cm.exception)
        expected_substring = 'I blew up'
        self.assertIn(expected_substring, obtained_message)

    def test_origin_info_propagated_to_new_nodes(self):
        if False:
            print('Hello World!')

        class TestTransformer(transformer.Base):

            def visit_If(self, node):
                if False:
                    i = 10
                    return i + 15
                return gast.Pass()
        tr = TestTransformer(self._simple_context())

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            x = 1
            if x > 0:
                x = 1
            return x
        (node, source) = parser.parse_entity(test_fn, future_features=())
        origin_info.resolve(node, source, 'test_file', 100, 0)
        node = tr.visit(node)
        created_pass_node = node.body[1]
        self.assertEqual(anno.getanno(created_pass_node, anno.Basic.ORIGIN).loc.lineno, 102)

    def test_origin_info_preserved_in_moved_nodes(self):
        if False:
            for i in range(10):
                print('nop')

        class TestTransformer(transformer.Base):

            def visit_If(self, node):
                if False:
                    print('Hello World!')
                return node.body
        tr = TestTransformer(self._simple_context())

        def test_fn():
            if False:
                for i in range(10):
                    print('nop')
            x = 1
            if x > 0:
                x = 1
                x += 3
            return x
        (node, source) = parser.parse_entity(test_fn, future_features=())
        origin_info.resolve(node, source, 'test_file', 100, 0)
        node = tr.visit(node)
        assign_node = node.body[1]
        aug_assign_node = node.body[2]
        self.assertEqual(anno.getanno(assign_node, anno.Basic.ORIGIN).loc.lineno, 103)
        self.assertEqual(anno.getanno(aug_assign_node, anno.Basic.ORIGIN).loc.lineno, 104)

class CodeGeneratorTest(unittest.TestCase):

    def _simple_context(self):
        if False:
            while True:
                i = 10
        entity_info = transformer.EntityInfo(name='test_fn', source_code=None, source_file=None, future_features=(), namespace=None)
        return transformer.Context(entity_info, None, None)

    def test_basic_codegen(self):
        if False:
            return 10

        class TestCodegen(transformer.CodeGenerator):

            def visit_Assign(self, node):
                if False:
                    for i in range(10):
                        print('nop')
                self.emit(parser.unparse(node, include_encoding_marker=False))
                self.emit('\n')

            def visit_Return(self, node):
                if False:
                    i = 10
                    return i + 15
                self.emit(parser.unparse(node, include_encoding_marker=False))
                self.emit('\n')

            def visit_If(self, node):
                if False:
                    i = 10
                    return i + 15
                self.emit('if ')
                self.emit(parser.unparse(node.test, include_encoding_marker=False))
                self.emit(' {\n')
                self.visit_block(node.body)
                self.emit('} else {\n')
                self.visit_block(node.orelse)
                self.emit('}\n')
        tg = TestCodegen(self._simple_context())

        def test_fn():
            if False:
                return 10
            x = 1
            if x > 0:
                x = 2
                if x > 1:
                    x = 3
            return x
        (node, source) = parser.parse_entity(test_fn, future_features=())
        origin_info.resolve(node, source, 'test_file', 100, 0)
        tg.visit(node)
        r = re.compile('.*'.join(['x = 1', 'if \\(?x > 0\\)? {', 'x = 2', 'if \\(?x > 1\\)? {', 'x = 3', '} else {', '}', '} else {', '}', 'return x']), re.DOTALL)
        self.assertRegex(tg.code_buffer, r)