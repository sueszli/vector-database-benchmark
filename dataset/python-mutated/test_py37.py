import __future__
from compiler.consts import CO_NOFREE
from compiler.pycodegen import CodeGenerator
from .common import CompilerTest
LOAD_METHOD = 'LOAD_METHOD'
CALL_METHOD = 'CALL_METHOD'

class Python37Tests(CompilerTest):

    def test_compile_method(self):
        if False:
            print('Hello World!')
        code = self.compile('x.f()')
        self.assertInBytecode(code, LOAD_METHOD)
        self.assertInBytecode(code, CALL_METHOD, 0)
        code = self.compile('x.f(42)')
        self.assertInBytecode(code, LOAD_METHOD)
        self.assertInBytecode(code, CALL_METHOD, 1)

    def test_compile_method_varargs(self):
        if False:
            while True:
                i = 10
        code = self.compile('x.f(*foo)')
        self.assertNotInBytecode(code, LOAD_METHOD)

    def test_compile_method_kwarg(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile('x.f(kwarg=1)')
        self.assertNotInBytecode(code, LOAD_METHOD)

    def test_compile_method_normal(self):
        if False:
            return 10
        code = self.compile('f()')
        self.assertNotInBytecode(code, LOAD_METHOD)

    def test_future_gen_stop(self):
        if False:
            return 10
        code = self.compile('from __future__ import generator_stop')
        self.assertEqual(code.co_flags, CO_NOFREE)

    def test_future_annotations_flag(self):
        if False:
            while True:
                i = 10
        code = self.compile('from __future__ import annotations')
        self.assertEqual(code.co_flags, CO_NOFREE | __future__.CO_FUTURE_ANNOTATIONS)

    def test_async_aiter(self):
        if False:
            while True:
                i = 10
        outer_graph = self.to_graph('\n            async def f():\n                async for x in y:\n                    pass\n        ')
        for outer_instr in self.graph_to_instrs(outer_graph):
            if outer_instr.opname == 'LOAD_CONST' and isinstance(outer_instr.oparg, CodeGenerator):
                saw_aiter = False
                for instr in self.graph_to_instrs(outer_instr.oparg.graph):
                    if saw_aiter:
                        self.assertNotEqual(instr.opname, 'LOAD_CONST')
                        break
                    if instr.opname == 'GET_AITER':
                        saw_aiter = True
                break

    def test_try_except_pop_except(self):
        if False:
            i = 10
            return i + 15
        'POP_EXCEPT moved after POP_BLOCK in Python 3.10'
        graph = self.to_graph('\n            try:\n                pass\n            except Exception as e:\n                pass\n        ')
        prev_instr = None
        for instr in self.graph_to_instrs(graph):
            if instr.opname == 'POP_EXCEPT':
                self.assertEqual(prev_instr.opname, 'POP_BLOCK', prev_instr.opname)
            prev_instr = instr

    def test_future_annotations(self):
        if False:
            print('Hello World!')
        annotations = ['42']
        for annotation in annotations:
            code = self.compile(f'from __future__ import annotations\ndef f() -> {annotation}:\n    pass')
            self.assertInBytecode(code, 'LOAD_CONST', ('return', annotation))
        self.assertEqual(code.co_flags, CO_NOFREE | __future__.CO_FUTURE_ANNOTATIONS)

    def test_circular_import_as(self):
        if False:
            for i in range(10):
                print('nop')
        'verifies that we emit an IMPORT_FROM to enable circular imports\n        when compiling an absolute import to verify that they can support\n        circular imports'
        code = self.compile(f'import x.y as b')
        self.assertInBytecode(code, 'IMPORT_FROM')
        self.assertNotInBytecode(code, 'LOAD_ATTR')

    def test_compile_opt_unary_jump(self):
        if False:
            print('Hello World!')
        graph = self.to_graph('if not abc: foo')
        self.assertNotInGraph(graph, 'POP_JUMP_IF_FALSE')

    def test_compile_opt_bool_or_jump(self):
        if False:
            i = 10
            return i + 15
        graph = self.to_graph('if abc or bar: foo')
        self.assertNotInGraph(graph, 'JUMP_IF_TRUE_OR_POP')

    def test_compile_opt_bool_and_jump(self):
        if False:
            i = 10
            return i + 15
        graph = self.to_graph('if abc and bar: foo')
        self.assertNotInGraph(graph, 'JUMP_IF_FALSE_OR_POP')

    def test_compile_opt_assert_or_bool(self):
        if False:
            print('Hello World!')
        graph = self.to_graph('assert abc or bar')
        self.assertNotInGraph(graph, 'JUMP_IF_TRUE_OR_POP')

    def test_compile_opt_assert_and_bool(self):
        if False:
            i = 10
            return i + 15
        graph = self.to_graph('assert abc and bar')
        self.assertNotInGraph(graph, 'JUMP_IF_FALSE_OR_POP')

    def test_compile_opt_if_exp(self):
        if False:
            while True:
                i = 10
        graph = self.to_graph('assert not a if c else b')
        self.assertNotInGraph(graph, 'UNARY_NOT')

    def test_compile_opt_cmp_op(self):
        if False:
            while True:
                i = 10
        graph = self.to_graph('assert not a > b')
        self.assertNotInGraph(graph, 'UNARY_NOT')

    def test_compile_opt_chained_cmp_op(self):
        if False:
            for i in range(10):
                print('nop')
        graph = self.to_graph('assert not a > b > c')
        self.assertNotInGraph(graph, 'UNARY_NOT')