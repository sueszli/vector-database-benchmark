import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_jit.py TestPythonBindings\n\ninstead.')

class TestPythonBindings(JitTestCase):

    def test_cu_get_functions(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def test_get_python_cu_fn(x: torch.Tensor):
            if False:
                i = 10
                return i + 15
            return 2 * x
        cu = torch.jit._state._python_cu
        self.assertTrue('test_get_python_cu_fn' in (str(fn.name) for fn in cu.get_functions()))

    def test_cu_create_function(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(x: torch.Tensor):
            if False:
                print('Hello World!')
            return 2 * x
        cu = torch._C.CompilationUnit()
        cu.create_function('test_fn', fn.graph)
        inp = torch.randn(5)
        self.assertEqual(inp * 2, cu.find_function('test_fn')(inp))
        self.assertEqual(cu.find_function('doesnt_exist'), None)
        self.assertEqual(inp * 2, cu.test_fn(inp))
        with self.assertRaises(AttributeError):
            cu.doesnt_exist(inp)

    def test_invalidation(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def test_invalidation_fn(x: torch.Tensor):
            if False:
                print('Hello World!')
            return 2 * x
        gr = test_invalidation_fn.graph.copy()
        n = gr.insertNode(gr.create('prim::profile'))
        v = n.output()
        str((n, v))
        torch._C._jit_pass_dce(gr)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            str(n)
        with self.assertRaisesRegex(RuntimeError, 'invalidated'):
            str(v)

    def test_graph_iterator_keepalive(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def test_iterator_keepalive_fn(x: torch.Tensor):
            if False:
                return 10
            return 2 * x
        n = test_iterator_keepalive_fn.inlined_graph.nodes()
        list(n)
        i = test_iterator_keepalive_fn.inlined_graph.inputs()
        list(i)
        o = test_iterator_keepalive_fn.inlined_graph.outputs()
        list(o)

    def test_aliasdb(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def test_aliasdb_fn(x: torch.Tensor):
            if False:
                return 10
            return 2 * x
        gr = test_aliasdb_fn.graph.copy()
        alias_db = gr.alias_db()
        self.assertTrue('WILDCARD' in str(alias_db))
        self.assertTrue('digraph alias_db' in alias_db.to_graphviz_str())

    def test_graph_create(self):
        if False:
            return 10
        gr = torch._C.Graph()
        with self.assertRaises(ValueError):
            gr.create('prim::Constant', [None])

    def test_add_input(self):
        if False:
            print('Hello World!')
        gr = torch._C.Graph()
        foo_value = gr.addInput('foo')
        assert foo_value in gr.inputs()

    def test_canonicalize(self):
        if False:
            for i in range(10):
                print('nop')
        ir = '\ngraph(%p207 : Tensor,\n      %1 : Tensor,\n      %p407 : int):\n  %11 : Tensor = aten::view_expand_placeholder(%1)\n  %12 : Tensor = aten::pointwise_placeholder(%11, %p207, %p407)\n  %13 : Tensor = aten::view_expand_placeholder(%12)\n  %14 : Tensor = aten::pointwise_placeholder(%13)\n  return (%14)\n        '
        graph1 = torch._C.parse_ir(ir)
        graph1 = torch._C._jit_pass_canonicalize(graph1, True)
        graph2 = torch._C.parse_ir(ir)
        graph2 = torch._C._jit_pass_canonicalize(graph2)
        self.assertEqual(str(graph1), str(graph2))
        FileCheck().check('%p207').check_not('%14').run(graph1)
        graph3 = torch._C.parse_ir(ir)
        graph3 = torch._C._jit_pass_canonicalize(graph3, False)
        FileCheck().check_not('%p207').run(graph3)