from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Symtab import ModuleScope
from Cython.TestUtils import TransformTest
from Cython.Compiler.Visitor import MethodDispatcherTransform
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, AnalyseDeclarationsTransform, AnalyseExpressionsTransform, InterpretCompilerDirectives

class TestMethodDispatcherTransform(TransformTest):
    _tree = None

    def _build_tree(self):
        if False:
            while True:
                i = 10
        if self._tree is None:
            context = None

            def fake_module(node):
                if False:
                    print('Hello World!')
                scope = ModuleScope('test', None, None)
                return ModuleNode(node.pos, doc=None, body=node, scope=scope, full_module_name='test', directive_comments={})
            pipeline = [fake_module, NormalizeTree(context), InterpretCompilerDirectives(context, {}), AnalyseDeclarationsTransform(context), AnalyseExpressionsTransform(context)]
            self._tree = self.run_pipeline(pipeline, u"\n                cdef bytes s = b'asdfg'\n                cdef dict d = {1:2}\n                x = s * 3\n                d.get('test')\n            ")
        return self._tree

    def test_builtin_method(self):
        if False:
            print('Hello World!')
        calls = [0]

        class Test(MethodDispatcherTransform):

            def _handle_simple_method_dict_get(self, node, func, args, unbound):
                if False:
                    for i in range(10):
                        print('nop')
                calls[0] += 1
                return node
        tree = self._build_tree()
        Test(None)(tree)
        self.assertEqual(1, calls[0])

    def test_binop_method(self):
        if False:
            while True:
                i = 10
        calls = {'bytes': 0, 'object': 0}

        class Test(MethodDispatcherTransform):

            def _handle_simple_method_bytes___mul__(self, node, func, args, unbound):
                if False:
                    print('Hello World!')
                calls['bytes'] += 1
                return node

            def _handle_simple_method_object___mul__(self, node, func, args, unbound):
                if False:
                    while True:
                        i = 10
                calls['object'] += 1
                return node
        tree = self._build_tree()
        Test(None)(tree)
        self.assertEqual(1, calls['bytes'])
        self.assertEqual(0, calls['object'])