import pickle
from pytype import config
from pytype import load_pytd
from pytype.imports import pickle_utils
from pytype.platform_utils import path_utils
from pytype.pytd import pytd_utils
from pytype.pytd import serialize_ast
from pytype.pytd import visitors
from pytype.tests import test_base
from pytype.tests import test_utils
import unittest

class SerializeAstTest(test_base.UnitTest):

    def _store_ast(self, temp_dir, module_name, pickled_ast_filename, ast=None, loader=None, src_path=None, metadata=None):
        if False:
            print('Hello World!')
        if not (ast and loader):
            (ast, loader) = self._get_ast(temp_dir=temp_dir, module_name=module_name)
        pickle_utils.StoreAst(ast, pickled_ast_filename, src_path=src_path, metadata=metadata)
        module_map = {name: module.ast for (name, module) in loader._modules.items()}
        return module_map

    def _get_ast(self, temp_dir, module_name, src=None):
        if False:
            return 10
        src = src or '\n        import module2\n        from module2 import f\n        from typing import List\n\n        constant = True\n\n        x = List[int]\n        b = List[int]\n\n        class SomeClass:\n          def __init__(self, a: module2.ObjectMod2):\n            pass\n\n        def ModuleFunction():\n          pass\n    '
        pyi_filename = temp_dir.create_file('module1.pyi', src)
        temp_dir.create_file('module2.pyi', '\n        import queue\n        def f() -> queue.Queue: ...\n        class ObjectMod2:\n          def __init__(self):\n            pass\n    ')
        loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=temp_dir.path))
        ast = loader.load_file(module_name, pyi_filename)
        loader._modules[module_name].ast = ast = ast.Visit(visitors.CanonicalOrderingVisitor())
        return (ast, loader)

    def test_find_class_types_visitor(self):
        if False:
            i = 10
            return i + 15
        module_name = 'foo.bar'
        with test_utils.Tempdir() as d:
            (ast, _) = self._get_ast(temp_dir=d, module_name=module_name)
        indexer = serialize_ast.FindClassTypesVisitor()
        ast.Visit(indexer)
        self.assertEqual(len(indexer.class_type_nodes), 10)

    def test_node_index_visitor_usage(self):
        if False:
            return 10
        'Confirms that the node index is used.\n\n    This removes the first node from the class_type_nodes list and checks that\n    that node is not updated by ProcessAst.\n    '
        with test_utils.Tempdir() as d:
            module_name = 'module1'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, module_name, pickled_ast_filename)
            del module_map[module_name]
            serialized_ast = pickle_utils.LoadPickle(pickled_ast_filename)
            serialized_ast = serialized_ast.Replace(class_type_nodes=sorted(serialized_ast.class_type_nodes)[1:])
            loaded_ast = serialize_ast.ProcessAst(serialized_ast, module_map)
            with self.assertRaisesRegex(ValueError, "Unresolved class: 'builtins.NoneType'"):
                loaded_ast.Visit(visitors.VerifyLookup())

    def test_pickle(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            (ast, _) = self._get_ast(temp_dir=d, module_name='foo.bar.module1')
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            result = pickle_utils.StoreAst(ast, pickled_ast_filename)
            self.assertIsNone(result)
            with open(pickled_ast_filename, 'rb') as fi:
                serialized_ast = pickle.load(fi)
            self.assertTrue(serialized_ast.ast)
            self.assertCountEqual(dict(serialized_ast.dependencies), ['builtins', 'foo.bar.module1', 'module2', 'queue'])

    def test_unrestorable_child(self):
        if False:
            while True:
                i = 10

        class RenameVisitor(visitors.Visitor):

            def __init__(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__(*args, **kwargs)
                self._init = False

            def EnterFunction(self, func):
                if False:
                    print('Hello World!')
                if func.name == '__init__':
                    self._init = True
                    return None
                return False

            def LeaveFunction(self, func):
                if False:
                    print('Hello World!')
                self._init = False

            def VisitClassType(self, cls_type):
                if False:
                    for i in range(10):
                        print('nop')
                if self._init:
                    cls_type = cls_type.Replace(name='other_module.unknown_Reference')
                    cls_type.cls = None
                return cls_type
        with test_utils.Tempdir() as d:
            src = '\n        import other_module\n        x = other_module.UnusedReferenceNeededToKeepTheImport\n\n        class SomeClass:\n          def __init__(will_be_replaced_with_visitor) -> None:\n            pass\n\n        def func(a:SomeClass) -> None:\n          pass\n      '
            d.create_file('other_module.pyi', '\n          from typing import Any\n          def __getattr__(self, name) -> Any: ...')
            (ast, loader) = self._get_ast(temp_dir=d, module_name='module1', src=src)
            ast = ast.Visit(RenameVisitor())
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, 'module1', pickled_ast_filename, ast=ast, loader=loader)
            del module_map['module1']
            serialized_ast = pickle_utils.LoadPickle(pickled_ast_filename)
            loaded_ast = serialize_ast.ProcessAst(serialized_ast, module_map)
            cls = loaded_ast.functions[0].signatures[0].params[0].type.cls
            cls.Visit(visitors.VerifyLookup())

    def test_load_top_level(self):
        if False:
            return 10
        'Tests that a pickled file can be read.'
        with test_utils.Tempdir() as d:
            module_name = 'module1'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, module_name, pickled_ast_filename)
            original_ast = module_map[module_name]
            del module_map[module_name]
            loaded_ast = serialize_ast.ProcessAst(pickle_utils.LoadPickle(pickled_ast_filename), module_map)
            self.assertTrue(loaded_ast)
            self.assertIsNot(loaded_ast, original_ast)
            self.assertEqual(loaded_ast.name, module_name)
            self.assertTrue(pytd_utils.ASTeq(original_ast, loaded_ast))
            loaded_ast.Visit(visitors.VerifyLookup())

    def test_load_with_same_module_name(self):
        if False:
            return 10
        'Explicitly set the module name and reload with the same name.\n\n    The difference to testLoadTopLevel is that the module name does not match\n    the filelocation.\n    '
        with test_utils.Tempdir() as d:
            module_name = 'foo.bar.module1'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, module_name, pickled_ast_filename)
            original_ast = module_map[module_name]
            del module_map[module_name]
            loaded_ast = serialize_ast.ProcessAst(pickle_utils.LoadPickle(pickled_ast_filename), module_map)
            self.assertTrue(loaded_ast)
            self.assertIsNot(loaded_ast, original_ast)
            self.assertEqual(loaded_ast.name, 'foo.bar.module1')
            self.assertTrue(pytd_utils.ASTeq(original_ast, loaded_ast))
            loaded_ast.Visit(visitors.VerifyLookup())

    def test_unrestorable_dependency_error_with_module_index(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            module_name = 'module1'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, module_name, pickled_ast_filename)
            module_map = {}
            with self.assertRaises(serialize_ast.UnrestorableDependencyError):
                serialize_ast.ProcessAst(pickle_utils.LoadPickle(pickled_ast_filename), module_map)

    def test_unrestorable_dependency_error_without_module_index(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            module_name = 'module1'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, module_name, pickled_ast_filename)
            module_map = {}
            loaded_ast = pickle_utils.LoadPickle(pickled_ast_filename)
            loaded_ast.modified_class_types = None
            with self.assertRaises(serialize_ast.UnrestorableDependencyError):
                serialize_ast.ProcessAst(loaded_ast, module_map)

    def test_load_with_different_module_name(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            original_module_name = 'module1'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, original_module_name, pickled_ast_filename)
            original_ast = module_map[original_module_name]
            del module_map[original_module_name]
            new_module_name = 'wurstbrot.module2'
            serializable_ast = pickle_utils.LoadPickle(pickled_ast_filename)
            serializable_ast = serialize_ast.EnsureAstName(serializable_ast, new_module_name, fix=True)
            loaded_ast = serialize_ast.ProcessAst(serializable_ast, module_map)
            self.assertTrue(loaded_ast)
            self.assertIsNot(loaded_ast, original_ast)
            self.assertEqual(loaded_ast.name, new_module_name)
            loaded_ast.Visit(visitors.VerifyLookup())
            self.assertFalse(pytd_utils.ASTeq(original_ast, loaded_ast))
            (ast_new_module, _) = self._get_ast(temp_dir=d, module_name=new_module_name)
            self.assertTrue(pytd_utils.ASTeq(ast_new_module, loaded_ast))

    def test_store_removes_init(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            original_module_name = 'module1.__init__'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            module_map = self._store_ast(d, original_module_name, pickled_ast_filename, src_path='module1/__init__.py')
            serializable_ast = pickle_utils.LoadPickle(pickled_ast_filename)
            expected_name = 'module1'
            self.assertIn(original_module_name, module_map)
            self.assertNotIn(expected_name, module_map)
            self.assertEqual(serializable_ast.ast.name, expected_name)

    def test_function(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            foo = d.create_file('foo.pickle')
            module_map = self._store_ast(d, 'foo', foo, ast=self._get_ast(d, 'foo'))
            p = pickle_utils.LoadPickle(foo)
            ast = serialize_ast.ProcessAst(p, module_map)
            (f,) = (a for a in ast.aliases if a.name == 'foo.f')
            (signature,) = f.type.signatures
            self.assertIsNotNone(signature.return_type.cls)

    def test_pickle_metadata(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            module_name = 'module1'
            pickled_ast_filename = path_utils.join(d.path, 'module1.pyi.pickled')
            self._store_ast(d, module_name, pickled_ast_filename, metadata=['meta', 'data'])
            serialized_ast = pickle_utils.LoadPickle(pickled_ast_filename)
            self.assertSequenceEqual(serialized_ast.metadata, ['meta', 'data'])
if __name__ == '__main__':
    unittest.main()